from __future__ import annotations

from contextlib import contextmanager
import errno
import os
from pathlib import Path
import select
import stat
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import hermes_state_maintenance as maintenance


_REPOSITORY_ROOT = Path(__file__).parent.parent
_HOLDER_SCRIPT = r"""
import sys
from pathlib import Path

import hermes_state_maintenance as maintenance

profile = Path(sys.argv[1])
acquire = (
    maintenance.acquire_profile_state_shared
    if sys.argv[2] == "shared"
    else maintenance.acquire_profile_state_exclusive
)
lease = acquire(profile, timeout_seconds=5.0)
print("held", flush=True)
sys.stdin.readline()
lease.close()
"""
_WAITING_EXCLUSIVE_SCRIPT = r"""
import sys
import time
from pathlib import Path

import hermes_state_maintenance as maintenance

profile = Path(sys.argv[1])
print("ready", flush=True)
sys.stdin.readline()
print("attempting", flush=True)
started = time.monotonic()
lease = maintenance.acquire_profile_state_exclusive(
    profile, timeout_seconds=5.0
)
elapsed = time.monotonic() - started
print("waited" if elapsed >= 1.5 else "immediate", flush=True)
sys.stdin.readline()
lease.close()
"""
_TIMED_SHARED_SCRIPT = r"""
import sys
from pathlib import Path

import hermes_state_maintenance as maintenance

try:
    lease = maintenance.acquire_profile_state_shared(
        Path(sys.argv[1]), timeout_seconds=0.05
    )
except maintenance.ProfileStateMaintenanceError as exc:
    print(exc.category, flush=True)
else:
    lease.close()
    print("acquired", flush=True)
"""


def _profile(tmp_path: Path, name: str = "profile") -> Path:
    profile = tmp_path / name
    profile.mkdir(mode=0o700)
    profile.chmod(0o700)
    return profile


def _lock_path(profile: Path) -> Path:
    return profile / maintenance._PROFILE_STATE_LOCK_NAME


def _read_line(process: subprocess.Popen[str], timeout_seconds: float = 5.0) -> str:
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], timeout_seconds)
    assert readable
    return process.stdout.readline().strip()


def _spawn_holder(profile: Path, kind: str) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-c", _HOLDER_SCRIPT, os.fspath(profile), kind],
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _read_line(process) == "held"
    except BaseException:
        process.kill()
        process.wait(timeout=5.0)
        raise
    return process


def _timed_shared_attempt(profile: Path) -> str:
    process = subprocess.Popen(
        [sys.executable, "-c", _TIMED_SHARED_SCRIPT, os.fspath(profile)],
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        result = _read_line(process, timeout_seconds=1.0)
        process.wait(timeout=5.0)
        assert process.returncode == 0
        return result
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)


def _release_holder(process: subprocess.Popen[str]) -> None:
    assert process.stdin is not None
    process.stdin.write("release\n")
    process.stdin.flush()
    process.stdin.close()
    process.wait(timeout=5.0)
    assert process.returncode == 0


@contextmanager
def _held_by_process(profile: Path, kind: str):
    process = _spawn_holder(profile, kind)
    try:
        yield process
    finally:
        if process.poll() is None:
            _release_holder(process)


def _write_private_file(path: Path, payload: bytes = b"") -> None:
    fd = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
        0o600,
    )
    try:
        if payload:
            assert os.write(fd, payload) == len(payload)
        os.fsync(fd)
    finally:
        os.close(fd)
    path.chmod(0o600)


def _assert_categorical(exc: BaseException, *, category: str, profile: Path) -> None:
    assert isinstance(exc, maintenance.ProfileStateMaintenanceError)
    assert exc.category == category
    assert str(exc) == category
    assert os.fspath(profile) not in str(exc)
    assert profile.name not in str(exc)
    assert exc.__cause__ is None


def test_shared_lease_close_is_idempotent_and_lock_inode_persists(tmp_path):
    profile = _profile(tmp_path)

    first = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    assert isinstance(first, maintenance.SharedStateLease)
    first.close()
    first.close()

    entries = list(profile.iterdir())
    assert len(entries) == 1
    first_stat = entries[0].lstat()
    assert stat.S_ISREG(first_stat.st_mode)
    assert stat.S_IMODE(first_stat.st_mode) == 0o600

    second = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    second.close()
    second_stat = entries[0].lstat()
    assert (second_stat.st_dev, second_stat.st_ino) == (
        first_stat.st_dev,
        first_stat.st_ino,
    )


def test_concurrent_close_detaches_descriptors_once_before_fd_reuse(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    lease = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    lock_fd = lease._lock_fd
    real_close = os.close
    real_flock = maintenance.fcntl.flock
    reused_fds: list[int] = []
    unlock_calls: list[int] = []

    class CloseGate:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self._guard = threading.Lock()
            self._entries = 0
            self.first_entered = threading.Event()
            self.release_first = threading.Event()
            self.second_attempted = threading.Event()
            self.second_entered = threading.Event()
            self.release_second = threading.Event()

        def __enter__(self):
            with self._guard:
                self._entries += 1
                entry = self._entries
            if entry == 1:
                self._lock.acquire()
                self.first_entered.set()
                assert self.release_first.wait(timeout=1.0)
            else:
                self.second_attempted.set()
                self._lock.acquire()
                self.second_entered.set()
                assert self.release_second.wait(timeout=1.0)
            return self

        def __exit__(self, *_exc_info: object) -> None:
            self._lock.release()

    gate = CloseGate()
    lease._close_lock = gate

    def record_unlock(fd: int, operation: int) -> None:
        if operation == maintenance.fcntl.LOCK_UN:
            unlock_calls.append(fd)
        real_flock(fd, operation)

    def close_and_reuse(fd: int) -> None:
        real_close(fd)
        if fd == lock_fd and not reused_fds:
            reused_fd = os.open(os.devnull, os.O_RDONLY)
            assert reused_fd == lock_fd
            reused_fds.append(reused_fd)

    monkeypatch.setattr(maintenance.fcntl, "flock", record_unlock)
    monkeypatch.setattr(maintenance.os, "close", close_and_reuse)
    first = threading.Thread(target=lease.close)
    second = threading.Thread(target=lease.close)
    first.start()
    assert gate.first_entered.wait(timeout=1.0)
    second.start()
    assert gate.second_attempted.wait(timeout=1.0)
    gate.release_first.set()
    assert gate.second_entered.wait(timeout=1.0)
    first.join(timeout=1.0)
    assert not first.is_alive()

    gate.release_second.set()
    second.join(timeout=1.0)
    assert not second.is_alive()
    assert unlock_calls == [lock_fd]
    assert len(reused_fds) == 1
    os.fstat(reused_fds[0])
    real_close(reused_fds[0])


def test_lease_context_manager_releases_authority(tmp_path):
    profile = _profile(tmp_path)

    with maintenance.acquire_profile_state_shared(
        profile, timeout_seconds=1.0
    ) as lease:
        assert isinstance(lease, maintenance.SharedStateLease)

    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    assert isinstance(exclusive, maintenance.ExclusiveMaintenanceLease)
    exclusive.close()


def test_inflight_shared_holder_delays_exclusive_acquisition(tmp_path):
    profile = _profile(tmp_path)
    shared = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    process = subprocess.Popen(
        [sys.executable, "-c", _WAITING_EXCLUSIVE_SCRIPT, os.fspath(profile)],
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert _read_line(process) == "ready"
        assert process.stdin is not None
        process.stdin.write("go\n")
        process.stdin.flush()
        assert _read_line(process) == "attempting"
        time.sleep(2.0)
        shared.close()
        assert _read_line(process) == "waited"
        _release_holder(process)
    finally:
        shared.close()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)


def test_exclusive_holder_blocks_later_shared_acquisition(tmp_path):
    profile = _profile(tmp_path)

    with _held_by_process(profile, "exclusive"):
        assert _timed_shared_attempt(profile) == "profile_state_lease_timeout"


def test_exclusive_holder_does_not_block_another_profile(tmp_path):
    blocked_profile = _profile(tmp_path, "blocked")
    unrelated_profile = _profile(tmp_path, "unrelated")

    with _held_by_process(blocked_profile, "exclusive"):
        lease = maintenance.acquire_profile_state_shared(
            unrelated_profile, timeout_seconds=0.25
        )
        lease.close()


def test_timeout_is_categorical_and_path_free(tmp_path):
    profile = _profile(tmp_path, "do-not-disclose")

    with _held_by_process(profile, "exclusive"):
        with pytest.raises(maintenance.ProfileStateLeaseTimeout) as exc_info:
            maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.05)

    _assert_categorical(
        exc_info.value,
        category="profile_state_lease_timeout",
        profile=profile,
    )


def test_waiter_does_not_retry_flock_at_monotonic_deadline(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    initial = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    initial.close()
    real_flock = maintenance.fcntl.flock
    now = [0.0]
    attempts = 0

    def contend_then_acquire(fd: int, operation: int) -> None:
        nonlocal attempts
        if operation & (maintenance.fcntl.LOCK_SH | maintenance.fcntl.LOCK_EX):
            attempts += 1
            if attempts == 1:
                raise BlockingIOError(errno.EAGAIN, "synthetic contention")
        real_flock(fd, operation)

    monkeypatch.setattr(maintenance.fcntl, "flock", contend_then_acquire)
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        maintenance.time,
        "sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )

    with pytest.raises(maintenance.ProfileStateLeaseTimeout):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.01)

    assert attempts == 1


def test_empty_lock_initializer_does_not_retry_at_monotonic_deadline(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    _write_private_file(_lock_path(profile))
    real_flock = maintenance.fcntl.flock
    now = [0.0]
    attempts = 0

    def contend_then_acquire(fd: int, operation: int) -> None:
        nonlocal attempts
        if operation & (maintenance.fcntl.LOCK_SH | maintenance.fcntl.LOCK_EX):
            attempts += 1
            if attempts == 1:
                raise BlockingIOError(errno.EAGAIN, "synthetic contention")
        real_flock(fd, operation)

    monkeypatch.setattr(maintenance.fcntl, "flock", contend_then_acquire)
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        maintenance.time,
        "sleep",
        lambda duration: now.__setitem__(0, now[0] + duration),
    )

    with pytest.raises(maintenance.ProfileStateLeaseTimeout):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.01)

    assert attempts == 1
    assert _lock_path(profile).stat().st_size == 0


def test_repeated_eintr_cannot_retry_flock_past_deadline(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    initial = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    initial.close()
    real_flock = maintenance.fcntl.flock
    now = [0.0]
    attempts = 0

    def interrupt_then_acquire(fd: int, operation: int) -> None:
        nonlocal attempts
        if operation & (maintenance.fcntl.LOCK_SH | maintenance.fcntl.LOCK_EX):
            attempts += 1
            if attempts <= 2:
                now[0] += 0.006
                raise OSError(errno.EINTR, "synthetic interruption")
        real_flock(fd, operation)

    monkeypatch.setattr(maintenance.fcntl, "flock", interrupt_then_acquire)
    monkeypatch.setattr(maintenance.time, "monotonic", lambda: now[0])

    with pytest.raises(maintenance.ProfileStateLeaseTimeout):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.01)

    assert attempts == 2


def test_process_death_releases_kernel_lock(tmp_path):
    profile = _profile(tmp_path)
    process = _spawn_holder(profile, "exclusive")
    process.kill()
    process.wait(timeout=5.0)

    lease = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    lease.close()


@pytest.mark.require_symlinks
def test_profile_symlink_is_rejected(tmp_path):
    target = _profile(tmp_path, "target")
    profile = tmp_path / "profile-link"
    profile.symlink_to(target, target_is_directory=True)

    with pytest.raises(maintenance.UnsafeProfileState) as exc_info:
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)

    _assert_categorical(
        exc_info.value,
        category="unsafe_profile_state",
        profile=profile,
    )


@pytest.mark.require_symlinks
def test_lock_symlink_is_rejected(tmp_path):
    profile = _profile(tmp_path)
    target = profile / "other"
    _write_private_file(target)
    _lock_path(profile).symlink_to(target.name)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)


def test_profile_with_wrong_mode_is_rejected(tmp_path):
    profile = _profile(tmp_path)
    profile.chmod(0o750)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)


def test_lock_with_wrong_mode_is_rejected(tmp_path):
    profile = _profile(tmp_path)
    lease = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    lease.close()
    _lock_path(profile).chmod(0o640)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)


def test_profile_with_wrong_owner_is_rejected(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    real_euid = os.geteuid()
    monkeypatch.setattr(
        maintenance.os,
        "geteuid",
        lambda: real_euid + 1,
    )

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)


def test_lock_inode_with_wrong_owner_is_rejected(tmp_path, monkeypatch, capsys):
    profile = _profile(tmp_path, "owner-do-not-disclose")
    initial = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    initial.close()
    lock_stat = _lock_path(profile).stat()
    lock_key = (lock_stat.st_dev, lock_stat.st_ino)
    real_fstat = os.fstat
    inspected_lock = False

    def wrong_lock_owner(fd: int) -> os.stat_result:
        nonlocal inspected_lock
        value = real_fstat(fd)
        if (value.st_dev, value.st_ino) == lock_key:
            inspected_lock = True
            fields = list(value)
            fields[4] = value.st_uid + 1
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(maintenance.os, "fstat", wrong_lock_owner)

    with pytest.raises(maintenance.UnsafeProfileState) as exc_info:
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)

    assert inspected_lock
    _assert_categorical(
        exc_info.value,
        category="unsafe_profile_state",
        profile=profile,
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_non_regular_lock_is_rejected_without_blocking(tmp_path):
    profile = _profile(tmp_path)
    _lock_path(profile).mkdir(mode=0o700)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)


def test_hard_linked_lock_is_rejected(tmp_path):
    profile = _profile(tmp_path)
    lease = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    lease.close()
    os.link(_lock_path(profile), profile / "second-link")

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.1)


def test_lock_replacement_during_acquisition_is_rejected(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    initial = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    initial.close()
    lock_path = _lock_path(profile)
    original_payload = lock_path.read_bytes()
    real_flock = maintenance.fcntl.flock
    replaced = False

    def replace_before_lock(fd: int, operation: int) -> None:
        nonlocal replaced
        if not replaced and operation & (
            maintenance.fcntl.LOCK_SH | maintenance.fcntl.LOCK_EX
        ):
            replaced = True
            lock_path.rename(profile / "retired-lock")
            _write_private_file(lock_path, original_payload)
        real_flock(fd, operation)

    monkeypatch.setattr(maintenance.fcntl, "flock", replace_before_lock)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.2)


def test_replaced_lock_cannot_split_the_lock_inode(tmp_path):
    profile = _profile(tmp_path)
    first = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    lock_path = _lock_path(profile)
    original_payload = lock_path.read_bytes()
    lock_path.rename(profile / "held-lock-inode")
    _write_private_file(lock_path, original_payload)

    try:
        with pytest.raises(maintenance.UnsafeProfileState):
            maintenance.acquire_profile_state_exclusive(profile, timeout_seconds=0.2)
    finally:
        first.close()


def test_profile_substitution_during_acquisition_is_rejected(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    initial = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    initial.close()
    real_flock = maintenance.fcntl.flock
    substituted = False

    def substitute_before_lock(fd: int, operation: int) -> None:
        nonlocal substituted
        if not substituted and operation & (
            maintenance.fcntl.LOCK_SH | maintenance.fcntl.LOCK_EX
        ):
            substituted = True
            profile.rename(tmp_path / "retired-profile")
            profile.mkdir(mode=0o700)
            profile.chmod(0o700)
        real_flock(fd, operation)

    monkeypatch.setattr(maintenance.fcntl, "flock", substitute_before_lock)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=0.2)


@pytest.mark.parametrize("timeout", [-1.0, float("nan"), float("inf"), True])
def test_invalid_timeout_is_categorical(tmp_path, timeout):
    profile = _profile(tmp_path)

    with pytest.raises(maintenance.UnsafeProfileState):
        maintenance.acquire_profile_state_shared(profile, timeout_seconds=timeout)


@pytest.mark.parametrize(
    "acquire",
    [
        maintenance.acquire_profile_state_shared,
        maintenance.acquire_profile_state_exclusive,
    ],
)
def test_public_acquisition_rejects_unsupported_platform_before_lock_creation(
    tmp_path, monkeypatch, acquire
):
    profile = _profile(tmp_path, "unsupported-do-not-disclose")
    monkeypatch.setattr(maintenance.sys, "platform", "win32")

    with pytest.raises(maintenance.ProfileStateMaintenanceUnsupported) as exc_info:
        acquire(profile, timeout_seconds=0.1)

    _assert_categorical(
        exc_info.value,
        category="profile_state_maintenance_unsupported",
        profile=profile,
    )
    assert not _lock_path(profile).exists()


@pytest.mark.parametrize(
    "acquire",
    [
        maintenance.acquire_profile_state_shared,
        maintenance.acquire_profile_state_exclusive,
    ],
)
@pytest.mark.parametrize(
    "missing_capability",
    [None, "flock", "LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN"],
)
def test_public_acquisition_rejects_incomplete_fcntl_before_lock_creation(
    tmp_path, monkeypatch, acquire, missing_capability
):
    profile = _profile(tmp_path, "capability-do-not-disclose")
    if missing_capability is None:
        incomplete_fcntl = None
    else:
        capabilities = {
            name: getattr(maintenance.fcntl, name)
            for name in ("flock", "LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN")
            if name != missing_capability
        }
        incomplete_fcntl = SimpleNamespace(**capabilities)
    monkeypatch.setattr(maintenance, "fcntl", incomplete_fcntl)

    with pytest.raises(maintenance.ProfileStateMaintenanceUnsupported) as exc_info:
        acquire(profile, timeout_seconds=0.1)

    _assert_categorical(
        exc_info.value,
        category="profile_state_maintenance_unsupported",
        profile=profile,
    )
    assert not _lock_path(profile).exists()
