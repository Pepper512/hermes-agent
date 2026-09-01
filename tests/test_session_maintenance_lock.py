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
from unittest.mock import MagicMock, patch

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
_PUBLISH_BARRIER_SCRIPT = r"""
import os
import sys
from pathlib import Path

import hermes_state_maintenance as maintenance

lease = maintenance.acquire_profile_state_exclusive(
    Path(sys.argv[1]), timeout_seconds=5.0
)
maintenance.publish_recovery_barrier(lease, sys.argv[2])
print("published", flush=True)
os._exit(0)
"""
_BARRIER_STRESS_SCRIPT = r"""
import os
import sys
from pathlib import Path

import hermes_state_maintenance as maintenance
from hermes_state import SessionDB

blocked_path = Path(sys.argv[1]) / "state.db"
open_path = Path(sys.argv[2]) / "state.db"
try:
    SessionDB(blocked_path)
except maintenance.ProfileStateMaintenanceError as exc:
    category = exc.category
else:
    category = "unexpected_write_authority"

db = SessionDB(open_path)
db.set_meta(f"stress-{os.getpid()}", "ok")
db.close()
print(f"{category}:unrelated_ok", flush=True)
"""


def _profile(tmp_path: Path, name: str = "profile") -> Path:
    profile = tmp_path / name
    profile.mkdir(mode=0o700)
    profile.chmod(0o700)
    return profile


def _lock_path(profile: Path) -> Path:
    return profile / maintenance._PROFILE_STATE_LOCK_NAME


def _barrier_path(profile: Path) -> Path:
    return profile / maintenance._RECOVERY_BARRIER_NAME


def _operation_nonce(character: str = "a") -> str:
    return character * 64


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


def test_published_recovery_barrier_blocks_later_same_profile_writer(tmp_path):
    profile = _profile(tmp_path, "do-not-disclose")
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with maintenance.acquire_profile_state_shared(
        profile, timeout_seconds=1.0
    ) as shared:
        with pytest.raises(maintenance.ProfileStateRecoveryRequired) as exc_info:
            maintenance.require_no_recovery_barrier(shared)

    _assert_categorical(
        exc_info.value,
        category="profile_state_recovery_required",
        profile=profile,
    )


def test_recovery_barrier_retires_only_with_exact_nonce(tmp_path):
    profile = _profile(tmp_path, "do-not-disclose")
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce("a"))
        published_identity = (
            _barrier_path(profile).stat().st_dev,
            _barrier_path(profile).stat().st_ino,
        )
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.retire_recovery_barrier(exclusive, _operation_nonce("b"))
        assert _barrier_path(profile).exists()
        maintenance.retire_recovery_barrier(exclusive, _operation_nonce("a"))

    assert not _barrier_path(profile).exists()
    retired = list(profile.glob(maintenance._RECOVERY_BARRIER_RETIRED_PREFIX + "*"))
    assert len(retired) == 1
    assert (retired[0].stat().st_dev, retired[0].stat().st_ino) == published_identity
    with maintenance.acquire_profile_state_shared(
        profile, timeout_seconds=1.0
    ) as shared:
        maintenance.require_no_recovery_barrier(shared)


def test_recovery_barrier_retirement_rejects_last_window_replacement(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    nonce = _operation_nonce()
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    maintenance.publish_recovery_barrier(exclusive, nonce)
    payload = _barrier_path(profile).read_bytes()
    real_retire = getattr(maintenance, "_rename_barrier_no_replace", None)
    injected = False

    def replace_at_retirement(profile_fd: int, source: str, destination: str) -> None:
        nonlocal injected
        assert real_retire is not None
        if not injected:
            injected = True
            _barrier_path(profile).rename(profile / "held-original")
            _write_private_file(_barrier_path(profile), payload)
        real_retire(profile_fd, source, destination)

    monkeypatch.setattr(
        maintenance,
        "_rename_barrier_no_replace",
        replace_at_retirement,
        raising=False,
    )
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.retire_recovery_barrier(exclusive, nonce)
        assert _barrier_path(profile).exists()
    finally:
        exclusive.close()


def test_recovery_barrier_retirement_fsync_failure_republishes(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    nonce = _operation_nonce()
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    maintenance.publish_recovery_barrier(exclusive, nonce)
    real_fsync = maintenance.os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "synthetic retirement durability failure")
        real_fsync(fd)

    monkeypatch.setattr(maintenance.os, "fsync", fail_directory_fsync)
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.retire_recovery_barrier(exclusive, nonce)
        assert _barrier_path(profile).exists()
    finally:
        exclusive.close()


def test_recovery_barrier_retirement_baseexception_republishes(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    nonce = _operation_nonce()
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    maintenance.publish_recovery_barrier(exclusive, nonce)
    real_fsync = maintenance.os.fsync

    def cancel_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise KeyboardInterrupt
        real_fsync(fd)

    monkeypatch.setattr(maintenance.os, "fsync", cancel_directory_fsync)
    try:
        with pytest.raises(KeyboardInterrupt):
            maintenance.retire_recovery_barrier(exclusive, nonce)
        assert _barrier_path(profile).exists()
    finally:
        exclusive.close()


def test_recovery_barrier_close_failure_preserves_category_and_fd_custody(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    nonce = _operation_nonce()
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, nonce)
    barrier_inode = _barrier_path(profile).stat().st_ino
    shared = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    real_close = maintenance.os.close
    reused_fds: list[int] = []
    barrier_close_calls = 0

    def close_then_reuse(fd: int) -> None:
        nonlocal barrier_close_calls
        try:
            is_barrier = os.fstat(fd).st_ino == barrier_inode
        except OSError:
            is_barrier = False
        real_close(fd)
        if is_barrier:
            barrier_close_calls += 1
            reused_fd = os.open(os.devnull, os.O_RDONLY)
            assert reused_fd == fd
            reused_fds.append(reused_fd)
            raise OSError(errno.EINTR, "synthetic ambiguous close")

    monkeypatch.setattr(maintenance.os, "close", close_then_reuse)
    try:
        with pytest.raises(maintenance.ProfileStateRecoveryRequired):
            maintenance.require_no_recovery_barrier(shared)
    finally:
        monkeypatch.setattr(maintenance.os, "close", real_close)
        shared.close()
    assert barrier_close_calls == 1
    assert len(reused_fds) == 1
    os.fstat(reused_fds[0])
    real_close(reused_fds[0])

    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.retire_recovery_barrier(exclusive, nonce)


def test_barrier_publication_baseexception_releases_staged_descriptor(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )

    def cancel_link(*_args, **_kwargs) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(maintenance.os, "link", cancel_link)
    try:
        with pytest.raises(KeyboardInterrupt):
            maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
    finally:
        exclusive.close()
    assert not _barrier_path(profile).exists()
    assert not any(
        entry.name.startswith(maintenance._RECOVERY_BARRIER_STAGE_PREFIX)
        for entry in profile.iterdir()
    )


def test_read_only_sessiondb_neither_creates_nor_retires_barrier_authority(tmp_path):
    from hermes_state import SessionDB

    profile = _profile(tmp_path)
    state_path = profile / "state.db"
    db = SessionDB(state_path)
    db.set_meta("synthetic", "value")
    db.close()
    nonce = _operation_nonce()
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, nonce)
    expected_payload = _barrier_path(profile).read_bytes()
    recovery_entries = {
        entry.name
        for entry in profile.iterdir()
        if entry.name.startswith(".hermes-state-recovery")
    }

    read_only = SessionDB(state_path, read_only=True)
    try:
        assert read_only.get_meta("synthetic") == "value"
    finally:
        read_only.close()

    assert _barrier_path(profile).read_bytes() == expected_payload
    assert {
        entry.name
        for entry in profile.iterdir()
        if entry.name.startswith(".hermes-state-recovery")
    } == recovery_entries
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.retire_recovery_barrier(exclusive, nonce)


def test_multi_profile_acquisition_baseexception_releases_prior_lease(
    tmp_path, monkeypatch
):
    first = _profile(tmp_path, "a-profile")
    second = _profile(tmp_path, "b-profile")
    real_acquire = maintenance.acquire_profile_state_shared
    calls = 0

    def cancel_second(profile_root: Path, *, timeout_seconds: float):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return real_acquire(profile_root, timeout_seconds=timeout_seconds)

    monkeypatch.setattr(maintenance, "acquire_profile_state_shared", cancel_second)
    with pytest.raises(KeyboardInterrupt):
        with maintenance._profile_state_mutation_scope((second, first)):
            raise AssertionError("scope must not be entered")
    assert calls == 2

    for profile in (first, second):
        with maintenance.acquire_profile_state_exclusive(profile, timeout_seconds=0.1):
            pass


@pytest.mark.require_symlinks
def test_recovery_barrier_symlink_is_rejected(tmp_path):
    profile = _profile(tmp_path)
    with maintenance.acquire_profile_state_shared(
        profile, timeout_seconds=1.0
    ) as shared:
        target = profile / "other"
        _write_private_file(target, b"not-authority")
        _barrier_path(profile).symlink_to(target.name)
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.require_no_recovery_barrier(shared)


@pytest.mark.parametrize(
    "payload",
    [
        b"HERMES_STATE_RECOVERY_BARRIER_V2\n" + b"a" * 64 + b"\n",
        b"HERMES_STATE_RECOVERY_BARRIER_V1\n" + b"g" * 64 + b"\n",
    ],
)
def test_recovery_barrier_rejects_wrong_schema_or_nonce(payload, tmp_path):
    profile = _profile(tmp_path)
    with maintenance.acquire_profile_state_shared(
        profile, timeout_seconds=1.0
    ) as shared:
        _write_private_file(_barrier_path(profile), payload)
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.require_no_recovery_barrier(shared)


def test_recovery_barrier_replacement_during_check_is_rejected(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    nonce = _operation_nonce()
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, nonce)
    payload = _barrier_path(profile).read_bytes()
    shared = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    real_stat = maintenance.os.stat
    replaced = False

    def replace_before_named_stat(path, *args, **kwargs):
        nonlocal replaced
        if path == maintenance._RECOVERY_BARRIER_NAME and not replaced:
            replaced = True
            _barrier_path(profile).rename(profile / "retired-barrier")
            _write_private_file(_barrier_path(profile), payload)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(maintenance.os, "stat", replace_before_named_stat)
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.require_no_recovery_barrier(shared)
    finally:
        shared.close()


def test_recovery_barrier_disappearance_during_check_is_categorical(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
    shared = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    real_stat = maintenance.os.stat
    disappeared = False

    def disappear_before_named_stat(path, *args, **kwargs):
        nonlocal disappeared
        if path == maintenance._RECOVERY_BARRIER_NAME and not disappeared:
            disappeared = True
            _barrier_path(profile).unlink()
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(maintenance.os, "stat", disappear_before_named_stat)
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier) as exc_info:
            maintenance.require_no_recovery_barrier(shared)
    finally:
        shared.close()

    _assert_categorical(
        exc_info.value,
        category="unsafe_recovery_barrier",
        profile=profile,
    )


def test_recovery_barrier_persists_after_publisher_process_death(tmp_path):
    profile = _profile(tmp_path)
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _PUBLISH_BARRIER_SCRIPT,
            os.fspath(profile),
            _operation_nonce(),
        ],
        cwd=_REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert _read_line(process) == "published"
    process.wait(timeout=5.0)
    assert process.returncode == 0

    with maintenance.acquire_profile_state_shared(
        profile, timeout_seconds=1.0
    ) as shared:
        with pytest.raises(maintenance.ProfileStateRecoveryRequired):
            maintenance.require_no_recovery_barrier(shared)


def test_recovery_barrier_file_fsync_failure_does_not_publish(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    real_fsync = maintenance.os.fsync

    def fail_file_fsync(fd: int) -> None:
        if stat.S_ISREG(os.fstat(fd).st_mode) and fd != exclusive._lock_fd:
            raise OSError(errno.EIO, "synthetic file durability failure")
        real_fsync(fd)

    monkeypatch.setattr(maintenance.os, "fsync", fail_file_fsync)
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
    finally:
        exclusive.close()
    assert not _barrier_path(profile).exists()


def test_recovery_barrier_directory_fsync_failure_stays_blocking(tmp_path, monkeypatch):
    profile = _profile(tmp_path)
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    real_fsync = maintenance.os.fsync

    def fail_directory_fsync(fd: int) -> None:
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError(errno.EIO, "synthetic directory durability failure")
        real_fsync(fd)

    monkeypatch.setattr(maintenance.os, "fsync", fail_directory_fsync)
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
    finally:
        exclusive.close()
    assert _barrier_path(profile).exists()


def test_recovery_barrier_retirement_identity_mismatch_keeps_barrier(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    nonce = _operation_nonce()
    exclusive = maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    )
    maintenance.publish_recovery_barrier(exclusive, nonce)
    payload = _barrier_path(profile).read_bytes()
    real_stat = maintenance.os.stat
    replaced = False

    def replace_before_named_stat(path, *args, **kwargs):
        nonlocal replaced
        if path == maintenance._RECOVERY_BARRIER_NAME and not replaced:
            replaced = True
            _barrier_path(profile).rename(profile / "retired-barrier")
            _write_private_file(_barrier_path(profile), payload)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(maintenance.os, "stat", replace_before_named_stat)
    try:
        with pytest.raises(maintenance.UnsafeRecoveryBarrier):
            maintenance.retire_recovery_barrier(exclusive, nonce)
    finally:
        exclusive.close()
    assert _barrier_path(profile).exists()


def test_recovery_barrier_authority_is_lease_typed_and_profile_scoped(tmp_path):
    blocked = _profile(tmp_path, "blocked")
    unrelated = _profile(tmp_path, "unrelated")
    with maintenance.acquire_profile_state_exclusive(
        blocked, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
        with pytest.raises(maintenance.UnsafeProfileState):
            maintenance.require_no_recovery_barrier(exclusive)

    with maintenance.acquire_profile_state_shared(
        blocked, timeout_seconds=1.0
    ) as shared:
        with pytest.raises(maintenance.UnsafeProfileState):
            maintenance.publish_recovery_barrier(shared, _operation_nonce())
        with pytest.raises(maintenance.UnsafeProfileState):
            maintenance.retire_recovery_barrier(shared, _operation_nonce())

    with maintenance.acquire_profile_state_shared(
        unrelated, timeout_seconds=1.0
    ) as unrelated_shared:
        maintenance.require_no_recovery_barrier(unrelated_shared)
    assert not _barrier_path(unrelated).exists()


def test_already_open_sessiondb_rechecks_barrier_before_each_later_mutation(
    tmp_path,
):
    from hermes_state import SessionDB

    profile = _profile(tmp_path)
    db = SessionDB(profile / "state.db")
    try:
        db.set_meta("synthetic-key", "before")
        with maintenance.acquire_profile_state_exclusive(
            profile, timeout_seconds=1.0
        ) as exclusive:
            maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

        assert db.get_meta("synthetic-key") == "before"
        with pytest.raises(maintenance.ProfileStateRecoveryRequired):
            db.set_meta("synthetic-key", "blocked")
        assert db.get_meta("synthetic-key") == "before"

        with maintenance.acquire_profile_state_exclusive(
            profile, timeout_seconds=1.0
        ) as exclusive:
            maintenance.retire_recovery_barrier(exclusive, _operation_nonce())
        db.set_meta("synthetic-key", "after")
        assert db.get_meta("synthetic-key") == "after"
    finally:
        db.close()


def test_sessiondb_writable_open_refuses_before_database_creation(tmp_path):
    from hermes_state import SessionDB

    profile = _profile(tmp_path)
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        SessionDB(profile / "state.db")
    assert not (profile / "state.db").exists()

    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.retire_recovery_barrier(exclusive, _operation_nonce())


@pytest.mark.parametrize(
    "writer",
    [
        lambda db: db._try_wal_checkpoint(),
        lambda db: db.optimize_fts(),
        lambda db: db.rebuild_fts(),
        lambda db: db._merge_fts_incrementally(max_pages=1, max_commands=1),
        lambda db: db.optimize_fts_storage(),
        lambda db: db.vacuum(),
        lambda db: db.purge_stale_tool_call_markers(backup=False),
        lambda db: db.maybe_auto_prune_and_vacuum(vacuum=False),
    ],
    ids=[
        "checkpoint",
        "fts-optimize",
        "fts-rebuild",
        "fts-merge",
        "fts-storage",
        "vacuum",
        "marker-purge",
        "auto-prune",
    ],
)
def test_direct_sessiondb_maintenance_writer_refuses_active_barrier(tmp_path, writer):
    from hermes_state import SessionDB

    profile = _profile(tmp_path)
    db = SessionDB(profile / "state.db")
    try:
        with maintenance.acquire_profile_state_exclusive(
            profile, timeout_seconds=1.0
        ) as exclusive:
            maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
        with pytest.raises(maintenance.ProfileStateRecoveryRequired):
            writer(db)
        with maintenance.acquire_profile_state_exclusive(
            profile, timeout_seconds=1.0
        ) as exclusive:
            maintenance.retire_recovery_barrier(exclusive, _operation_nonce())
    finally:
        db.close()


@pytest.mark.parametrize(
    "writer_name",
    [
        "preflight_db_writability",
        "_db_opens_cleanly",
        "_live_writer_holds_db",
        "repair_state_db_schema",
        "quarantine_zeroed_state_db",
    ],
)
def test_direct_state_repair_writer_refuses_active_barrier(tmp_path, writer_name):
    import hermes_state

    profile = _profile(tmp_path)
    db = hermes_state.SessionDB(profile / "state.db")
    db.close()
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
    writer = getattr(hermes_state, writer_name)
    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        writer(profile / "state.db")
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.retire_recovery_barrier(exclusive, _operation_nonce())


@pytest.mark.parametrize(
    "writer_name",
    [
        "flush_pending_to_file",
        "spool_dropped_transcript_message",
        "flush_agent_history_to_file",
    ],
)
def test_shutdown_spool_writer_refuses_active_barrier(
    tmp_path, monkeypatch, writer_name
):
    from gateway import shutdown_flush

    profile = _profile(tmp_path)
    monkeypatch.setenv("HERMES_HOME", os.fspath(profile))
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    writer = getattr(shutdown_flush, writer_name)
    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        if writer_name == "flush_pending_to_file":
            writer({"synthetic": "value"})
        elif writer_name == "spool_dropped_transcript_message":
            writer("synthetic", {"role": "user", "content": "value"})
        else:
            writer("synthetic", [{"role": "user", "content": "value"}])
    assert not (profile / "pending_messages").exists()


def test_session_mirror_writer_refuses_active_fixed_profile_barrier(tmp_path):
    from gateway.session import SessionStore

    profile = _profile(tmp_path)
    sessions_dir = profile / "sessions"
    sessions_dir.mkdir(mode=0o700)
    store = object.__new__(SessionStore)
    store.sessions_dir = sessions_dir
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        store._save_sessions_json({})
    assert not (sessions_dir / "sessions.json").exists()


@pytest.mark.parametrize("operation", ["load", "persist"])
def test_gateway_routing_writer_checks_dynamic_database_profile(
    tmp_path, monkeypatch, operation
):
    from gateway.session import SessionStore

    fixed_parent = tmp_path / "fixed"
    dynamic_parent = tmp_path / "dynamic"
    fixed_parent.mkdir()
    dynamic_parent.mkdir()
    fixed_profile = _profile(fixed_parent)
    dynamic_profile = _profile(dynamic_parent)
    monkeypatch.setenv("HERMES_HOME", os.fspath(fixed_profile))
    store = object.__new__(SessionStore)
    store.sessions_dir = fixed_profile / "sessions"
    store._db = SimpleNamespace(db_path=dynamic_profile / "state.db")
    store._loaded = False
    with maintenance.acquire_profile_state_exclusive(
        dynamic_profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        if operation == "load":
            store._ensure_loaded_locked()
        else:
            store._persist_routing_data({}, 1)
    assert not store.sessions_dir.exists()


def test_gateway_profile_resolution_requires_explicit_database_path(tmp_path):
    from gateway.session import _session_store_profile_roots

    profile = _profile(tmp_path)
    store = SimpleNamespace(
        sessions_dir=profile / "sessions",
        _db=SimpleNamespace(),
    )

    with pytest.raises(AttributeError):
        _session_store_profile_roots(store)


def test_gateway_profile_resolution_without_database_uses_fixed_profile(tmp_path):
    from gateway.session import _session_store_profile_roots

    profile = _profile(tmp_path)
    store = SimpleNamespace(
        sessions_dir=profile / "sessions",
        _db=None,
    )

    assert _session_store_profile_roots(store) == (profile,)


def test_multi_profile_mutation_scope_uses_canonical_resolved_path_order(
    tmp_path, monkeypatch
):
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    first_profile = _profile(real_parent, "a-profile")
    second_profile = _profile(real_parent, "b-profile")
    alias_parent = tmp_path / "alias"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    aliased_second_profile = alias_parent / second_profile.name
    acquired: list[Path] = []

    class SyntheticLease:
        def close(self) -> None:
            return None

    def acquire(profile_root: Path, *, timeout_seconds: float):
        assert timeout_seconds >= 0
        acquired.append(profile_root)
        return SyntheticLease()

    monkeypatch.setattr(maintenance, "acquire_profile_state_shared", acquire)
    monkeypatch.setattr(
        maintenance,
        "require_no_recovery_barrier",
        lambda _lease: None,
    )

    with maintenance._profile_state_mutation_scope((
        aliased_second_profile,
        first_profile,
    )):
        pass

    assert acquired == [first_profile, aliased_second_profile]


def test_ghost_prune_holds_one_lease_through_sidecar_removal(tmp_path, monkeypatch):
    from hermes_state import SessionDB

    profile = _profile(tmp_path)
    sessions_dir = profile / "sessions"
    sessions_dir.mkdir(mode=0o700)
    sidecar = sessions_dir / "ghost.json"
    _write_private_file(sidecar, b"{}")
    db = SessionDB(profile / "state.db")
    db.create_session("ghost", "tui")
    db.end_session("ghost", "user_exit")
    db._conn.execute("UPDATE sessions SET started_at = 0 WHERE id = 'ghost'")
    db._conn.commit()

    sidecar_phase = threading.Event()
    allow_sidecar = threading.Event()
    barrier_published = threading.Event()
    allow_retirement = threading.Event()
    errors: list[BaseException] = []
    real_remove = db._remove_session_files

    def paused_remove(path: Path, session_id: str) -> None:
        sidecar_phase.set()
        assert allow_sidecar.wait(timeout=5.0)
        real_remove(path, session_id)

    def prune() -> None:
        try:
            assert db.prune_empty_ghost_sessions(sessions_dir) == 1
        except BaseException as exc:
            errors.append(exc)

    def maintain() -> None:
        try:
            with maintenance.acquire_profile_state_exclusive(
                profile, timeout_seconds=5.0
            ) as exclusive:
                maintenance.publish_recovery_barrier(exclusive, _operation_nonce())
                barrier_published.set()
                assert allow_retirement.wait(timeout=5.0)
                maintenance.retire_recovery_barrier(exclusive, _operation_nonce())
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(db, "_remove_session_files", paused_remove)
    prune_thread = threading.Thread(target=prune)
    maintenance_thread = threading.Thread(target=maintain)
    prune_thread.start()
    assert sidecar_phase.wait(timeout=5.0)
    maintenance_thread.start()
    published_before_sidecar_finished = barrier_published.wait(timeout=0.2)
    allow_sidecar.set()
    prune_thread.join(timeout=5.0)
    assert not prune_thread.is_alive()
    assert barrier_published.wait(timeout=5.0)
    allow_retirement.set()
    maintenance_thread.join(timeout=5.0)
    assert not maintenance_thread.is_alive()
    db.close()

    assert not published_before_sidecar_finished
    assert not errors
    assert not sidecar.exists()


@pytest.mark.parametrize("operation", ["drain", "recover"])
def test_shutdown_recovery_writer_checks_supplied_database_profile(
    tmp_path, monkeypatch, operation
):
    from gateway import shutdown_flush

    spool_parent = tmp_path / "spool"
    database_parent = tmp_path / "database"
    spool_parent.mkdir()
    database_parent.mkdir()
    spool_profile = _profile(spool_parent)
    database_profile = _profile(database_parent)
    monkeypatch.setenv("HERMES_HOME", os.fspath(spool_profile))
    with maintenance.acquire_profile_state_exclusive(
        database_profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        if operation == "drain":
            shutdown_flush.drain_transcript_spool(
                "synthetic",
                lambda _message: None,
                replay_profile_root=database_profile,
            )
        else:
            shutdown_flush.recover_pending_to_db(
                SimpleNamespace(db_path=database_profile / "state.db")
            )
    assert not (spool_profile / "pending_messages").exists()


def test_request_dump_writer_refuses_active_barrier(tmp_path):
    from agent.agent_runtime_helpers import dump_api_request_debug

    profile = _profile(tmp_path)
    logs_dir = profile / "sessions"
    logs_dir.mkdir(mode=0o700)
    agent = SimpleNamespace(
        _persist_disabled=False,
        persistence_policy="durable",
        logs_dir=logs_dir,
        client=SimpleNamespace(api_key="synthetic"),
        session_id="synthetic",
        base_url="https://invalid.test",
        api_mode="chat_completions",
        log_prefix="",
        verbose_logging=False,
        _mask_api_key_for_logs=lambda _value: "masked",
        _vprint=lambda _value: None,
    )
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        dump_api_request_debug(
            agent,
            {"messages": []},
            reason="synthetic",
        )
    assert list(logs_dir.iterdir()) == []


def test_durable_agent_initialization_refuses_active_barrier(tmp_path, monkeypatch):
    from agent import agent_init
    from run_agent import AIAgent

    profile = _profile(tmp_path)
    agent = object.__new__(AIAgent)
    agent._base_url = ""
    agent._base_url_lower = ""
    agent._base_url_hostname = ""
    monkeypatch.setattr(agent_init, "get_hermes_home", lambda: profile)
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with (
        patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(None, None),
        ),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch(
            "agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()
        ),
        patch("agent.anthropic_adapter.resolve_anthropic_token", return_value=""),
        patch("agent.anthropic_adapter._is_oauth_token", return_value=False),
        patch("agent.azure_identity_adapter.is_token_provider", return_value=False),
        patch(
            "hermes_cli.model_normalize.normalize_model_for_provider",
            return_value="synthetic",
        ),
        patch("agent.credential_pool.load_pool", return_value=MagicMock()),
        patch("hermes_cli.config.load_config", return_value={}),
        patch("hermes_cli.config.get_compatible_custom_providers", return_value=[]),
        patch("hermes_cli.plugins.discover_plugins"),
        patch("agent.iteration_budget.IterationBudget"),
        patch("hermes_cli.config.cfg_get", return_value=None),
    ):
        with pytest.raises(maintenance.ProfileStateRecoveryRequired):
            agent_init.init_agent(
                agent,
                base_url="https://api.anthropic.com",
                api_key="synthetic",
                model="synthetic",
                skip_context_files=True,
                skip_memory=True,
                quiet_mode=True,
            )
    assert not (profile / "sessions").exists()


def test_optional_session_snapshot_writer_refuses_active_barrier(tmp_path):
    from run_agent import AIAgent

    profile = _profile(tmp_path)
    logs_dir = profile / "sessions"
    logs_dir.mkdir(mode=0o700)
    agent = SimpleNamespace(
        _persist_disabled=False,
        persistence_policy="durable",
        _session_json_enabled=True,
        _session_messages=[],
        session_id="synthetic",
        logs_dir=logs_dir,
        model="synthetic",
        base_url="https://invalid.test",
        platform="test",
        session_start=__import__("datetime").datetime.now(),
        _cached_system_prompt="",
        tools=[],
        verbose_logging=False,
        _clean_session_content=lambda value: value,
        _redact_message_content=lambda value: value,
    )
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        AIAgent._save_session_log(
            agent,
            messages=[{"role": "user", "content": "value"}],
        )
    assert list(logs_dir.iterdir()) == []


@pytest.mark.parametrize("operation", ["copy", "restore"])
def test_snapshot_database_writer_refuses_active_barrier(tmp_path, operation):
    from hermes_cli.backup import _safe_copy_db, _safe_restore_db
    from hermes_state import SessionDB

    profile = _profile(tmp_path)
    state_path = profile / "state.db"
    db = SessionDB(state_path)
    db.close()
    snapshot = tmp_path / "snapshot.db"
    assert _safe_copy_db(state_path, snapshot)
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        if operation == "copy":
            _safe_copy_db(state_path, tmp_path / "second.db")
        else:
            _safe_restore_db(snapshot, state_path)


def test_update_restore_writer_refuses_active_barrier(tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    profile = _profile(tmp_path)
    state_path = profile / "state.db"
    state_path.write_bytes(b"synthetic-current")
    snapshot = tmp_path / "snapshot.db"
    snapshot.write_bytes(b"synthetic-snapshot")
    monkeypatch.setattr(update_cmd, "_foreign_db_holder_pids", None, raising=False)
    with maintenance.acquire_profile_state_exclusive(
        profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    with pytest.raises(maintenance.ProfileStateRecoveryRequired):
        update_cmd._restore_state_db_from_snapshot(state_path, snapshot)
    assert state_path.read_bytes() == b"synthetic-current"


def test_cross_process_barrier_stress_is_profile_scoped(tmp_path):
    from hermes_state import SessionDB

    blocked_profile = _profile(tmp_path, "blocked")
    open_profile = _profile(tmp_path, "open")
    for profile in (blocked_profile, open_profile):
        db = SessionDB(profile / "state.db")
        db.close()
    with maintenance.acquire_profile_state_exclusive(
        blocked_profile, timeout_seconds=1.0
    ) as exclusive:
        maintenance.publish_recovery_barrier(exclusive, _operation_nonce())

    processes = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _BARRIER_STRESS_SCRIPT,
                os.fspath(blocked_profile),
                os.fspath(open_profile),
            ],
            cwd=_REPOSITORY_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _index in range(8)
    ]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=15.0)
            assert process.returncode == 0, stderr
            assert stdout.strip() == "profile_state_recovery_required:unrelated_ok"
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5.0)

    db = SessionDB(open_profile / "state.db", read_only=True)
    try:
        rows = db._conn.execute(
            "SELECT COUNT(*) FROM state_meta WHERE key LIKE 'stress-%'"
        ).fetchone()[0]
    finally:
        db.close()
    assert rows == len(processes)


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


def test_same_inode_lock_initialization_size_transition_is_retried(
    tmp_path, monkeypatch
):
    profile = _profile(tmp_path)
    _write_private_file(_lock_path(profile))
    real_stat = maintenance.os.stat
    injected_transition = False

    def stat_during_initialization(path, *args, **kwargs):
        nonlocal injected_transition
        value = real_stat(path, *args, **kwargs)
        if path == maintenance._PROFILE_STATE_LOCK_NAME and not injected_transition:
            injected_transition = True
            fields = list(value)
            fields[6] = 1
            return os.stat_result(fields)
        return value

    monkeypatch.setattr(maintenance.os, "stat", stat_during_initialization)
    monkeypatch.setattr(
        maintenance.os,
        "supports_dir_fd",
        maintenance.os.supports_dir_fd | {stat_during_initialization},
    )
    monkeypatch.setattr(
        maintenance.os,
        "supports_follow_symlinks",
        maintenance.os.supports_follow_symlinks | {stat_during_initialization},
    )

    lease = maintenance.acquire_profile_state_shared(profile, timeout_seconds=1.0)
    lease.close()

    assert injected_transition is True
    assert _lock_path(profile).read_bytes().startswith(maintenance._LOCK_RECORD_PREFIX)


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
