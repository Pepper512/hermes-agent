"""Profile-scoped coordination for Hermes state maintenance.

This module deliberately exposes no path-bearing diagnostics.  The held
profile directory and persistent lock inode are the authority; callers only
receive an opaque shared or exclusive lease, or a fixed categorical error.
"""

from __future__ import annotations

import errno
import math
import os
from pathlib import Path
import stat
import sys
import threading
import time

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised by a native Windows lane
    fcntl = None  # type: ignore[assignment]


_PROFILE_STATE_LOCK_NAME = ".hermes-state-maintenance.lock"
_PROFILE_MODE = 0o700
_LOCK_MODE = 0o600
_LOCK_RECORD_PREFIX = b"HERMES_STATE_MAINTENANCE_LOCK_V1\n"
_LOCK_RECORD_LIMIT = 160
_LOCK_POLL_SECONDS = 0.01
_LEASE_AUTHORITY = object()


class ProfileStateMaintenanceError(RuntimeError):
    """Base class for fixed, path-free maintenance failures."""

    category = "profile_state_maintenance_error"

    def __init__(self) -> None:
        super().__init__(self.category)


class ProfileStateLeaseTimeout(ProfileStateMaintenanceError):
    """The bounded lease-acquisition budget expired."""

    category = "profile_state_lease_timeout"


class UnsafeProfileState(ProfileStateMaintenanceError):
    """Profile filesystem evidence was not safe enough to authorize."""

    category = "unsafe_profile_state"


class ProfileStateMaintenanceUnsupported(ProfileStateMaintenanceError):
    """The host lacks the required maintenance-lock semantics."""

    category = "profile_state_maintenance_unsupported"


def _require_supported_platform(platform: str) -> None:
    """Reject platforms outside the reviewed Darwin/Linux lock contract."""
    if platform != "darwin" and not platform.startswith("linux"):
        raise ProfileStateMaintenanceUnsupported


def _require_runtime_capabilities() -> None:
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    required_fcntl_constants = ("LOCK_SH", "LOCK_EX", "LOCK_NB", "LOCK_UN")
    if (
        fcntl is None
        or not callable(getattr(fcntl, "flock", None))
        or any(
            type(getattr(fcntl, name, None)) is not int
            for name in required_fcntl_constants
        )
        or any(not hasattr(os, name) for name in required_flags)
        or not hasattr(os, "pread")
        or not hasattr(os, "pwrite")
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
        or os.stat not in os.supports_follow_symlinks
    ):
        raise ProfileStateMaintenanceUnsupported


def _validate_timeout(timeout_seconds: float) -> float:
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds < 0
    ):
        raise UnsafeProfileState
    return float(timeout_seconds)


def _validate_profile_argument(profile_root: Path) -> None:
    if (
        type(profile_root) is not type(Path())
        or not profile_root.is_absolute()
        or ".." in profile_root.parts
        or profile_root.name in ("", ".", "..")
        or "\x00" in os.fspath(profile_root)
    ):
        raise UnsafeProfileState


def _current_identity() -> tuple[int, int]:
    try:
        return os.geteuid(), os.getegid()
    except AttributeError:
        raise ProfileStateMaintenanceUnsupported from None


def _profile_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
    )


def _lock_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        value.st_gid,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
    )


def _valid_profile_stat(value: os.stat_result) -> bool:
    expected_uid, _expected_gid = _current_identity()
    return (
        stat.S_ISDIR(value.st_mode)
        and stat.S_IMODE(value.st_mode) == _PROFILE_MODE
        and value.st_uid == expected_uid
        and value.st_nlink >= 2
    )


def _valid_lock_stat(value: os.stat_result) -> bool:
    expected_uid, _expected_gid = _current_identity()
    return (
        stat.S_ISREG(value.st_mode)
        and stat.S_IMODE(value.st_mode) == _LOCK_MODE
        and value.st_uid == expected_uid
        and value.st_nlink == 1
        and 0 <= value.st_size <= _LOCK_RECORD_LIMIT
    )


def _open_profile(profile_root: Path) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    profile_fd = os.open(profile_root, flags)
    try:
        held = os.fstat(profile_fd)
        named = os.stat(profile_root, follow_symlinks=False)
        if (
            not _valid_profile_stat(held)
            or not _valid_profile_stat(named)
            or _profile_identity(held) != _profile_identity(named)
            or held.st_nlink != named.st_nlink
        ):
            raise UnsafeProfileState
        return profile_fd, _profile_identity(held)
    except BaseException:
        os.close(profile_fd)
        raise


def _revalidate_profile(
    profile_root: Path,
    profile_fd: int,
    expected_identity: tuple[int, ...],
) -> tuple[int, ...]:
    held = os.fstat(profile_fd)
    named = os.stat(profile_root, follow_symlinks=False)
    held_identity = _profile_identity(held)
    if (
        not _valid_profile_stat(held)
        or not _valid_profile_stat(named)
        or _profile_identity(named) != held_identity
        or held.st_nlink != named.st_nlink
        or held_identity != expected_identity
    ):
        raise UnsafeProfileState
    return held_identity


def _open_lock(profile_fd: int) -> tuple[int, bool]:
    common_flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        common_flags |= os.O_CLOEXEC
    try:
        lock_fd = os.open(
            _PROFILE_STATE_LOCK_NAME,
            common_flags | os.O_CREAT | os.O_EXCL,
            _LOCK_MODE,
            dir_fd=profile_fd,
        )
    except FileExistsError:
        lock_fd = os.open(
            _PROFILE_STATE_LOCK_NAME,
            common_flags,
            dir_fd=profile_fd,
        )
        return lock_fd, False
    os.fchmod(lock_fd, _LOCK_MODE)
    return lock_fd, True


def _lock_record(value: os.stat_result) -> bytes:
    return _LOCK_RECORD_PREFIX + f"{value.st_dev:x}:{value.st_ino:x}\n".encode("ascii")


def _read_lock_record_state(lock_fd: int, held: os.stat_result) -> str:
    payload = os.pread(lock_fd, _LOCK_RECORD_LIMIT + 1, 0)
    if not payload:
        return "empty"
    if payload == _lock_record(held):
        return "valid"
    return "invalid"


def _validate_lock_metadata(
    profile_fd: int,
    lock_fd: int,
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> tuple[os.stat_result, tuple[int, ...]]:
    held = os.fstat(lock_fd)
    named = os.stat(
        _PROFILE_STATE_LOCK_NAME,
        dir_fd=profile_fd,
        follow_symlinks=False,
    )
    held_identity = _lock_identity(held)
    if (
        not _valid_lock_stat(held)
        or not _valid_lock_stat(named)
        or held_identity != _lock_identity(named)
        or (expected_identity is not None and held_identity != expected_identity)
    ):
        raise UnsafeProfileState
    return held, held_identity


def _validate_initialized_lock(
    profile_fd: int,
    lock_fd: int,
    *,
    expected_identity: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    held, held_identity = _validate_lock_metadata(
        profile_fd,
        lock_fd,
        expected_identity=expected_identity,
    )
    if _read_lock_record_state(lock_fd, held) != "valid":
        raise UnsafeProfileState
    held_after, held_identity_after = _validate_lock_metadata(
        profile_fd,
        lock_fd,
        expected_identity=held_identity,
    )
    if (
        held_identity_after != held_identity
        or _read_lock_record_state(lock_fd, held_after) != "valid"
    ):
        raise UnsafeProfileState
    return held_identity_after


def _write_all_at(lock_fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.pwrite(lock_fd, payload[offset:], offset)
        if written <= 0:
            raise OSError(errno.EIO, "lock initialization failed")
        offset += written


def _initialize_lock(profile_fd: int, lock_fd: int) -> tuple[int, ...]:
    held, held_identity = _validate_lock_metadata(profile_fd, lock_fd)
    if held.st_size != 0:
        if _read_lock_record_state(lock_fd, held) != "valid":
            raise UnsafeProfileState
        return held_identity
    payload = _lock_record(held)
    _write_all_at(lock_fd, payload)
    os.fsync(lock_fd)
    os.fsync(profile_fd)
    return _validate_initialized_lock(profile_fd, lock_fd)


def _is_contention(exc: OSError) -> bool:
    return exc.errno in {
        errno.EACCES,
        errno.EAGAIN,
        getattr(errno, "EWOULDBLOCK", errno.EAGAIN),
    }


def _try_flock(lock_fd: int, operation: int) -> str:
    assert fcntl is not None
    try:
        fcntl.flock(lock_fd, operation | fcntl.LOCK_NB)
        return "acquired"
    except OSError as exc:
        if exc.errno == errno.EINTR:
            return "interrupted"
        if _is_contention(exc):
            return "contended"
        raise UnsafeProfileState from None


def _try_flock_before_deadline(
    lock_fd: int,
    operation: int,
    deadline: float,
    *,
    initial_attempt: bool,
) -> str:
    if not initial_attempt and time.monotonic() >= deadline:
        raise ProfileStateLeaseTimeout
    return _try_flock(lock_fd, operation)


def _wait_for_flock(
    lock_fd: int,
    operation: int,
    deadline: float,
    *,
    initial_attempt: bool,
) -> None:
    attempt_is_initial = initial_attempt
    while True:
        result = _try_flock_before_deadline(
            lock_fd,
            operation,
            deadline,
            initial_attempt=attempt_is_initial,
        )
        attempt_is_initial = False
        if result == "acquired":
            return
        if result == "interrupted":
            continue
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProfileStateLeaseTimeout
        time.sleep(min(_LOCK_POLL_SECONDS, remaining))


def _acquire_initialized_lock(
    profile_fd: int,
    lock_fd: int,
    *,
    operation: int,
    deadline: float,
) -> tuple[int, ...]:
    """Acquire *operation*, safely completing an interrupted first create."""
    assert fcntl is not None
    owns_lock = False
    attempt_is_initial = True
    try:
        while True:
            held, _ = _validate_lock_metadata(profile_fd, lock_fd)
            record_state = _read_lock_record_state(lock_fd, held)
            if record_state == "valid":
                _wait_for_flock(
                    lock_fd,
                    operation,
                    deadline,
                    initial_attempt=attempt_is_initial,
                )
                owns_lock = True
                return _validate_initialized_lock(profile_fd, lock_fd)

            result = _try_flock_before_deadline(
                lock_fd,
                fcntl.LOCK_EX,
                deadline,
                initial_attempt=attempt_is_initial,
            )
            attempt_is_initial = False
            if result == "acquired":
                owns_lock = True
                held_after, _ = _validate_lock_metadata(profile_fd, lock_fd)
                state_after = _read_lock_record_state(lock_fd, held_after)
                if state_after == "empty":
                    lock_identity = _initialize_lock(profile_fd, lock_fd)
                elif state_after == "valid":
                    lock_identity = _validate_initialized_lock(profile_fd, lock_fd)
                else:
                    raise UnsafeProfileState
                if operation == fcntl.LOCK_SH:
                    _wait_for_flock(
                        lock_fd,
                        fcntl.LOCK_SH,
                        deadline,
                        initial_attempt=False,
                    )
                return lock_identity

            if result == "interrupted":
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ProfileStateLeaseTimeout
            time.sleep(min(_LOCK_POLL_SECONDS, remaining))
    except BaseException:
        if owns_lock:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            except OSError:
                pass
        raise


class _ProfileStateLease:
    __slots__ = (
        "_profile_fd",
        "_lock_fd",
        "_profile_identity",
        "_lock_identity",
        "_close_lock",
    )

    def __init__(
        self,
        authority: object,
        *,
        profile_fd: int,
        lock_fd: int,
        profile_identity: tuple[int, ...],
        lock_identity: tuple[int, ...],
    ) -> None:
        if authority is not _LEASE_AUTHORITY:
            raise TypeError("profile state leases are acquired, not constructed")
        self._profile_fd = profile_fd
        self._lock_fd = lock_fd
        self._profile_identity = profile_identity
        self._lock_identity = lock_identity
        self._close_lock = threading.Lock()

    def close(self) -> None:
        """Release this lease; repeated calls are harmless."""
        with self._close_lock:
            lock_fd = self._lock_fd
            profile_fd = self._profile_fd
            if lock_fd < 0 and profile_fd < 0:
                return
            self._lock_fd = -1
            self._profile_fd = -1
        try:
            if lock_fd >= 0 and fcntl is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
        finally:
            if lock_fd >= 0:
                try:
                    os.close(lock_fd)
                except OSError:
                    pass
            if profile_fd >= 0:
                try:
                    os.close(profile_fd)
                except OSError:
                    pass

    def __enter__(self):
        if self._lock_fd < 0 or self._profile_fd < 0:
            raise UnsafeProfileState
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class SharedStateLease(_ProfileStateLease):
    """A held shared profile-state lease."""


class ExclusiveMaintenanceLease(_ProfileStateLease):
    """A held exclusive profile-state maintenance lease."""


def _acquire_profile_state(
    profile_root: Path,
    *,
    timeout_seconds: float,
    exclusive: bool,
    lease_type: type[SharedStateLease] | type[ExclusiveMaintenanceLease],
) -> SharedStateLease | ExclusiveMaintenanceLease:
    _require_supported_platform(sys.platform)
    _require_runtime_capabilities()
    assert fcntl is not None
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    timeout = _validate_timeout(timeout_seconds)
    _validate_profile_argument(profile_root)
    deadline = time.monotonic() + timeout
    profile_fd = -1
    lock_fd = -1
    acquired = False
    try:
        profile_fd, profile_identity = _open_profile(profile_root)
        lock_fd, _created = _open_lock(profile_fd)
        profile_identity = _revalidate_profile(
            profile_root,
            profile_fd,
            profile_identity,
        )
        lock_identity = _acquire_initialized_lock(
            profile_fd,
            lock_fd,
            operation=operation,
            deadline=deadline,
        )
        acquired = True
        _revalidate_profile(profile_root, profile_fd, profile_identity)
        lock_identity = _validate_initialized_lock(
            profile_fd,
            lock_fd,
            expected_identity=lock_identity,
        )
        lease = lease_type(
            _LEASE_AUTHORITY,
            profile_fd=profile_fd,
            lock_fd=lock_fd,
            profile_identity=profile_identity,
            lock_identity=lock_identity,
        )
        profile_fd = -1
        lock_fd = -1
        return lease
    except ProfileStateMaintenanceError:
        raise
    except (OSError, TypeError, ValueError, OverflowError):
        raise UnsafeProfileState from None
    finally:
        if lock_fd >= 0:
            if acquired and fcntl is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            try:
                os.close(lock_fd)
            except OSError:
                pass
        if profile_fd >= 0:
            try:
                os.close(profile_fd)
            except OSError:
                pass


def acquire_profile_state_shared(
    profile_root: Path, *, timeout_seconds: float
) -> SharedStateLease:
    """Acquire a bounded shared lease for one owner-only profile."""
    lease = _acquire_profile_state(
        profile_root,
        timeout_seconds=timeout_seconds,
        exclusive=False,
        lease_type=SharedStateLease,
    )
    assert isinstance(lease, SharedStateLease)
    return lease


def acquire_profile_state_exclusive(
    profile_root: Path, *, timeout_seconds: float
) -> ExclusiveMaintenanceLease:
    """Acquire a bounded exclusive maintenance lease for one profile."""
    lease = _acquire_profile_state(
        profile_root,
        timeout_seconds=timeout_seconds,
        exclusive=True,
        lease_type=ExclusiveMaintenanceLease,
    )
    assert isinstance(lease, ExclusiveMaintenanceLease)
    return lease
