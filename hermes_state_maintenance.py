"""Profile-scoped coordination for Hermes state maintenance.

This module deliberately exposes no path-bearing diagnostics.  The held
profile directory and persistent lock inode are the authority; callers only
receive an opaque shared or exclusive lease, or a fixed categorical error.
"""

from __future__ import annotations

from contextlib import contextmanager
import ctypes
import errno
from functools import wraps
import math
import os
from pathlib import Path
import secrets
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
_RECOVERY_BARRIER_NAME = ".hermes-state-recovery.barrier"
_RECOVERY_BARRIER_STAGE_PREFIX = ".hermes-state-recovery.stage."
_RECOVERY_BARRIER_RETIRED_PREFIX = ".hermes-state-recovery.retired."
_RECOVERY_BARRIER_RECORD_PREFIX = b"HERMES_STATE_RECOVERY_BARRIER_V1\n"
_RECOVERY_BARRIER_NONCE_BYTES = 64
_RECOVERY_BARRIER_RECORD_SIZE = (
    len(_RECOVERY_BARRIER_RECORD_PREFIX) + _RECOVERY_BARRIER_NONCE_BYTES + 1
)
_WRITER_LEASE_TIMEOUT_SECONDS = 60.0
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


class ProfileStateRecoveryRequired(ProfileStateMaintenanceError):
    """A durable maintenance recovery barrier blocks profile mutation."""

    category = "profile_state_recovery_required"


class UnsafeRecoveryBarrier(ProfileStateMaintenanceError):
    """Recovery-barrier evidence was malformed, substituted, or ambiguous."""

    category = "unsafe_recovery_barrier"


class _LockInitializationInProgress(Exception):
    """The same safe lock inode changed size while its record was initialized."""


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


def _barrier_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _valid_barrier_stat(value: os.stat_result) -> bool:
    expected_uid, _expected_gid = _current_identity()
    return (
        stat.S_ISREG(value.st_mode)
        and stat.S_IMODE(value.st_mode) == _LOCK_MODE
        and value.st_uid == expected_uid
        and value.st_nlink == 1
        and value.st_size == _RECOVERY_BARRIER_RECORD_SIZE
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
    allow_initialization_transition: bool = False,
) -> tuple[os.stat_result, tuple[int, ...]]:
    held = os.fstat(lock_fd)
    named = os.stat(
        _PROFILE_STATE_LOCK_NAME,
        dir_fd=profile_fd,
        follow_symlinks=False,
    )
    held_identity = _lock_identity(held)
    named_identity = _lock_identity(named)
    if not _valid_lock_stat(held) or not _valid_lock_stat(named):
        raise UnsafeProfileState
    if held_identity != named_identity:
        if (
            allow_initialization_transition
            and expected_identity is None
            and held_identity[:-1] == named_identity[:-1]
            and 0 in (held.st_size, named.st_size)
        ):
            raise _LockInitializationInProgress
        raise UnsafeProfileState
    if expected_identity is not None and held_identity != expected_identity:
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
            try:
                held, _ = _validate_lock_metadata(
                    profile_fd,
                    lock_fd,
                    allow_initialization_transition=True,
                )
            except _LockInitializationInProgress:
                attempt_is_initial = False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ProfileStateLeaseTimeout
                time.sleep(min(_LOCK_POLL_SECONDS, remaining))
                continue
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


def _validate_operation_nonce(operation_nonce: str) -> str:
    if (
        type(operation_nonce) is not str
        or len(operation_nonce) != _RECOVERY_BARRIER_NONCE_BYTES
        or any(character not in "0123456789abcdef" for character in operation_nonce)
    ):
        raise UnsafeRecoveryBarrier
    return operation_nonce


def _barrier_record(operation_nonce: str) -> bytes:
    nonce = _validate_operation_nonce(operation_nonce)
    return _RECOVERY_BARRIER_RECORD_PREFIX + nonce.encode("ascii") + b"\n"


def _parse_barrier_record(payload: bytes) -> str:
    if (
        len(payload) != _RECOVERY_BARRIER_RECORD_SIZE
        or not payload.startswith(_RECOVERY_BARRIER_RECORD_PREFIX)
        or not payload.endswith(b"\n")
    ):
        raise UnsafeRecoveryBarrier
    nonce_bytes = payload[len(_RECOVERY_BARRIER_RECORD_PREFIX) : -1]
    try:
        nonce = nonce_bytes.decode("ascii")
    except UnicodeDecodeError:
        raise UnsafeRecoveryBarrier from None
    return _validate_operation_nonce(nonce)


def _validate_live_lease(
    lease: _ProfileStateLease,
    expected_type: type[SharedStateLease] | type[ExclusiveMaintenanceLease],
) -> int:
    if type(lease) is not expected_type:
        raise UnsafeProfileState
    profile_fd = lease._profile_fd
    lock_fd = lease._lock_fd
    if profile_fd < 0 or lock_fd < 0:
        raise UnsafeProfileState
    profile_stat = os.fstat(profile_fd)
    if (
        not _valid_profile_stat(profile_stat)
        or _profile_identity(profile_stat) != lease._profile_identity
    ):
        raise UnsafeProfileState
    _validate_initialized_lock(
        profile_fd,
        lock_fd,
        expected_identity=lease._lock_identity,
    )
    return profile_fd


def _open_recovery_barrier(profile_fd: int) -> int:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    return os.open(_RECOVERY_BARRIER_NAME, flags, dir_fd=profile_fd)


def _close_fd_once(fd: int) -> None:
    """Relinquish descriptor custody once without retrying an ambiguous close."""
    try:
        os.close(fd)
    except OSError:
        pass


def _rename_barrier_no_replace(
    profile_fd: int,
    source: str,
    destination: str,
) -> None:
    """Atomically rename one anchored name without replacing its destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin":
        try:
            rename = libc.renameatx_np
        except AttributeError:
            raise ProfileStateMaintenanceUnsupported from None
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            profile_fd,
            source_bytes,
            profile_fd,
            destination_bytes,
            0x00000004,  # RENAME_EXCL
        )
    elif sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError:
            raise ProfileStateMaintenanceUnsupported from None
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(
            profile_fd,
            source_bytes,
            profile_fd,
            destination_bytes,
            1,  # RENAME_NOREPLACE
        )
    else:
        raise ProfileStateMaintenanceUnsupported
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))


def _validate_retired_barrier(
    profile_fd: int,
    barrier_fd: int,
    identity: tuple[int, ...],
    retired_name: str,
) -> None:
    """Prove the held barrier moved to quarantine and the fixed name is absent."""
    try:
        held = os.fstat(barrier_fd)
        retired = os.stat(
            retired_name,
            dir_fd=profile_fd,
            follow_symlinks=False,
        )
        try:
            os.stat(
                _RECOVERY_BARRIER_NAME,
                dir_fd=profile_fd,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            fixed_name_absent = True
        else:
            fixed_name_absent = False
    except OSError:
        raise UnsafeRecoveryBarrier from None
    if (
        not fixed_name_absent
        or not _valid_barrier_stat(held)
        or not _valid_barrier_stat(retired)
        or _barrier_identity(held) != identity
        or _barrier_identity(retired) != identity
    ):
        raise UnsafeRecoveryBarrier


def _read_recovery_barrier(
    profile_fd: int,
) -> tuple[int, tuple[int, ...], str] | None:
    try:
        barrier_fd = _open_recovery_barrier(profile_fd)
    except FileNotFoundError:
        return None
    except OSError:
        raise UnsafeRecoveryBarrier from None
    try:
        held = os.fstat(barrier_fd)
        named = os.stat(
            _RECOVERY_BARRIER_NAME,
            dir_fd=profile_fd,
            follow_symlinks=False,
        )
        identity = _barrier_identity(held)
        if (
            not _valid_barrier_stat(held)
            or not _valid_barrier_stat(named)
            or _barrier_identity(named) != identity
        ):
            raise UnsafeRecoveryBarrier
        payload = os.pread(barrier_fd, _RECOVERY_BARRIER_RECORD_SIZE + 1, 0)
        nonce = _parse_barrier_record(payload)
        held_after = os.fstat(barrier_fd)
        named_after = os.stat(
            _RECOVERY_BARRIER_NAME,
            dir_fd=profile_fd,
            follow_symlinks=False,
        )
        if (
            _barrier_identity(held_after) != identity
            or _barrier_identity(named_after) != identity
        ):
            raise UnsafeRecoveryBarrier
        return barrier_fd, identity, nonce
    except ProfileStateMaintenanceError:
        _close_fd_once(barrier_fd)
        raise
    except (OSError, TypeError, ValueError, OverflowError):
        _close_fd_once(barrier_fd)
        raise UnsafeRecoveryBarrier from None
    except BaseException:
        _close_fd_once(barrier_fd)
        raise


def _write_all(fd: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        try:
            written = os.write(fd, payload[offset:])
        except OSError as exc:
            if exc.errno == errno.EINTR:
                continue
            raise
        if written <= 0:
            raise OSError(errno.EIO, "barrier publication failed")
        offset += written


def _stage_recovery_barrier(profile_fd: int, payload: bytes) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _attempt in range(16):
        stage_name = _RECOVERY_BARRIER_STAGE_PREFIX + secrets.token_hex(16)
        try:
            stage_fd = os.open(
                stage_name,
                flags,
                _LOCK_MODE,
                dir_fd=profile_fd,
            )
        except FileExistsError:
            continue
        try:
            os.fchmod(stage_fd, _LOCK_MODE)
            _write_all(stage_fd, payload)
            os.fsync(stage_fd)
            return stage_fd, stage_name
        except BaseException:
            _close_fd_once(stage_fd)
            try:
                os.unlink(stage_name, dir_fd=profile_fd)
            except OSError:
                pass
            raise
    raise UnsafeRecoveryBarrier


def publish_recovery_barrier(
    lease: ExclusiveMaintenanceLease,
    operation_nonce: str,
) -> None:
    """Durably publish the fixed categorical recovery barrier."""
    payload = _barrier_record(operation_nonce)
    with lease._close_lock:
        profile_fd = _validate_live_lease(lease, ExclusiveMaintenanceLease)
        existing = _read_recovery_barrier(profile_fd)
        if existing is not None:
            existing_fd, _identity, _nonce = existing
            _close_fd_once(existing_fd)
            raise ProfileStateRecoveryRequired
        stage_fd = -1
        stage_name = ""
        published = False
        try:
            stage_fd, stage_name = _stage_recovery_barrier(profile_fd, payload)
            os.link(
                stage_name,
                _RECOVERY_BARRIER_NAME,
                src_dir_fd=profile_fd,
                dst_dir_fd=profile_fd,
                follow_symlinks=False,
            )
            published = True
            os.unlink(stage_name, dir_fd=profile_fd)
            stage_name = ""
            os.fsync(profile_fd)
            barrier = _read_recovery_barrier(profile_fd)
            if barrier is None:
                raise UnsafeRecoveryBarrier
            barrier_fd, _identity, nonce = barrier
            _close_fd_once(barrier_fd)
            if nonce != operation_nonce:
                raise UnsafeRecoveryBarrier
        except ProfileStateMaintenanceError:
            raise
        except (OSError, TypeError, ValueError, OverflowError):
            raise UnsafeRecoveryBarrier from None
        finally:
            if stage_fd >= 0:
                _close_fd_once(stage_fd)
            if stage_name:
                try:
                    os.unlink(stage_name, dir_fd=profile_fd)
                    if not published:
                        os.fsync(profile_fd)
                except OSError:
                    pass


def require_no_recovery_barrier(lease: SharedStateLease) -> None:
    """Refuse a mutation while any durable recovery barrier is present."""
    with lease._close_lock:
        profile_fd = _validate_live_lease(lease, SharedStateLease)
        barrier = _read_recovery_barrier(profile_fd)
        if barrier is None:
            return
        barrier_fd, _identity, _nonce = barrier
        _close_fd_once(barrier_fd)
        raise ProfileStateRecoveryRequired


def _republish_barrier_after_failed_retirement(
    profile_fd: int,
    payload: bytes,
) -> None:
    stage_fd = -1
    stage_name = ""
    try:
        stage_fd, stage_name = _stage_recovery_barrier(profile_fd, payload)
        os.link(
            stage_name,
            _RECOVERY_BARRIER_NAME,
            src_dir_fd=profile_fd,
            dst_dir_fd=profile_fd,
            follow_symlinks=False,
        )
        os.unlink(stage_name, dir_fd=profile_fd)
        stage_name = ""
        try:
            os.fsync(profile_fd)
        except OSError:
            pass
    except OSError:
        pass
    finally:
        if stage_fd >= 0:
            _close_fd_once(stage_fd)
        if stage_name:
            try:
                os.unlink(stage_name, dir_fd=profile_fd)
            except OSError:
                pass


def retire_recovery_barrier(
    lease: ExclusiveMaintenanceLease,
    operation_nonce: str,
) -> None:
    """Durably retire only the exact nonce-bound recovery barrier."""
    payload = _barrier_record(operation_nonce)
    with lease._close_lock:
        profile_fd = _validate_live_lease(lease, ExclusiveMaintenanceLease)
        barrier = _read_recovery_barrier(profile_fd)
        if barrier is None:
            raise UnsafeRecoveryBarrier
        barrier_fd, identity, nonce = barrier
        retirement_attempted = False
        try:
            if nonce != operation_nonce:
                raise UnsafeRecoveryBarrier
            retired_name = _RECOVERY_BARRIER_RETIRED_PREFIX + secrets.token_hex(16)
            # Record uncertainty before entering the mutating syscall.  A
            # catchable signal can arrive after the kernel rename succeeds but
            # before the helper returns; every exception from this point must
            # conservatively attempt no-replace republication.
            retirement_attempted = True
            _rename_barrier_no_replace(
                profile_fd,
                _RECOVERY_BARRIER_NAME,
                retired_name,
            )
            _validate_retired_barrier(
                profile_fd,
                barrier_fd,
                identity,
                retired_name,
            )
            os.fsync(profile_fd)
            _validate_retired_barrier(
                profile_fd,
                barrier_fd,
                identity,
                retired_name,
            )
        except ProfileStateMaintenanceError:
            if retirement_attempted:
                _republish_barrier_after_failed_retirement(profile_fd, payload)
            raise
        except (OSError, TypeError, ValueError, OverflowError):
            if retirement_attempted:
                _republish_barrier_after_failed_retirement(profile_fd, payload)
            raise UnsafeRecoveryBarrier from None
        except BaseException:
            if retirement_attempted:
                try:
                    _republish_barrier_after_failed_retirement(profile_fd, payload)
                except BaseException:
                    pass
            raise
        finally:
            _close_fd_once(barrier_fd)


@contextmanager
def _profile_state_mutation_scope(
    profile_roots: tuple[Path, ...],
    *,
    timeout_seconds: float = _WRITER_LEASE_TIMEOUT_SECONDS,
):
    """Hold ordered shared authority and check every profile barrier."""
    timeout = _validate_timeout(timeout_seconds)
    unique_roots = {os.fspath(root): root for root in profile_roots}
    ordered_roots = tuple(
        sorted(unique_roots.values(), key=_canonical_profile_order_key)
    )
    deadline = time.monotonic() + timeout
    leases: list[SharedStateLease] = []
    try:
        for profile_root in ordered_roots:
            remaining = max(0.0, deadline - time.monotonic())
            lease = acquire_profile_state_shared(
                profile_root,
                timeout_seconds=remaining,
            )
            leases.append(lease)
        for lease in leases:
            require_no_recovery_barrier(lease)
        yield tuple(leases)
    finally:
        for lease in reversed(leases):
            lease.close()


def _leased_profile_mutation(profile_roots):
    """Decorate one audited writer with its complete profile-lease span."""

    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            roots = tuple(profile_roots(*args, **kwargs))
            if not roots:
                return function(*args, **kwargs)
            with _profile_state_mutation_scope(roots):
                return function(*args, **kwargs)

        return wrapped

    return decorate


def _fsync_directory(directory: Path) -> None:
    """Durably publish a canonical profile-sidecar directory mutation."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    directory_fd = os.open(directory, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _canonical_profile_order_key(profile_root: Path) -> tuple[str, str]:
    """Return a deterministic resolved-path key without changing authority."""
    _validate_profile_argument(profile_root)
    try:
        resolved = profile_root.resolve(strict=True)
    except (OSError, RuntimeError):
        raise UnsafeProfileState from None
    return os.fspath(resolved), os.fspath(profile_root)


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
