"""Atomic durable publication for anonymously staged TTS audio.

This module is intentionally not wired to the public TTS entry points.  It
consumes the one-shot transaction permit, copies one sealed descriptor into a
provider-invisible same-filesystem temporary, and publishes through the exact
host primitive authorized by the approved design.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
import sys
import threading
from typing import Callable, Final

from agent.file_safety import is_write_approval_required, is_write_denied
from hermes_cli.persistence import PersistenceObservation, PersistencePolicy
from tools.path_security import has_traversal_component
from tools.tts_staging import (
    MAX_ANONYMOUS_AUDIO_BYTES,
    AnonymousAudioScrubError,
    AnonymousAudioStage,
    SealedAudio,
)
from tools.tts_transaction import DurablePublicationPermit


_PUBLISH_ERROR: Final[str] = "tts_durable_publication_failed"
_PUBLISH_UNCERTAIN: Final[str] = "tts_durable_publication_uncertain"
_SCRUB_ERROR: Final[str] = "tts_anonymous_scrub_failed"
_COPY_CHUNK_BYTES: Final[int] = 64 * 1024
RENAME_NOREPLACE: Final[int] = 1
RENAME_EXCL: Final[int | None] = 0x00000004
RENAME_NOFOLLOW_ANY: Final[int | None] = 0x00000010
RENAME_RESOLVE_BENEATH: Final[int | None] = 0x00000020


class TTSPublishError(RuntimeError):
    """Fixed, path-free durable publication failure."""


class TTSPublishUncertain(TTSPublishError):
    """Publication linearized but directory durability could not be proved."""


@dataclass(frozen=True, slots=True)
class PublishedAudio:
    """The authorized final path after successful durable publication."""

    path: Path


@dataclass(slots=True)
class _HeldParent:
    fd: int
    path: Path
    stat: os.stat_result | None
    close_state: str = "open"
    close_proof: int | None = None
    close_active: bool = False
    close_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


@dataclass(slots=True)
class _PublicationTemp:
    fd: int
    name: str
    stat: os.stat_result | None
    destination_name: str
    publication_armed: bool = False
    published: bool = False
    close_state: str = "open"
    close_proof: int | None = None
    close_active: bool = False
    close_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )


@dataclass(slots=True)
class _PublicationCustody:
    parent: _HeldParent | None = None
    temp: _PublicationTemp | None = None


@dataclass(slots=True)
class _PublicationOutcome:
    status: str


@dataclass(frozen=True, slots=True)
class _PreparedPublicationCall:
    kind: str
    callable_ref: Callable[..., object]
    source: str | bytes
    destination: str | bytes
    source_dir_fd: int
    destination_dir_fd: int
    flags: int
    expected_argtypes: tuple[object, ...] | None = None
    expected_restype: object = None


def _load_libc_symbol(name: str):
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        function = getattr(libc, name)
    except (AttributeError, OSError):
        return None
    function.restype = ctypes.c_int
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    return function


_DARWIN_RENAMEATX_NP = _load_libc_symbol("renameatx_np")
_LINUX_RENAMEAT2 = _load_libc_symbol("renameat2")


def _darwin_renameatx_np(
    src_dir_fd: int,
    src: bytes,
    dst_dir_fd: int,
    dst: bytes,
    flags: int,
) -> None:
    function = _DARWIN_RENAMEATX_NP
    if function is None:
        raise OSError(errno.ENOSYS, "durable publication unavailable")
    ctypes.set_errno(0)
    if function(src_dir_fd, src, dst_dir_fd, dst, flags) != 0:
        error = ctypes.get_errno() or errno.EIO
        raise OSError(error, "durable publication unavailable")


def _linux_renameat2(
    src_dir_fd: int,
    src: bytes,
    dst_dir_fd: int,
    dst: bytes,
    flags: int,
) -> None:
    function = _LINUX_RENAMEAT2
    if function is None:
        raise OSError(errno.ENOSYS, "durable publication unavailable")
    ctypes.set_errno(0)
    if function(src_dir_fd, src, dst_dir_fd, dst, flags) != 0:
        error = ctypes.get_errno() or errno.EIO
        raise OSError(error, "durable publication unavailable")


def _require_absent_primitive_for_platform(platform: str) -> None:
    if platform == "darwin":
        if (
            _DARWIN_RENAMEATX_NP is None
            or type(RENAME_EXCL) is not int
            or type(RENAME_NOFOLLOW_ANY) is not int
            or type(RENAME_RESOLVE_BENEATH) is not int
        ):
            raise TTSPublishError(_PUBLISH_ERROR)
        return
    if platform.startswith("linux"):
        if _LINUX_RENAMEAT2 is None:
            raise TTSPublishError(_PUBLISH_ERROR)
        return
    raise TTSPublishError(_PUBLISH_ERROR)


def _rename_absent_darwin(
    src_dir_fd: int,
    src: str,
    dst_dir_fd: int,
    dst: str,
) -> None:
    _require_absent_primitive_for_platform("darwin")
    flags = RENAME_EXCL | RENAME_NOFOLLOW_ANY | RENAME_RESOLVE_BENEATH
    _darwin_renameatx_np(
        src_dir_fd,
        os.fsencode(src),
        dst_dir_fd,
        os.fsencode(dst),
        flags,
    )


def _rename_absent_linux(
    src_dir_fd: int,
    src: str,
    dst_dir_fd: int,
    dst: str,
) -> None:
    _require_absent_primitive_for_platform("linux")
    _linux_renameat2(
        src_dir_fd,
        os.fsencode(src),
        dst_dir_fd,
        os.fsencode(dst),
        RENAME_NOREPLACE,
    )


def _rename_absent_for_host(
    src_dir_fd: int,
    src: str,
    dst_dir_fd: int,
    dst: str,
) -> None:
    if sys.platform == "darwin":
        _rename_absent_darwin(src_dir_fd, src, dst_dir_fd, dst)
        return
    if sys.platform.startswith("linux"):
        _rename_absent_linux(src_dir_fd, src, dst_dir_fd, dst)
        return
    raise TTSPublishError(_PUBLISH_ERROR)


def _replace_existing(
    src_dir_fd: int,
    source: str,
    dst_dir_fd: int,
    destination: str,
) -> None:
    os.replace(
        source,
        destination,
        src_dir_fd=src_dir_fd,
        dst_dir_fd=dst_dir_fd,
    )


def _create_publication_call_preparer():
    canonical_call_record = _PreparedPublicationCall
    canonical_fsencode = os.fsencode
    canonical_replace = os.replace
    canonical_replace_helper = _replace_existing
    canonical_host_helper = _rename_absent_for_host
    canonical_darwin_helper = _rename_absent_darwin
    canonical_darwin_wrapper = _darwin_renameatx_np
    canonical_linux_helper = _rename_absent_linux
    canonical_linux_wrapper = _linux_renameat2
    canonical_darwin_symbol = _DARWIN_RENAMEATX_NP
    canonical_linux_symbol = _LINUX_RENAMEAT2
    canonical_rename_noreplace = RENAME_NOREPLACE
    canonical_rename_excl = RENAME_EXCL
    canonical_rename_nofollow_any = RENAME_NOFOLLOW_ANY
    canonical_rename_resolve_beneath = RENAME_RESOLVE_BENEATH
    canonical_argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    canonical_restype = ctypes.c_int

    def prepare(
        *,
        replacing: bool,
        parent_fd: int,
        source: str,
        destination: str,
        platform: str,
    ) -> _PreparedPublicationCall:
        if (
            _PreparedPublicationCall is not canonical_call_record
            or os.fsencode is not canonical_fsencode
            or os.replace is not canonical_replace
            or _replace_existing is not canonical_replace_helper
            or _rename_absent_for_host is not canonical_host_helper
            or _rename_absent_darwin is not canonical_darwin_helper
            or _darwin_renameatx_np is not canonical_darwin_wrapper
            or _rename_absent_linux is not canonical_linux_helper
            or _linux_renameat2 is not canonical_linux_wrapper
            or _DARWIN_RENAMEATX_NP is not canonical_darwin_symbol
            or _LINUX_RENAMEAT2 is not canonical_linux_symbol
            or RENAME_NOREPLACE != canonical_rename_noreplace
            or RENAME_EXCL != canonical_rename_excl
            or RENAME_NOFOLLOW_ANY != canonical_rename_nofollow_any
            or RENAME_RESOLVE_BENEATH != canonical_rename_resolve_beneath
        ):
            raise TTSPublishError(_PUBLISH_ERROR)
        if replacing:
            return canonical_call_record(
                kind="replace",
                callable_ref=canonical_replace,
                source=source,
                destination=destination,
                source_dir_fd=parent_fd,
                destination_dir_fd=parent_fd,
                flags=0,
            )
        if platform == "darwin":
            _require_absent_primitive_for_platform(platform)
            assert canonical_darwin_symbol is not None
            return canonical_call_record(
                kind="darwin",
                callable_ref=canonical_darwin_symbol,
                source=canonical_fsencode(source),
                destination=canonical_fsencode(destination),
                source_dir_fd=parent_fd,
                destination_dir_fd=parent_fd,
                flags=canonical_rename_excl
                | canonical_rename_nofollow_any
                | canonical_rename_resolve_beneath,
                expected_argtypes=canonical_argtypes,
                expected_restype=canonical_restype,
            )
        if platform.startswith("linux"):
            _require_absent_primitive_for_platform(platform)
            assert canonical_linux_symbol is not None
            return canonical_call_record(
                kind="linux",
                callable_ref=canonical_linux_symbol,
                source=canonical_fsencode(source),
                destination=canonical_fsencode(destination),
                source_dir_fd=parent_fd,
                destination_dir_fd=parent_fd,
                flags=canonical_rename_noreplace,
                expected_argtypes=canonical_argtypes,
                expected_restype=canonical_restype,
            )
        raise TTSPublishError(_PUBLISH_ERROR)

    return prepare


_prepare_publication_call = _create_publication_call_preparer()
del _create_publication_call_preparer


def _validate_destination(destination: object) -> Path:
    if type(destination) is not type(Path()):
        raise TTSPublishError(_PUBLISH_ERROR)
    raw = os.fspath(destination)
    if (
        type(raw) is not str
        or not raw
        or "\x00" in raw
        or has_traversal_component(raw)
        or destination.name in ("", ".", "..")
    ):
        raise TTSPublishError(_PUBLISH_ERROR)
    try:
        canonical = Path(os.path.realpath(os.path.expanduser(raw)))
        denied = is_write_denied(str(canonical))
        approval_required = is_write_approval_required(str(canonical))
    except (OSError, TypeError, ValueError):
        raise TTSPublishError(_PUBLISH_ERROR) from None
    if denied or approval_required:
        raise TTSPublishError(_PUBLISH_ERROR)
    return canonical


def _open_held_parent(
    destination: Path,
    custody: _PublicationCustody | None = None,
) -> _HeldParent:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    parent = _HeldParent(fd=-1, path=destination.parent, stat=None)
    if custody is not None:
        custody.parent = parent
    try:
        parent.fd = os.open(destination.parent, flags)
        held = os.fstat(parent.fd)
        parent.stat = held
        named = os.stat(destination.parent, follow_symlinks=False)
        if not _same_parent(held, named):
            raise TTSPublishError(_PUBLISH_ERROR)
        return parent
    except BaseException as exc:
        if parent.fd >= 0:
            try:
                _close_owned_fd(parent)
            except BaseException:
                try:
                    _close_owned_fd(parent)
                except BaseException:
                    pass
        if isinstance(exc, TTSPublishError):
            raise
        if isinstance(exc, (OSError, TypeError, ValueError)):
            raise TTSPublishError(_PUBLISH_ERROR) from None
        raise


def _same_parent(held: os.stat_result, named: os.stat_result) -> bool:
    return (
        stat.S_ISDIR(held.st_mode)
        and stat.S_ISDIR(named.st_mode)
        and held.st_mode == named.st_mode
        and held.st_dev == named.st_dev
        and held.st_ino == named.st_ino
        and held.st_uid == named.st_uid
        and held.st_gid == named.st_gid
        and held.st_nlink == named.st_nlink
    )


def _authorize_destination(parent: _HeldParent, name: str) -> bool:
    if sys.platform != "darwin" and not sys.platform.startswith("linux"):
        raise TTSPublishError(_PUBLISH_ERROR)
    try:
        target = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
    except FileNotFoundError:
        _require_absent_primitive_for_platform(sys.platform)
        return False
    except OSError:
        raise TTSPublishError(_PUBLISH_ERROR) from None
    if not stat.S_ISREG(target.st_mode):
        raise TTSPublishError(_PUBLISH_ERROR)
    return True


def _create_publication_temp(
    parent: _HeldParent,
    destination_name: str,
    custody: _PublicationCustody | None = None,
) -> _PublicationTemp:
    temp = _PublicationTemp(
        fd=-1,
        name="",
        stat=None,
        destination_name=destination_name,
    )
    if custody is not None:
        custody.temp = temp
    return temp


def _materialize_publication_temp(
    parent: _HeldParent,
    temp: _PublicationTemp,
) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    for _attempt in range(8):
        name = f".hermes-tts-publish-{secrets.token_hex(16)}"
        try:
            fd = os.open(name, flags, 0o600, dir_fd=parent.fd)
        except FileExistsError:
            continue
        except OSError:
            raise TTSPublishError(_PUBLISH_ERROR) from None
        temp.fd = fd
        temp.name = name
        try:
            temp.stat = os.fstat(fd)
            os.fchown(fd, os.getuid(), os.getgid())
            os.fchmod(fd, 0o600)
            held = os.fstat(fd)
            named = os.stat(name, dir_fd=parent.fd, follow_symlinks=False)
            if not _same_temp(held, named, require_size=0):
                raise TTSPublishError(_PUBLISH_ERROR)
            parent_held = os.fstat(parent.fd)
            parent_named = os.stat(parent.path, follow_symlinks=False)
            if not _same_parent(parent_held, parent_named):
                raise TTSPublishError(_PUBLISH_ERROR)
            parent.stat = parent_held
            temp.stat = held
            return
        except BaseException:
            try:
                _scrub_and_close_owned_fd(temp)
            except AnonymousAudioScrubError:
                raise
            raise
    raise TTSPublishError(_PUBLISH_ERROR)


def _same_temp(
    held: os.stat_result,
    named: os.stat_result,
    *,
    require_size: int,
) -> bool:
    return (
        stat.S_ISREG(held.st_mode)
        and stat.S_ISREG(named.st_mode)
        and stat.S_IMODE(held.st_mode) == 0o600
        and stat.S_IMODE(named.st_mode) == 0o600
        and held.st_uid == os.getuid()
        and named.st_uid == held.st_uid
        and held.st_gid == os.getgid()
        and named.st_gid == held.st_gid
        and held.st_nlink == 1
        and named.st_nlink == 1
        and held.st_dev == named.st_dev
        and held.st_ino == named.st_ino
        and held.st_size == require_size
        and named.st_size == require_size
    )


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError(errno.EIO, "publication copy failed")
        view = view[written:]


def _copy_sealed_to_publication(
    stage: AnonymousAudioStage,
    sealed: SealedAudio,
    temp: _PublicationTemp,
) -> None:
    if (
        type(stage) is not AnonymousAudioStage
        or type(sealed) is not SealedAudio
        or sealed._authority is not stage._authority
        or sealed._fd != stage._fd
        or type(sealed._size) is not int
        or sealed._size <= 0
        or sealed._size > MAX_ANONYMOUS_AUDIO_BYTES
    ):
        raise TTSPublishError(_PUBLISH_ERROR)
    digest = hashlib.sha256()
    remaining = sealed._size
    os.lseek(stage._fd, 0, os.SEEK_SET)
    while remaining:
        chunk = os.read(stage._fd, min(remaining, _COPY_CHUNK_BYTES))
        if not chunk:
            raise OSError(errno.EIO, "publication copy failed")
        _write_all(temp.fd, chunk)
        digest.update(chunk)
        remaining -= len(chunk)
    if digest.digest() != sealed._digest:
        raise TTSPublishError(_PUBLISH_ERROR)
    source_after = os.fstat(stage._fd)
    if not stage._matches_seal(source_after, sealed):
        raise TTSPublishError(_PUBLISH_ERROR)


def _revalidate_parent(parent: _HeldParent) -> None:
    held = os.fstat(parent.fd)
    named = os.stat(parent.path, follow_symlinks=False)
    if (
        not _same_parent(held, named)
        or held.st_dev != parent.stat.st_dev
        or held.st_ino != parent.stat.st_ino
        or held.st_uid != parent.stat.st_uid
        or held.st_gid != parent.stat.st_gid
        or held.st_mode != parent.stat.st_mode
        or held.st_nlink != parent.stat.st_nlink
    ):
        raise TTSPublishError(_PUBLISH_ERROR)


def _revalidate_publication_temp(
    parent: _HeldParent,
    temp: _PublicationTemp,
    size: int,
) -> None:
    if temp.stat is None:
        raise TTSPublishError(_PUBLISH_ERROR)
    held = os.fstat(temp.fd)
    named = os.stat(temp.name, dir_fd=parent.fd, follow_symlinks=False)
    if (
        not _same_temp(held, named, require_size=size)
        or held.st_dev != temp.stat.st_dev
        or held.st_ino != temp.stat.st_ino
        or held.st_uid != temp.stat.st_uid
        or held.st_gid != temp.stat.st_gid
    ):
        raise TTSPublishError(_PUBLISH_ERROR)


def _verify_publication_digest(
    temp: _PublicationTemp,
    sealed: SealedAudio,
) -> None:
    if (
        type(sealed._size) is not int
        or sealed._size <= 0
        or sealed._size > MAX_ANONYMOUS_AUDIO_BYTES
    ):
        raise TTSPublishError(_PUBLISH_ERROR)
    digest = hashlib.sha256()
    remaining = sealed._size
    try:
        os.lseek(temp.fd, 0, os.SEEK_SET)
        while remaining:
            chunk = os.read(temp.fd, min(remaining, _COPY_CHUNK_BYTES))
            if not chunk:
                raise TTSPublishError(_PUBLISH_ERROR)
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(temp.fd, 1):
            raise TTSPublishError(_PUBLISH_ERROR)
    finally:
        try:
            os.lseek(temp.fd, 0, os.SEEK_SET)
        except OSError:
            pass
    if digest.digest() != sealed._digest:
        raise TTSPublishError(_PUBLISH_ERROR)


def _fsync_publication_file(fd: int) -> None:
    os.fsync(fd)


def _fsync_parent(fd: int) -> None:
    os.fsync(fd)


def _same_owned_open_description(
    owner: _HeldParent | _PublicationTemp,
) -> bool:
    proof = owner.close_proof
    if proof is None:
        return False
    try:
        held = os.fstat(owner.fd)
        offset = os.lseek(owner.fd, 0, os.SEEK_CUR)
    except (OSError, TypeError, ValueError):
        return False
    expected = owner.stat
    return (
        expected is not None
        and held.st_dev == expected.st_dev
        and held.st_ino == expected.st_ino
        and stat.S_IFMT(held.st_mode) == stat.S_IFMT(expected.st_mode)
        and held.st_uid == expected.st_uid
        and held.st_gid == expected.st_gid
        and offset == proof
    )


def _close_owned_fd(owner: _HeldParent | _PublicationTemp) -> None:
    with owner.close_lock:
        owns_attempt = False
        try:
            if owner.close_active:
                return
            owns_attempt = True
            owner.close_active = True
            if owner.close_state == "released":
                return
            if owner.close_state == "open":
                if owner.stat is None:
                    owner.stat = os.fstat(owner.fd)
                proof = (1 << 60) | secrets.randbits(59)
                os.lseek(owner.fd, proof, os.SEEK_SET)
                owner.close_proof = proof
                owner.close_state = "attempted"
            elif not _same_owned_open_description(owner):
                owner.close_state = "released"
                return
            os.close(owner.fd)
            owner.close_state = "released"
        finally:
            if owns_attempt:
                owner.close_active = False


def _scrub_and_close_owned_fd(owner: _PublicationTemp) -> None:
    if owner.close_state != "open":
        _close_owned_fd(owner)
        return
    failed = False
    try:
        os.ftruncate(owner.fd, 0)
    except BaseException:
        failed = True
        try:
            os.ftruncate(owner.fd, 0)
        except BaseException:
            pass
    try:
        os.fsync(owner.fd)
    except BaseException:
        failed = True
    try:
        _close_owned_fd(owner)
    except BaseException:
        failed = True
        try:
            _close_owned_fd(owner)
        except BaseException:
            pass
    if failed:
        raise AnonymousAudioScrubError(_SCRUB_ERROR)


def _bind_publication_helpers(function: Callable[..., _PublicationOutcome]):
    canonical_prepare = _prepare_publication_call
    canonical_verify = _verify_publication_digest

    def bound(
        stage: AnonymousAudioStage,
        sealed: SealedAudio,
        observation: PersistenceObservation,
        destination: Path,
    ) -> _PublicationOutcome:
        custody = _PublicationCustody()
        outcome = _PublicationOutcome("failed")
        try:
            return function(
                stage,
                sealed,
                observation,
                destination,
                canonical_prepare,
                canonical_verify,
                custody,
                outcome,
            )
        except AnonymousAudioScrubError:
            raise
        except BaseException:
            if outcome.status in ("uncertain", "published"):
                try:
                    _finish_publication_ownership(
                        stage,
                        custody.temp,
                        custody.parent,
                        outcome,
                    )
                except AnonymousAudioScrubError:
                    raise
                except BaseException:
                    pass
                return _PublicationOutcome("uncertain")
            raise

    return bound


@_bind_publication_helpers
def _publish_one(
    stage: AnonymousAudioStage,
    sealed: SealedAudio,
    observation: PersistenceObservation,
    destination: Path,
    canonical_prepare: Callable[..., _PreparedPublicationCall],
    canonical_verify: Callable[[_PublicationTemp, SealedAudio], None],
    custody: _PublicationCustody,
    outcome: _PublicationOutcome,
) -> _PublicationOutcome:
    parent: _HeldParent | None = None
    temp: _PublicationTemp | None = None
    pending_stop: BaseException | None = None
    try:
        if sealed._output_format != destination.suffix.lower().lstrip("."):
            raise TTSPublishError(_PUBLISH_ERROR)
        _open_held_parent(destination, custody)
        parent = custody.parent
        if parent is None:
            raise TTSPublishError(_PUBLISH_ERROR)
        replacing = _authorize_destination(parent, destination.name)
        _create_publication_temp(parent, destination.name, custody)
        temp = custody.temp
        if temp is None:
            raise TTSPublishError(_PUBLISH_ERROR)
        _materialize_publication_temp(parent, temp)
        _copy_sealed_to_publication(stage, sealed, temp)
        _fsync_publication_file(temp.fd)
        _revalidate_parent(parent)
        _revalidate_publication_temp(parent, temp, sealed._size)
        if (
            _prepare_publication_call is not canonical_prepare
            or _verify_publication_digest is not canonical_verify
        ):
            raise TTSPublishError(_PUBLISH_ERROR)
        prepared = canonical_prepare(
            replacing=replacing,
            parent_fd=parent.fd,
            source=temp.name,
            destination=destination.name,
            platform=sys.platform,
        )
        native_callable = prepared.callable_ref
        source_name = prepared.source
        destination_name = prepared.destination
        source_dir_fd = prepared.source_dir_fd
        destination_dir_fd = prepared.destination_dir_fd
        flags = prepared.flags
        publication_kind = prepared.kind
        expected_argtypes = prepared.expected_argtypes
        expected_restype = prepared.expected_restype
        canonical_verify(temp, sealed)
        if publication_kind != "replace":
            try:
                signature_is_plain = (
                    tuple(native_callable.argtypes or ()) == expected_argtypes
                    and native_callable.restype is expected_restype
                    and getattr(native_callable, "errcheck", None) is None
                )
            except (AttributeError, TypeError, ValueError):
                signature_is_plain = False
            if not signature_is_plain:
                raise TTSPublishError(_PUBLISH_ERROR)
        outcome.status = "uncertain"
        temp.publication_armed = True
        if publication_kind == "replace":
            if (
                observation.current_policy is not PersistencePolicy.DURABLE
                or observation.ever_ephemeral
            ):
                outcome.status = "failed"
                raise TTSPublishError(_PUBLISH_ERROR)
            native_result = native_callable(
                source_name,
                destination_name,
                src_dir_fd=source_dir_fd,
                dst_dir_fd=destination_dir_fd,
            )
        else:
            ctypes.set_errno(0)
            if (
                observation.current_policy is not PersistencePolicy.DURABLE
                or observation.ever_ephemeral
            ):
                outcome.status = "failed"
                raise TTSPublishError(_PUBLISH_ERROR)
            native_result = native_callable(
                source_dir_fd,
                source_name,
                destination_dir_fd,
                destination_name,
                flags,
            )
        if publication_kind != "replace" and native_result != 0:
            error = ctypes.get_errno() or errno.EIO
            outcome.status = "failed"
            raise OSError(error, "durable publication unavailable")
        temp.published = True
        _fsync_parent(parent.fd)
        outcome.status = "published"
    except AnonymousAudioScrubError:
        raise
    except Exception:
        pass
    except BaseException as exc:
        pending_stop = exc
    finally:
        _finish_publication_ownership(stage, custody.temp, custody.parent, outcome)
    if pending_stop is not None and outcome.status != "uncertain":
        raise pending_stop
    return outcome


del _bind_publication_helpers


def _finish_publication_ownership(
    stage: AnonymousAudioStage,
    temp: _PublicationTemp | None,
    parent: _HeldParent | None,
    outcome: _PublicationOutcome,
) -> None:
    scrub_failed = False
    close_failed = False
    first_stop: BaseException | None = None
    if temp is not None:
        if temp.fd < 0:
            temp = None
    if temp is not None:
        preserve_temp = temp.published
        if temp.publication_armed and not temp.published:
            preserve_temp = True
            try:
                held = os.fstat(temp.fd)
                try:
                    source = os.stat(
                        temp.name,
                        dir_fd=parent.fd if parent is not None else None,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    source = None
                try:
                    destination = os.stat(
                        temp.destination_name,
                        dir_fd=parent.fd if parent is not None else None,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    destination = None
                source_is_held = source is not None and (
                    source.st_dev == held.st_dev and source.st_ino == held.st_ino
                )
                destination_is_held = destination is not None and (
                    destination.st_dev == held.st_dev
                    and destination.st_ino == held.st_ino
                )
                if source_is_held and not destination_is_held:
                    preserve_temp = False
            except BaseException:
                pass
        try:
            if preserve_temp:
                _close_owned_fd(temp)
            else:
                _scrub_and_close_owned_fd(temp)
        except BaseException as exc:
            if first_stop is None:
                first_stop = exc
            if preserve_temp:
                close_failed = True
                try:
                    _close_owned_fd(temp)
                except BaseException:
                    pass
            else:
                scrub_failed = True
                try:
                    _scrub_and_close_owned_fd(temp)
                except BaseException:
                    pass
    if parent is not None:
        try:
            _close_owned_fd(parent)
        except BaseException as exc:
            if first_stop is None:
                first_stop = exc
            close_failed = True
            try:
                _close_owned_fd(parent)
            except BaseException:
                pass
    try:
        stage.scrub_and_close()
    except BaseException as exc:
        if first_stop is None:
            first_stop = exc
        scrub_failed = True
        try:
            stage.scrub_and_close()
        except BaseException:
            pass
    if scrub_failed:
        raise AnonymousAudioScrubError(_SCRUB_ERROR)
    if close_failed:
        if first_stop is not None and not isinstance(first_stop, Exception):
            raise first_stop
        if outcome.status in ("published", "uncertain"):
            raise TTSPublishUncertain(_PUBLISH_UNCERTAIN)
        raise TTSPublishError(_PUBLISH_ERROR)
    if first_stop is not None:
        raise first_stop


def _scrub_all_stages(
    stages: tuple[tuple[AnonymousAudioStage, SealedAudio], ...],
) -> None:
    failed = False
    for stage, _sealed in stages:
        try:
            stage.scrub_and_close()
        except BaseException:
            failed = True
    if failed:
        raise AnonymousAudioScrubError(_SCRUB_ERROR)


def publish_durable(
    permit: DurablePublicationPermit,
    destination: Path,
) -> PublishedAudio:
    """Consume one live transaction permit and atomically publish one stage."""

    def consume(
        stages: tuple[tuple[AnonymousAudioStage, SealedAudio], ...],
        observation: PersistenceObservation,
    ) -> _PublicationOutcome:
        if len(stages) != 1:
            _scrub_all_stages(stages)
            return _PublicationOutcome("failed")
        authorized_destination = _validate_destination(destination)
        stage, sealed = stages[0]
        return _publish_one(stage, sealed, observation, authorized_destination)

    outcome = permit._consume_for_publication(consume)
    if type(outcome) is not _PublicationOutcome:
        raise TTSPublishError(_PUBLISH_ERROR)
    if outcome.status == "published":
        return PublishedAudio(path=destination)
    if outcome.status == "uncertain":
        raise TTSPublishUncertain(_PUBLISH_UNCERTAIN)
    raise TTSPublishError(_PUBLISH_ERROR)


__all__ = [
    "PublishedAudio",
    "TTSPublishError",
    "TTSPublishUncertain",
    "publish_durable",
]
