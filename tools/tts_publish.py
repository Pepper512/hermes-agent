"""Atomic durable publication for anonymously staged TTS audio.

This module is intentionally not wired to the public TTS entry points.  It
consumes the one-shot transaction permit, copies one sealed descriptor into a
provider-invisible same-filesystem temporary, and publishes through the exact
host primitive authorized by the approved design.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import errno
import hashlib
import os
from pathlib import Path
import secrets
import stat
import sys
from typing import Final

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
    stat: os.stat_result


@dataclass(slots=True)
class _PublicationTemp:
    fd: int
    name: str
    stat: os.stat_result
    published: bool = False


@dataclass(frozen=True, slots=True)
class _PublicationOutcome:
    status: str


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
    return destination


def _open_held_parent(destination: Path) -> _HeldParent:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = -1
    try:
        fd = os.open(destination.parent, flags)
        held = os.fstat(fd)
        named = os.stat(destination.parent, follow_symlinks=False)
        if not _same_parent(held, named):
            raise TTSPublishError(_PUBLISH_ERROR)
        return _HeldParent(fd=fd, path=destination.parent, stat=held)
    except TTSPublishError:
        if fd >= 0:
            os.close(fd)
        raise
    except (OSError, TypeError, ValueError):
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        raise TTSPublishError(_PUBLISH_ERROR) from None


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


def _create_publication_temp(parent: _HeldParent) -> _PublicationTemp:
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
        try:
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
            return _PublicationTemp(fd=fd, name=name, stat=held)
        except BaseException:
            try:
                _scrub_and_close_fd(fd)
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


def _final_policy_allows_publication(observation: PersistenceObservation) -> bool:
    return (
        observation.current_policy is PersistencePolicy.DURABLE
        and not observation.ever_ephemeral
    )


def _fsync_publication_file(fd: int) -> None:
    os.fsync(fd)


def _fsync_parent(fd: int) -> None:
    os.fsync(fd)


def _scrub_and_close_fd(fd: int) -> None:
    failed = False
    try:
        os.ftruncate(fd, 0)
    except OSError:
        failed = True
    try:
        os.fsync(fd)
    except OSError:
        failed = True
    try:
        os.close(fd)
    except OSError:
        failed = True
    if failed:
        raise AnonymousAudioScrubError(_SCRUB_ERROR)


def _close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        raise TTSPublishUncertain(_PUBLISH_UNCERTAIN) from None


def _publish_one(
    stage: AnonymousAudioStage,
    sealed: SealedAudio,
    observation: PersistenceObservation,
    destination: Path,
) -> _PublicationOutcome:
    parent: _HeldParent | None = None
    temp: _PublicationTemp | None = None
    outcome = _PublicationOutcome("failed")
    try:
        if sealed._output_format != destination.suffix.lower().lstrip("."):
            raise TTSPublishError(_PUBLISH_ERROR)
        parent = _open_held_parent(destination)
        replacing = _authorize_destination(parent, destination.name)
        temp = _create_publication_temp(parent)
        _copy_sealed_to_publication(stage, sealed, temp)
        _fsync_publication_file(temp.fd)
        _revalidate_parent(parent)
        _revalidate_publication_temp(parent, temp, sealed._size)
        publisher = _replace_existing if replacing else _rename_absent_for_host
        if not _final_policy_allows_publication(observation):
            raise TTSPublishError(_PUBLISH_ERROR)
        publisher(parent.fd, temp.name, parent.fd, destination.name)
        temp.published = True
        try:
            _fsync_parent(parent.fd)
        except OSError:
            outcome = _PublicationOutcome("uncertain")
        else:
            outcome = _PublicationOutcome("published")
    except AnonymousAudioScrubError:
        raise
    except Exception:
        outcome = _PublicationOutcome("failed")
    finally:
        _finish_publication_ownership(stage, temp, parent, outcome)
    return outcome


def _finish_publication_ownership(
    stage: AnonymousAudioStage,
    temp: _PublicationTemp | None,
    parent: _HeldParent | None,
    outcome: _PublicationOutcome,
) -> None:
    scrub_failed = False
    close_failed = False
    if temp is not None:
        try:
            if temp.published:
                os.close(temp.fd)
            else:
                _scrub_and_close_fd(temp.fd)
        except AnonymousAudioScrubError:
            scrub_failed = True
        except OSError:
            close_failed = True
    if parent is not None:
        try:
            os.close(parent.fd)
        except OSError:
            close_failed = True
    try:
        stage.scrub_and_close()
    except BaseException:
        scrub_failed = True
    if scrub_failed:
        raise AnonymousAudioScrubError(_SCRUB_ERROR)
    if close_failed:
        if outcome.status in ("published", "uncertain"):
            raise TTSPublishUncertain(_PUBLISH_UNCERTAIN)
        raise TTSPublishError(_PUBLISH_ERROR)


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
