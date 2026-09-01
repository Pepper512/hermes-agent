"""Anonymous descriptor staging for TTS provider output.

Provider code receives only an already-unlinked descriptor path and bounded
format metadata.  The trusted transaction retains descriptor ownership and is
the only code allowed to seal, read, scrub, or close the staged audio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import secrets
import signal
import stat
import sys
import tempfile
import threading
from typing import Final


_ALLOWED_FORMATS: Final[frozenset[str]] = frozenset({
    "mp3",
    "wav",
    "ogg",
    "flac",
    "m4a",
    "aac",
    "amr",
    "opus",
})
MAX_ANONYMOUS_AUDIO_BYTES: Final[int] = 25 * 1024 * 1024
_MAX_SIGNATURE_BYTES: Final[int] = 64 * 1024
_AMR_NB_FRAME_BITS: Final[tuple[int, ...]] = (
    95,
    103,
    118,
    134,
    148,
    159,
    204,
    244,
    39,
)
_AMR_WB_FRAME_BITS: Final[tuple[int, ...]] = (
    132,
    177,
    253,
    285,
    317,
    365,
    397,
    461,
    477,
    40,
)
_STAGE_ERROR: Final[str] = "tts_anonymous_stage_failed"
_UNSUPPORTED_ERROR: Final[str] = "tts_anonymous_stage_unsupported"
_SCRUB_ERROR: Final[str] = "tts_anonymous_scrub_failed"
_CONSTRUCTION_TOKEN = object()
_SEALED_TOKEN = object()


class AnonymousAudioStageError(RuntimeError):
    """Fixed, path-free anonymous-stage failure."""


class AnonymousAudioStageUnsupported(AnonymousAudioStageError):
    """The current host cannot provide the required anonymous sink contract."""


class AnonymousAudioScrubError(AnonymousAudioStageError):
    """Held-descriptor destruction did not complete cleanly."""


@dataclass(slots=True)
class _OwnedDescriptor:
    fd: int = -1
    stat: os.stat_result | None = None
    state: str = "empty"
    close_proof: int | None = None
    close_active: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)


def _require_signal_deferral_support():
    masker = getattr(signal, "pthread_sigmask", None)
    valid_signals = getattr(signal, "valid_signals", None)
    if not callable(masker) or not callable(valid_signals):
        raise AnonymousAudioStageUnsupported(_UNSUPPORTED_ERROR)
    excluded = {
        candidate
        for candidate in (
            getattr(signal, "SIGKILL", None),
            getattr(signal, "SIGSTOP", None),
        )
        if candidate is not None
    }
    catchable = frozenset(valid_signals()) - excluded
    if not catchable:
        raise AnonymousAudioStageUnsupported(_UNSUPPORTED_ERROR)
    return masker, catchable


def _open_owned_descriptor(owner: _OwnedDescriptor, *args, **kwargs) -> None:
    masker, catchable = _require_signal_deferral_support()
    previous_mask = masker(signal.SIG_BLOCK, catchable)
    try:
        owner.fd = os.open(*args, **kwargs)
        owner.state = "open"
        owner.stat = os.fstat(owner.fd)
    finally:
        masker(signal.SIG_SETMASK, previous_mask)


def _same_owned_open_description(owner: _OwnedDescriptor) -> bool:
    if owner.stat is None or owner.close_proof is None or owner.fd < 0:
        return False
    try:
        held = os.fstat(owner.fd)
        offset = os.lseek(owner.fd, 0, os.SEEK_CUR)
    except (OSError, TypeError, ValueError):
        return False
    expected = owner.stat
    return (
        held.st_dev == expected.st_dev
        and held.st_ino == expected.st_ino
        and stat.S_IFMT(held.st_mode) == stat.S_IFMT(expected.st_mode)
        and held.st_uid == expected.st_uid
        and held.st_gid == expected.st_gid
        and offset == owner.close_proof
    )


def _close_owned_descriptor(owner: _OwnedDescriptor) -> None:
    masker, catchable = _require_signal_deferral_support()
    with owner.lock:
        owns_attempt = False
        try:
            if owner.close_active:
                return
            owns_attempt = True
            owner.close_active = True
            if owner.state in ("empty", "released"):
                return
            if owner.state == "open":
                if owner.stat is None:
                    owner.stat = os.fstat(owner.fd)
                proof = (1 << 30) | secrets.randbits(30)
                os.lseek(owner.fd, proof, os.SEEK_SET)
                owner.close_proof = proof
                owner.state = "attempted"
            elif not _same_owned_open_description(owner):
                owner.state = "released"
                owner.fd = -1
                return
            previous_mask = masker(signal.SIG_BLOCK, catchable)
            try:
                os.close(owner.fd)
                owner.state = "released"
                owner.fd = -1
            finally:
                masker(signal.SIG_SETMASK, previous_mask)
        finally:
            if owns_attempt:
                owner.close_active = False


def _create_provider_audio_sink_boundary():
    """Create the stage-only sink issuer and read-only validation boundary."""

    issuer_identity = object()

    class ProviderAudioSink:
        """Immutable provider view issued only by ``AnonymousAudioStage``."""

        __slots__ = (
            "__issuer_identity",
            "__maximum_bytes",
            "__output_format",
            "__path",
        )

        def __new__(cls, *args: object, **kwargs: object):
            raise TypeError("ProviderAudioSink is issued only by AnonymousAudioStage")

        def __init_subclass__(cls, **kwargs: object) -> None:
            raise TypeError("ProviderAudioSink cannot be subclassed")

        def __setattr__(self, name: str, value: object) -> None:
            raise TypeError("ProviderAudioSink is immutable")

        def __reduce__(self):
            raise TypeError("ProviderAudioSink cannot be reconstructed")

        def __reduce_ex__(self, protocol: int):
            raise TypeError("ProviderAudioSink cannot be reconstructed")

        def __copy__(self):
            raise TypeError("ProviderAudioSink cannot be copied")

        def __deepcopy__(self, memo: object):
            raise TypeError("ProviderAudioSink cannot be copied")

        @property
        def path(self) -> str:
            return object.__getattribute__(self, "_ProviderAudioSink__path")

        @property
        def output_format(self) -> str:
            return object.__getattribute__(
                self, "_ProviderAudioSink__output_format"
            )

        @property
        def maximum_bytes(self) -> int:
            return object.__getattribute__(
                self, "_ProviderAudioSink__maximum_bytes"
            )

    def issue(path: str, output_format: str, maximum_bytes: int) -> ProviderAudioSink:
        _validate_request(output_format, maximum_bytes)
        if type(path) is not str or not path:
            raise AnonymousAudioStageError(_STAGE_ERROR)
        sink = object.__new__(ProviderAudioSink)
        object.__setattr__(
            sink,
            "_ProviderAudioSink__issuer_identity",
            issuer_identity,
        )
        object.__setattr__(sink, "_ProviderAudioSink__path", path)
        object.__setattr__(
            sink,
            "_ProviderAudioSink__output_format",
            output_format,
        )
        object.__setattr__(
            sink,
            "_ProviderAudioSink__maximum_bytes",
            maximum_bytes,
        )
        return sink

    def validate(sink: object) -> tuple[str, str, int]:
        if type(sink) is not ProviderAudioSink:
            raise AnonymousAudioStageError(_STAGE_ERROR)
        try:
            identity = object.__getattribute__(
                sink, "_ProviderAudioSink__issuer_identity"
            )
            path = object.__getattribute__(sink, "_ProviderAudioSink__path")
            output_format = object.__getattribute__(
                sink, "_ProviderAudioSink__output_format"
            )
            maximum_bytes = object.__getattribute__(
                sink, "_ProviderAudioSink__maximum_bytes"
            )
        except (AttributeError, TypeError):
            raise AnonymousAudioStageError(_STAGE_ERROR) from None
        if identity is not issuer_identity or type(path) is not str or not path:
            raise AnonymousAudioStageError(_STAGE_ERROR)
        _validate_request(output_format, maximum_bytes)
        return path, output_format, maximum_bytes

    return ProviderAudioSink, issue, validate


(
    ProviderAudioSink,
    _temporary_issue_provider_audio_sink,
    _validate_provider_audio_sink,
) = _create_provider_audio_sink_boundary()


def _capture_provider_audio_sink_issuer(issuer):
    """Bind the issuer into the stage factory without a durable module name."""

    def decorate(function):
        def wrapped(cls, *args, **kwargs):
            return function(cls, *args, _sink_issuer=issuer, **kwargs)

        return wrapped

    return decorate


@dataclass(frozen=True)
class SealedAudio:
    """Opaque proof that one stage passed its held-descriptor checks."""

    _token: object
    _authority: object
    _fd: int
    _device: int
    _digest: bytes
    _inode: int
    _size: int
    _output_format: str

    def __post_init__(self) -> None:
        if self._token is not _SEALED_TOKEN:
            raise AnonymousAudioStageError(_STAGE_ERROR)


def _descriptor_path_for_platform(fd: int, platform: str) -> str:
    """Resolve the native descriptor namespace without mutating host globals."""

    if not isinstance(fd, int) or isinstance(fd, bool) or fd < 0:
        raise AnonymousAudioStageError(_STAGE_ERROR)
    if platform == "darwin":
        return f"/dev/fd/{fd}"
    if platform.startswith("linux"):
        return f"/proc/self/fd/{fd}"
    raise AnonymousAudioStageUnsupported(_UNSUPPORTED_ERROR)


def _validate_request(output_format: str, maximum_bytes: int) -> None:
    if type(output_format) is not str or output_format not in _ALLOWED_FORMATS:
        raise AnonymousAudioStageError(_STAGE_ERROR)
    if (
        not isinstance(maximum_bytes, int)
        or isinstance(maximum_bytes, bool)
        or maximum_bytes <= 0
        or maximum_bytes > MAX_ANONYMOUS_AUDIO_BYTES
    ):
        raise AnonymousAudioStageError(_STAGE_ERROR)


def _current_posix_identity() -> tuple[int, int] | None:
    """Return the current numeric owner only when both POSIX APIs are usable."""

    get_uid = getattr(os, "getuid", None)
    get_gid = getattr(os, "getgid", None)
    if not callable(get_uid) or not callable(get_gid):
        return None
    try:
        uid = get_uid()
        gid = get_gid()
    except (OSError, TypeError, ValueError):
        return None
    if (
        type(uid) is not int
        or uid < 0
        or type(gid) is not int
        or gid < 0
    ):
        return None
    return uid, gid


def _require_host_capabilities(platform: str) -> tuple[int, int]:
    _descriptor_path_for_platform(0, platform)
    _require_signal_deferral_support()
    required_flags = ("O_DIRECTORY", "O_NOFOLLOW")
    if os.name != "posix" or any(not hasattr(os, name) for name in required_flags):
        raise AnonymousAudioStageUnsupported(_UNSUPPORTED_ERROR)
    if os.open not in os.supports_dir_fd or os.unlink not in os.supports_dir_fd:
        raise AnonymousAudioStageUnsupported(_UNSUPPORTED_ERROR)
    identity = _current_posix_identity()
    if identity is None:
        raise AnonymousAudioStageUnsupported(_UNSUPPORTED_ERROR)
    return identity


def _valid_format_signature(
    output_format: str, header: bytes, *, total_size: int
) -> bool:
    if output_format == "mp3":
        if header.startswith(b"ID3"):
            return (
                len(header) >= 10
                and header[3] in (2, 3, 4)
                and header[4] != 0xFF
                and all(size_byte & 0x80 == 0 for size_byte in header[6:10])
            )
        if len(header) < 4 or header[0] != 0xFF or header[1] & 0xE0 != 0xE0:
            return False
        version = (header[1] >> 3) & 0x03
        layer = (header[1] >> 1) & 0x03
        bitrate_index = (header[2] >> 4) & 0x0F
        sample_rate_index = (header[2] >> 2) & 0x03
        return (
            version != 0x01
            and layer != 0x00
            and bitrate_index not in (0x00, 0x0F)
            and sample_rate_index != 0x03
        )
    if output_format == "wav":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    if output_format == "ogg":
        return header.startswith(b"OggS")
    if output_format == "flac":
        return header.startswith(b"fLaC")
    if output_format == "m4a":
        if len(header) < 16 or header[4:8] != b"ftyp":
            return False
        box_size = int.from_bytes(header[:4], "big")
        return 16 <= box_size <= total_size
    if output_format == "aac":
        if len(header) < 7 or header[0] != 0xFF or header[1] & 0xF0 != 0xF0:
            return False
        layer = (header[1] >> 1) & 0x03
        sample_rate_index = (header[2] >> 2) & 0x0F
        header_size = 7 if header[1] & 0x01 else 9
        frame_length = ((header[3] & 0x03) << 11) | (header[4] << 3) | (header[5] >> 5)
        return (
            layer == 0
            and sample_rate_index < 0x0D
            and len(header) >= header_size
            and header_size <= frame_length <= total_size
        )
    if output_format == "amr":
        return _valid_amr_first_frame(header)
    if output_format == "opus":
        return _valid_opus_first_page(header, total_size=total_size)
    return False


def _valid_amr_first_frame(header: bytes) -> bool:
    if header.startswith(b"#!AMR-WB\n"):
        magic = b"#!AMR-WB\n"
        frame_bits = _AMR_WB_FRAME_BITS
    elif header.startswith(b"#!AMR\n"):
        magic = b"#!AMR\n"
        frame_bits = _AMR_NB_FRAME_BITS
    else:
        return False

    if len(header) <= len(magic):
        return False
    frame_header = header[len(magic)]
    if frame_header & 0x83 or not frame_header & 0x04:
        return False
    frame_type = (frame_header >> 3) & 0x0F
    if frame_type >= len(frame_bits):
        return False

    payload_bits = frame_bits[frame_type]
    payload_octets = (payload_bits + 7) // 8
    frame_end = len(magic) + 1 + payload_octets
    if len(header) < frame_end:
        return False
    unused_bits = payload_octets * 8 - payload_bits
    if unused_bits and header[frame_end - 1] & ((1 << unused_bits) - 1):
        return False
    return True


def _valid_opus_first_page(header: bytes, *, total_size: int) -> bool:
    if len(header) < 27 or header[:5] != b"OggS\x00":
        return False
    if header[5] & 0x01 or not header[5] & 0x02:
        return False
    segment_count = header[26]
    segment_table_end = 27 + segment_count
    if segment_count == 0 or len(header) < segment_table_end:
        return False
    segment_sizes = header[27:segment_table_end]
    page_end = segment_table_end + sum(segment_sizes)
    if page_end > len(header) or page_end > total_size:
        return False
    first_packet_size = 0
    packet_terminated = False
    for segment_size in segment_sizes:
        first_packet_size += segment_size
        if segment_size < 255:
            packet_terminated = True
            break
    if not packet_terminated or first_packet_size < 19:
        return False
    first_packet = header[segment_table_end : segment_table_end + first_packet_size]
    if not first_packet.startswith(b"OpusHead") or first_packet[8] != 1:
        return False
    channels = first_packet[9]
    mapping_family = first_packet[18]
    if mapping_family == 0:
        return len(first_packet) == 19 and channels in (1, 2)
    if (
        channels == 0
        or (mapping_family == 1 and channels > 8)
        or len(first_packet) != 21 + channels
    ):
        return False
    stream_count = first_packet[19]
    coupled_count = first_packet[20]
    if stream_count == 0 or coupled_count > stream_count:
        return False
    decoded_channels = stream_count + coupled_count
    if decoded_channels > 255:
        return False
    return all(
        mapping == 255 or mapping < decoded_channels for mapping in first_packet[21:]
    )


class AnonymousAudioStage:
    """Trusted owner of one already-unlinked TTS output descriptor."""

    __slots__ = (
        "_audio_owner",
        "_authority",
        "_cleanup_lock",
        "_closed",
        "_fd",
        "_initial_device",
        "_initial_inode",
        "_parent_fd",
        "_parent_owner",
        "_root_basename",
        "_root_device",
        "_root_fd",
        "_root_owner",
        "_root_gid",
        "_root_inode",
        "_root_uid",
        "_sealed",
        "_sink",
    )

    def __init__(
        self,
        token: object,
        *,
        audio_owner: _OwnedDescriptor,
        sink: ProviderAudioSink,
        parent_owner: _OwnedDescriptor,
        root_owner: _OwnedDescriptor,
        root_basename: str,
        root_stat: os.stat_result,
        file_stat: os.stat_result,
    ) -> None:
        if token is not _CONSTRUCTION_TOKEN:
            raise AnonymousAudioStageError(_STAGE_ERROR)
        self._audio_owner = audio_owner
        self._authority = object()
        self._cleanup_lock = threading.RLock()
        self._closed = False
        self._fd = audio_owner.fd
        self._initial_device = file_stat.st_dev
        self._initial_inode = file_stat.st_ino
        self._parent_fd = parent_owner.fd
        self._parent_owner = parent_owner
        self._root_basename = root_basename
        self._root_device = root_stat.st_dev
        self._root_gid = root_stat.st_gid
        self._root_inode = root_stat.st_ino
        self._root_uid = root_stat.st_uid
        self._root_fd = root_owner.fd
        self._root_owner = root_owner
        self._sealed = False
        self._sink = sink

    @classmethod
    def create(cls, output_format: str, maximum_bytes: int) -> "AnonymousAudioStage":
        """Create a production stage under the host temporary directory."""

        return cls._create_unlinked(
            output_format=output_format,
            maximum_bytes=maximum_bytes,
            parent=None,
            platform=sys.platform,
        )

    @classmethod
    @_capture_provider_audio_sink_issuer(_temporary_issue_provider_audio_sink)
    def _create_unlinked(
        cls,
        *,
        output_format: str,
        maximum_bytes: int,
        parent: Path | None,
        platform: str,
        _sink_issuer,
    ) -> "AnonymousAudioStage":
        _validate_request(output_format, maximum_bytes)
        owner_uid, owner_gid = _require_host_capabilities(platform)

        try:
            parent_path = (
                Path(tempfile.gettempdir()).resolve(strict=True)
                if parent is None
                else Path(parent)
            )
        except (OSError, TypeError, ValueError):
            raise AnonymousAudioStageError(_STAGE_ERROR) from None
        parent_owner = _OwnedDescriptor()
        root_owner = _OwnedDescriptor()
        audio_owner = _OwnedDescriptor()
        root_basename: str | None = None
        audio_basename: str | None = None
        try:
            parent_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            _open_owned_descriptor(parent_owner, parent_path, parent_flags)
            parent_fd = parent_owner.fd
            parent_stat = os.fstat(parent_fd)
            if not stat.S_ISDIR(parent_stat.st_mode):
                raise AnonymousAudioStageError(_STAGE_ERROR)

            root_basename = Path(
                tempfile.mkdtemp(prefix="hermes-tts-", dir=parent_path)
            ).name
            _open_owned_descriptor(
                root_owner,
                root_basename,
                parent_flags,
                dir_fd=parent_fd,
            )
            root_fd = root_owner.fd
            os.fchown(root_fd, owner_uid, owner_gid)
            os.fchmod(root_fd, 0o700)
            root_stat = os.fstat(root_fd)
            named_root = os.stat(
                root_basename,
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if not cls._valid_root_identity(root_stat, named_root):
                raise AnonymousAudioStageError(_STAGE_ERROR)

            audio_basename = f"audio-{secrets.token_hex(16)}"
            audio_flags = os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW
            _open_owned_descriptor(
                audio_owner,
                audio_basename,
                audio_flags,
                0o600,
                dir_fd=root_fd,
            )
            audio_fd = audio_owner.fd
            os.fchown(audio_fd, owner_uid, owner_gid)
            os.fchmod(audio_fd, 0o600)
            file_stat = os.fstat(audio_fd)
            if not cls._valid_held_file(file_stat, require_unlinked=False):
                raise AnonymousAudioStageError(_STAGE_ERROR)

            os.unlink(audio_basename, dir_fd=root_fd)
            file_stat = os.fstat(audio_fd)
            if not cls._valid_held_file(file_stat, require_unlinked=True):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            if os.listdir(root_fd):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            os.lseek(audio_fd, 0, os.SEEK_SET)

            sink = _sink_issuer(
                _descriptor_path_for_platform(audio_fd, platform),
                output_format,
                maximum_bytes,
            )
            return cls(
                _CONSTRUCTION_TOKEN,
                audio_owner=audio_owner,
                sink=sink,
                parent_owner=parent_owner,
                root_owner=root_owner,
                root_basename=root_basename,
                root_stat=root_stat,
                file_stat=file_stat,
            )
        except AnonymousAudioStageError:
            cls._close_failed_creation(
                audio_owner=audio_owner,
                root_owner=root_owner,
                parent_owner=parent_owner,
            )
            raise
        except (OSError, ValueError, TypeError):
            cls._close_failed_creation(
                audio_owner=audio_owner,
                root_owner=root_owner,
                parent_owner=parent_owner,
            )
            raise AnonymousAudioStageError(_STAGE_ERROR) from None
        except BaseException:
            cls._close_failed_creation(
                audio_owner=audio_owner,
                root_owner=root_owner,
                parent_owner=parent_owner,
            )
            raise

    @staticmethod
    def _valid_root_identity(held: os.stat_result, named: os.stat_result) -> bool:
        identity = _current_posix_identity()
        if identity is None:
            return False
        owner_uid, owner_gid = identity
        return (
            stat.S_ISDIR(held.st_mode)
            and stat.S_IMODE(held.st_mode) == 0o700
            and held.st_uid == owner_uid
            and held.st_gid == owner_gid
            and held.st_dev == named.st_dev
            and held.st_ino == named.st_ino
        )

    @staticmethod
    def _valid_held_file(held: os.stat_result, *, require_unlinked: bool) -> bool:
        identity = _current_posix_identity()
        if identity is None:
            return False
        owner_uid, owner_gid = identity
        return (
            stat.S_ISREG(held.st_mode)
            and stat.S_IMODE(held.st_mode) == 0o600
            and held.st_uid == owner_uid
            and held.st_gid == owner_gid
            and (held.st_nlink == 0 if require_unlinked else held.st_nlink == 1)
        )

    @staticmethod
    def _close_failed_creation(
        *,
        audio_owner: _OwnedDescriptor,
        root_owner: _OwnedDescriptor,
        parent_owner: _OwnedDescriptor,
    ) -> None:
        failed = False
        if audio_owner.state not in ("empty", "released"):
            try:
                os.ftruncate(audio_owner.fd, 0)
            except BaseException:
                failed = True
            try:
                os.fsync(audio_owner.fd)
            except BaseException:
                failed = True
            try:
                _close_owned_descriptor(audio_owner)
            except BaseException:
                failed = True
                try:
                    _close_owned_descriptor(audio_owner)
                except BaseException:
                    pass
        if root_owner.state not in ("empty", "released"):
            try:
                _close_owned_descriptor(root_owner)
            except BaseException:
                failed = True
                try:
                    _close_owned_descriptor(root_owner)
                except BaseException:
                    pass
        if parent_owner.state not in ("empty", "released"):
            try:
                _close_owned_descriptor(parent_owner)
            except BaseException:
                failed = True
                try:
                    _close_owned_descriptor(parent_owner)
                except BaseException:
                    pass
        if failed:
            raise AnonymousAudioScrubError(_SCRUB_ERROR)

    @property
    def sink(self) -> ProviderAudioSink:
        return self._sink

    def seal(self, provider_acknowledgement: object) -> SealedAudio:
        if self._closed or self._sealed:
            raise AnonymousAudioStageError(_STAGE_ERROR)
        if provider_acknowledgement is not None and (
            type(provider_acknowledgement) is not str
            or provider_acknowledgement != self._sink.path
        ):
            raise AnonymousAudioStageError(_STAGE_ERROR)
        try:
            os.fsync(self._fd)
            held = os.fstat(self._fd)
            if not self._valid_final_stat(held):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            os.lseek(self._fd, 0, os.SEEK_SET)
            header = os.read(self._fd, min(held.st_size, _MAX_SIGNATURE_BYTES))
            os.lseek(self._fd, 0, os.SEEK_SET)
            if not _valid_format_signature(
                self._sink.output_format,
                header,
                total_size=held.st_size,
            ):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            digest = self._digest_held_bytes(held.st_size)
            after_digest = os.fstat(self._fd)
            if (
                not self._valid_final_stat(after_digest)
                or after_digest.st_dev != held.st_dev
                or after_digest.st_ino != held.st_ino
                or after_digest.st_size != held.st_size
            ):
                raise AnonymousAudioStageError(_STAGE_ERROR)
        except AnonymousAudioStageError:
            raise
        except (OSError, ValueError, TypeError):
            raise AnonymousAudioStageError(_STAGE_ERROR) from None

        self._sealed = True
        return SealedAudio(
            _SEALED_TOKEN,
            self._authority,
            self._fd,
            held.st_dev,
            digest,
            held.st_ino,
            held.st_size,
            self._sink.output_format,
        )

    def _valid_final_stat(self, held: os.stat_result) -> bool:
        return (
            self._valid_held_file(held, require_unlinked=True)
            and held.st_dev == self._initial_device
            and held.st_ino == self._initial_inode
            and 0 < held.st_size <= self._sink.maximum_bytes
        )

    def read_bounded(self, sealed: SealedAudio) -> bytes:
        if (
            self._closed
            or not self._sealed
            or not isinstance(sealed, SealedAudio)
            or sealed._authority is not self._authority
            or sealed._fd != self._fd
            or sealed._output_format != self._sink.output_format
        ):
            raise AnonymousAudioStageError(_STAGE_ERROR)
        try:
            before = os.fstat(self._fd)
            if not self._matches_seal(before, sealed):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            os.lseek(self._fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = sealed._size
            while remaining:
                chunk = os.read(self._fd, min(remaining, 64 * 1024))
                if not chunk:
                    raise AnonymousAudioStageError(_STAGE_ERROR)
                chunks.append(chunk)
                remaining -= len(chunk)
            after = os.fstat(self._fd)
            if not self._matches_seal(after, sealed):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            data = b"".join(chunks)
            if hashlib.sha256(data).digest() != sealed._digest:
                raise AnonymousAudioStageError(_STAGE_ERROR)
            if not _valid_format_signature(
                sealed._output_format,
                data[:_MAX_SIGNATURE_BYTES],
                total_size=len(data),
            ):
                raise AnonymousAudioStageError(_STAGE_ERROR)
            return data
        except AnonymousAudioStageError:
            raise
        except (OSError, ValueError, TypeError):
            raise AnonymousAudioStageError(_STAGE_ERROR) from None

    def _digest_held_bytes(self, size: int) -> bytes:
        digest = hashlib.sha256()
        os.lseek(self._fd, 0, os.SEEK_SET)
        remaining = size
        while remaining:
            chunk = os.read(self._fd, min(remaining, 64 * 1024))
            if not chunk:
                raise AnonymousAudioStageError(_STAGE_ERROR)
            digest.update(chunk)
            remaining -= len(chunk)
        os.lseek(self._fd, 0, os.SEEK_SET)
        return digest.digest()

    def _matches_seal(self, held: os.stat_result, sealed: SealedAudio) -> bool:
        return (
            self._valid_final_stat(held)
            and held.st_dev == sealed._device
            and held.st_ino == sealed._inode
            and held.st_size == sealed._size
            and held.st_size <= self._sink.maximum_bytes
        )

    def scrub_and_close(self) -> None:
        with self._cleanup_lock:
            failed = False
            first_stop: BaseException | None = None
            if self._audio_owner.state not in ("empty", "released"):
                try:
                    os.ftruncate(self._audio_owner.fd, 0)
                except OSError:
                    failed = True
                except BaseException as exc:
                    first_stop = exc
                try:
                    os.fsync(self._audio_owner.fd)
                except OSError:
                    failed = True
                except BaseException as exc:
                    if first_stop is None:
                        first_stop = exc
                try:
                    _close_owned_descriptor(self._audio_owner)
                except OSError:
                    failed = True
                except BaseException as exc:
                    if first_stop is None:
                        first_stop = exc

            self._sync_descriptor_fields()
            try:
                self._teardown_exact_empty_root()
            except BaseException as exc:
                if first_stop is None:
                    first_stop = exc
            try:
                self._close_root_descriptors()
            except AnonymousAudioScrubError:
                failed = True
            except BaseException as exc:
                if first_stop is None:
                    first_stop = exc
            self._sync_descriptor_fields()

            if failed:
                raise AnonymousAudioScrubError(_SCRUB_ERROR)
            if first_stop is not None:
                raise first_stop

    def _sync_descriptor_fields(self) -> None:
        self._fd = self._audio_owner.fd
        self._root_fd = self._root_owner.fd
        self._parent_fd = self._parent_owner.fd
        self._closed = self._audio_owner.state == "released"

    def _teardown_exact_empty_root(self) -> None:
        if (
            self._root_owner.state in ("empty", "released")
            or self._parent_owner.state in ("empty", "released")
        ):
            return
        try:
            held = os.fstat(self._root_owner.fd)
            named = os.stat(
                self._root_basename,
                dir_fd=self._parent_owner.fd,
                follow_symlinks=False,
            )
            exact = (
                self._valid_root_identity(held, named)
                and held.st_dev == self._root_device
                and held.st_ino == self._root_inode
                and held.st_uid == self._root_uid
                and held.st_gid == self._root_gid
                and os.listdir(self._root_owner.fd) == []
            )
            if exact:
                os.rmdir(self._root_basename, dir_fd=self._parent_owner.fd)
        except OSError:
            pass

    def _close_root_descriptors(self) -> None:
        failed = False
        first_stop: BaseException | None = None
        if self._root_owner.state not in ("empty", "released"):
            try:
                _close_owned_descriptor(self._root_owner)
            except OSError:
                failed = True
            except BaseException as exc:
                first_stop = exc
        if self._parent_owner.state not in ("empty", "released"):
            try:
                _close_owned_descriptor(self._parent_owner)
            except OSError:
                failed = True
            except BaseException as exc:
                if first_stop is None:
                    first_stop = exc
        self._sync_descriptor_fields()
        if failed:
            raise AnonymousAudioScrubError(_SCRUB_ERROR)
        if first_stop is not None:
            raise first_stop


del _temporary_issue_provider_audio_sink
del _capture_provider_audio_sink_issuer
del _create_provider_audio_sink_boundary


def _create_anonymous_audio_stage_for_test(
    output_format: str,
    maximum_bytes: int,
    parent: Path,
    *,
    platform: str | None = None,
) -> AnonymousAudioStage:
    """Inject only the temporary parent for hermetic filesystem tests."""

    return AnonymousAudioStage._create_unlinked(
        output_format=output_format,
        maximum_bytes=maximum_bytes,
        parent=parent,
        platform=sys.platform if platform is None else platform,
    )
