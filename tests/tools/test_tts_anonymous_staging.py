from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile

import pytest

from tools.tts_staging import (
    AnonymousAudioStage,
    AnonymousAudioStageError,
    AnonymousAudioStageUnsupported,
    AnonymousAudioScrubError,
    ProviderAudioSink,
    _create_anonymous_audio_stage_for_test,
    _descriptor_path_for_platform,
)


VALID_AUDIO = {
    "mp3": b"ID3\x04\x00\x00\x00\x00\x00\x00payload",
    "wav": b"RIFF\x04\x00\x00\x00WAVEpayload",
    "ogg": b"OggS\x00payload",
    "flac": b"fLaCpayload",
}


def _test_stage(
    output_format: str, maximum_bytes: int, parent: Path
) -> AnonymousAudioStage:
    return _create_anonymous_audio_stage_for_test(
        output_format=output_format,
        maximum_bytes=maximum_bytes,
        parent=parent,
    )


def _sink_fd(stage: AnonymousAudioStage) -> int:
    return int(Path(stage.sink.path).name)


def _write_valid(stage: AnonymousAudioStage) -> None:
    os.write(_sink_fd(stage), VALID_AUDIO[stage.sink.output_format])


def test_provider_sink_exposes_only_descriptor_path_format_and_cap(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        sink = stage.sink
        assert isinstance(sink, ProviderAudioSink)
        assert set(vars(sink)) == {"path", "output_format", "maximum_bytes"}
        assert sink.output_format == "mp3"
        assert sink.maximum_bytes == 1024
        assert not any(
            hasattr(sink, forbidden)
            for forbidden in (
                "fd",
                "close",
                "scrub",
                "cleanup",
                "root",
                "destination",
            )
        )
    finally:
        stage.scrub_and_close()


def test_stage_unlinks_name_before_sink_is_observable(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        held = os.fstat(_sink_fd(stage))
        assert stat.S_ISREG(held.st_mode)
        assert stat.S_IMODE(held.st_mode) == 0o600
        assert held.st_uid == os.getuid()
        assert held.st_gid == os.getgid()
        assert held.st_nlink == 0
        roots = list(tmp_path.glob("hermes-tts-*"))
        assert len(roots) == 1
        root_stat = roots[0].stat()
        assert stat.S_IMODE(root_stat.st_mode) == 0o700
        assert root_stat.st_uid == os.getuid()
        assert root_stat.st_gid == os.getgid()
        assert all(not child.is_file() for child in tmp_path.rglob("*"))
    finally:
        stage.scrub_and_close()
    assert list(tmp_path.glob("hermes-tts-*")) == []


def test_stage_rejects_and_preserves_unexpected_root_namespace_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_fstat = os.fstat
    injected = False

    def fstat_then_inject(fd: int):
        nonlocal injected
        result = real_fstat(fd)
        if stat.S_ISREG(result.st_mode) and result.st_nlink == 0 and not injected:
            injected = True
            roots = list(tmp_path.glob("hermes-tts-*"))
            assert len(roots) == 1
            (roots[0] / "unproved-entry").touch(mode=0o600)
        return result

    monkeypatch.setattr(os, "fstat", fstat_then_inject)
    with pytest.raises(AnonymousAudioStageError):
        _test_stage("mp3", 1024, tmp_path)

    preserved = list(tmp_path.glob("hermes-tts-*/unproved-entry"))
    assert len(preserved) == 1


@pytest.mark.macos_only
def test_descriptor_path_is_dev_fd_on_darwin(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        assert stage.sink.path == f"/dev/fd/{_sink_fd(stage)}"
    finally:
        stage.scrub_and_close()


@pytest.mark.linux_only
def test_descriptor_path_is_proc_fd_on_linux(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        assert stage.sink.path == f"/proc/self/fd/{_sink_fd(stage)}"
    finally:
        stage.scrub_and_close()


def test_descriptor_path_resolver_rejects_unsupported_platform():
    with pytest.raises(AnonymousAudioStageUnsupported) as exc_info:
        _descriptor_path_for_platform(17, "win32")
    assert str(exc_info.value) == "tts_anonymous_stage_unsupported"


def test_unsupported_platform_rejects_before_root_creation(tmp_path: Path):
    before = set(tmp_path.iterdir())
    with pytest.raises(AnonymousAudioStageUnsupported):
        _create_anonymous_audio_stage_for_test(
            output_format="mp3",
            maximum_bytes=1024,
            parent=tmp_path,
            platform="win32",
        )
    assert set(tmp_path.iterdir()) == before


def test_production_temp_root_resolution_failure_is_fixed_and_path_free(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_tempdir() -> str:
        raise OSError("sensitive-temp-path")

    monkeypatch.setattr(tempfile, "gettempdir", fail_tempdir)
    with pytest.raises(AnonymousAudioStageError) as exc_info:
        AnonymousAudioStage.create("mp3", 1024)
    assert str(exc_info.value) == "tts_anonymous_stage_failed"
    assert "sensitive" not in str(exc_info.value)


@pytest.mark.parametrize("output_format", ["mp3", "wav", "ogg", "flac"])
def test_seal_accepts_none_ack_for_allowlisted_valid_audio(
    tmp_path: Path, output_format: str
):
    stage = _test_stage(output_format, 1024, tmp_path)
    try:
        _write_valid(stage)
        sealed = stage.seal(None)
        assert stage.read_bounded(sealed) == VALID_AUDIO[output_format]
    finally:
        stage.scrub_and_close()


def test_seal_accepts_exact_sink_ack(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        _write_valid(stage)
        sealed = stage.seal(stage.sink.path)
        assert stage.read_bounded(sealed) == VALID_AUDIO["mp3"]
    finally:
        stage.scrub_and_close()


def test_seal_rejects_object_ack_without_invoking_provider_equality(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    _write_valid(stage)

    class ProviderObject:
        def __eq__(self, other: object) -> bool:
            raise AssertionError("provider equality callback must not run")

    try:
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(ProviderObject())
    finally:
        stage.scrub_and_close()


@pytest.mark.parametrize(
    "acknowledgement", ["relative.mp3", "/tmp/different.mp3", b"not-a-path", False]
)
def test_seal_rejects_different_returned_path(
    tmp_path: Path, acknowledgement: object
):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        _write_valid(stage)
        with pytest.raises(AnonymousAudioStageError) as exc_info:
            stage.seal(acknowledgement)
        assert str(exc_info.value) == "tts_anonymous_stage_failed"
        assert stage.sink.path not in str(exc_info.value)
    finally:
        stage.scrub_and_close()


def test_seal_rejects_mode_drift(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        held_fd = _sink_fd(stage)
        _write_valid(stage)
        os.fchmod(held_fd, 0o640)
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("st_mode", stat.S_IFIFO | 0o600),
        ("st_uid", os.getuid() + 1),
        ("st_gid", os.getgid() + 1),
        ("st_nlink", 1),
    ],
)
def test_seal_rejects_nonregular_uid_gid_or_nlink_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: int,
):
    stage = _test_stage("mp3", 1024, tmp_path)
    held_fd = _sink_fd(stage)
    _write_valid(stage)
    real_fstat = os.fstat
    actual = real_fstat(held_fd)

    class DriftedStat:
        st_mode = actual.st_mode
        st_uid = actual.st_uid
        st_gid = actual.st_gid
        st_nlink = actual.st_nlink
        st_size = actual.st_size
        st_dev = actual.st_dev
        st_ino = actual.st_ino

    setattr(DriftedStat, field, replacement)

    def drift_target_only(fd: int):
        return DriftedStat() if fd == held_fd else real_fstat(fd)

    monkeypatch.setattr(os, "fstat", drift_target_only)
    try:
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        monkeypatch.setattr(os, "fstat", real_fstat)
        stage.scrub_and_close()


def test_seal_rejects_empty_audio(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


def test_seal_rejects_size_over_cap(tmp_path: Path):
    stage = _test_stage("mp3", 8, tmp_path)
    try:
        os.write(_sink_fd(stage), VALID_AUDIO["mp3"])
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


@pytest.mark.parametrize(
    ("output_format", "invalid_audio"),
    [
        ("mp3", b"not-mp3"),
        ("wav", b"RIFF\x04\x00\x00\x00NOPEpayload"),
        ("ogg", b"not-ogg"),
        ("flac", b"not-flac"),
    ],
)
def test_seal_rejects_invalid_format_signatures(
    tmp_path: Path, output_format: str, invalid_audio: bytes
):
    stage = _test_stage(output_format, 1024, tmp_path)
    try:
        os.write(_sink_fd(stage), invalid_audio)
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


def test_mp3_frame_sync_signature_is_accepted(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        frame = b"\xff\xfb\x90\x64payload"
        os.write(_sink_fd(stage), frame)
        sealed = stage.seal(None)
        assert stage.read_bounded(sealed) == frame
    finally:
        stage.scrub_and_close()


def test_read_rejects_same_size_mutation_after_seal(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        _write_valid(stage)
        sealed = stage.seal(None)
        held_fd = _sink_fd(stage)
        os.lseek(held_fd, -1, os.SEEK_END)
        os.write(held_fd, b"X")
        with pytest.raises(AnonymousAudioStageError):
            stage.read_bounded(sealed)
    finally:
        stage.scrub_and_close()


def test_seal_rejects_size_mutation_during_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage = _test_stage("mp3", 1024, tmp_path)
    held_fd = _sink_fd(stage)
    _write_valid(stage)
    real_read = os.read
    mutated = False

    def mutate_after_read(fd: int, count: int) -> bytes:
        nonlocal mutated
        data = real_read(fd, count)
        if fd == held_fd and not mutated:
            mutated = True
            os.lseek(fd, 0, os.SEEK_END)
            os.write(fd, b"X")
        return data

    monkeypatch.setattr(os, "read", mutate_after_read)
    try:
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


def test_sealed_audio_is_bound_to_its_own_stage(tmp_path: Path):
    first = _test_stage("mp3", 1024, tmp_path)
    second = _test_stage("mp3", 1024, tmp_path)
    try:
        _write_valid(first)
        _write_valid(second)
        first_sealed = first.seal(None)
        second.seal(None)
        with pytest.raises(AnonymousAudioStageError):
            second.read_bounded(first_sealed)
    finally:
        first.scrub_and_close()
        second.scrub_and_close()


def test_unsupported_format_and_invalid_cap_reject_before_materialization(
    tmp_path: Path,
):
    before = set(tmp_path.iterdir())
    for output_format, maximum_bytes in (
        ("aac", 1024),
        ("mp3", 0),
        ("mp3", True),
        ("mp3", 1.5),
        ("mp3", "1024"),
        ("mp3", None),
    ):
        with pytest.raises(AnonymousAudioStageError):
            _create_anonymous_audio_stage_for_test(
                output_format=output_format,
                maximum_bytes=maximum_bytes,  # type: ignore[arg-type]
                parent=tmp_path,
            )
        assert set(tmp_path.iterdir()) == before


def test_fixed_25_mib_cap_is_accepted(tmp_path: Path):
    stage = _test_stage("mp3", 25 * 1024 * 1024, tmp_path)
    stage.scrub_and_close()


@pytest.mark.parametrize("maximum_bytes", [25 * 1024 * 1024 + 1, 2**60])
def test_cap_above_fixed_25_mib_limit_rejects_before_root_creation(
    tmp_path: Path, maximum_bytes: int
):
    before = set(tmp_path.iterdir())
    stage = None
    try:
        with pytest.raises(AnonymousAudioStageError) as exc_info:
            stage = _test_stage("mp3", maximum_bytes, tmp_path)
        assert str(exc_info.value) == "tts_anonymous_stage_failed"
        assert set(tmp_path.iterdir()) == before
    finally:
        if stage is not None:
            stage.scrub_and_close()


def test_untrusted_format_object_rejects_without_invoking_hash(tmp_path: Path):
    before = set(tmp_path.iterdir())

    class ProviderFormat:
        def __hash__(self) -> int:
            raise AssertionError("untrusted format hash callback must not run")

    with pytest.raises(AnonymousAudioStageError):
        _create_anonymous_audio_stage_for_test(
            output_format=ProviderFormat(),  # type: ignore[arg-type]
            maximum_bytes=1024,
            parent=tmp_path,
        )
    assert set(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "invalid_mp3",
    [
        b"ID3",
        b"ID3\x04\x00\x00\x80\x00\x00\x00",
        b"\xff\xe0\x00\x00payload",
    ],
)
def test_seal_rejects_malformed_mp3_headers(tmp_path: Path, invalid_mp3: bytes):
    stage = _test_stage("mp3", 1024, tmp_path)
    try:
        os.write(_sink_fd(stage), invalid_mp3)
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


def test_seal_rejects_bad_flac_signature(tmp_path: Path):
    stage = _test_stage("flac", 1024, tmp_path)
    try:
        os.write(_sink_fd(stage), b"not-flac")
        with pytest.raises(AnonymousAudioStageError):
            stage.seal(None)
    finally:
        stage.scrub_and_close()


def test_failed_initial_unlink_scrubs_held_inode_and_preserves_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    real_ftruncate = os.ftruncate
    real_fsync = os.fsync
    real_close = os.close
    original_inode: tuple[int, int] | None = None
    replacement_basename: str | None = None
    unlink_attempts = 0
    held_operations: list[str] = []

    def is_original(fd: int) -> bool:
        if original_inode is None:
            return False
        try:
            current = os.fstat(fd)
        except OSError:
            return False
        return (current.st_dev, current.st_ino) == original_inode

    def record_ftruncate(fd: int, length: int):
        if is_original(fd):
            held_operations.append("ftruncate")
        return real_ftruncate(fd, length)

    def record_fsync(fd: int):
        if is_original(fd):
            held_operations.append("fsync")
        return real_fsync(fd)

    def record_close(fd: int):
        if is_original(fd):
            held_operations.append("close")
        return real_close(fd)

    def move_replace_and_raise(path: str, *, dir_fd: int):
        nonlocal original_inode, replacement_basename, unlink_attempts
        unlink_attempts += 1
        if unlink_attempts == 1:
            writer_fd = os.open(path, os.O_WRONLY | os.O_NOFOLLOW, dir_fd=dir_fd)
            os.write(writer_fd, b"private-audio")
            original = os.fstat(writer_fd)
            original_inode = (original.st_dev, original.st_ino)
            replacement_basename = path
            real_close(writer_fd)
            os.rename(
                path,
                "moved-original",
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
            replacement_fd = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                0o600,
                dir_fd=dir_fd,
            )
            os.write(replacement_fd, b"replacement")
            real_close(replacement_fd)
        raise OSError("injected unlink failure")

    supported_dir_fd = set(os.supports_dir_fd)
    supported_dir_fd.add(move_replace_and_raise)
    monkeypatch.setattr(os, "supports_dir_fd", supported_dir_fd)
    monkeypatch.setattr(os, "unlink", move_replace_and_raise)
    monkeypatch.setattr(os, "ftruncate", record_ftruncate)
    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "close", record_close)

    with pytest.raises(AnonymousAudioStageError) as exc_info:
        _test_stage("mp3", 1024, tmp_path)

    assert str(exc_info.value) == "tts_anonymous_stage_failed"
    assert "injected" not in str(exc_info.value)
    assert unlink_attempts == 1
    assert held_operations == ["ftruncate", "fsync", "close"]
    roots = list(tmp_path.glob("hermes-tts-*"))
    assert len(roots) == 1
    assert (roots[0] / "moved-original").read_bytes() == b""
    assert replacement_basename is not None
    assert (roots[0] / replacement_basename).read_bytes() == b"replacement"


def test_scrub_targets_held_inode_only(tmp_path: Path):
    stage = _test_stage("wav", 64, tmp_path)
    held_fd = _sink_fd(stage)
    os.write(held_fd, b"private-audio")
    stage.scrub_and_close()
    with pytest.raises(OSError):
        os.fstat(held_fd)
    stage.scrub_and_close()


def test_scrub_calls_ftruncate_fsync_close_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage = _test_stage("mp3", 1024, tmp_path)
    held_fd = _sink_fd(stage)
    os.write(held_fd, b"private-audio")
    calls: list[str] = []
    real_ftruncate = os.ftruncate
    real_fsync = os.fsync
    real_close = os.close

    def record_ftruncate(fd: int, length: int):
        if fd == held_fd:
            calls.append("ftruncate")
        return real_ftruncate(fd, length)

    def record_fsync(fd: int):
        if fd == held_fd:
            calls.append("fsync")
        return real_fsync(fd)

    def record_close(fd: int):
        if fd == held_fd:
            calls.append("close")
        return real_close(fd)

    monkeypatch.setattr(os, "ftruncate", record_ftruncate)
    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "close", record_close)
    stage.scrub_and_close()
    assert calls == ["ftruncate", "fsync", "close"]


def test_scrub_failure_never_unlinks_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage = _test_stage("mp3", 1024, tmp_path)
    held_fd = _sink_fd(stage)
    os.write(held_fd, b"private-audio")

    def fail_ftruncate(fd: int, length: int):
        raise OSError("injected truncation failure")

    def forbidden_unlink(*args, **kwargs):
        raise AssertionError("scrub must not unlink a pathname")

    monkeypatch.setattr(os, "ftruncate", fail_ftruncate)
    monkeypatch.setattr(os, "unlink", forbidden_unlink)
    with pytest.raises(AnonymousAudioScrubError) as exc_info:
        stage.scrub_and_close()
    assert str(exc_info.value) == "tts_anonymous_scrub_failed"
    assert "private" not in str(exc_info.value)
    with pytest.raises(OSError):
        os.fstat(held_fd)


def test_close_failure_can_be_retried_without_path_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage = _test_stage("mp3", 1024, tmp_path)
    held_fd = _sink_fd(stage)
    os.write(held_fd, b"private-audio")
    real_close = os.close

    def fail_target_close(fd: int):
        if fd == held_fd:
            raise OSError("injected close failure")
        return real_close(fd)

    monkeypatch.setattr(os, "close", fail_target_close)
    with pytest.raises(AnonymousAudioScrubError):
        stage.scrub_and_close()
    assert os.fstat(held_fd).st_size == 0

    monkeypatch.setattr(os, "close", real_close)
    stage.scrub_and_close()
    with pytest.raises(OSError):
        os.fstat(held_fd)


def test_root_namespace_drift_preserves_all_unproved_names(tmp_path: Path):
    stage = _test_stage("mp3", 1024, tmp_path)
    roots = list(tmp_path.glob("hermes-tts-*"))
    assert len(roots) == 1
    original_root = roots[0]
    moved_root = tmp_path / "moved-root"
    original_root.rename(moved_root)
    original_root.mkdir(mode=0o700)

    stage.scrub_and_close()

    assert original_root.is_dir()
    assert moved_root.is_dir()


def _assert_process_exit_closes_anonymous_inode_without_residue(tmp_path: Path):
    script = """
import os
import sys
from pathlib import Path
from tools.tts_staging import _create_anonymous_audio_stage_for_test

parent = Path(sys.argv[1])
stage = _create_anonymous_audio_stage_for_test("mp3", 1024, parent)
fd = int(Path(stage.sink.path).name)
os.write(fd, b"ID3\\x04\\x00\\x00\\x00\\x00\\x00\\x00payload")
os._exit(0)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert all(not child.is_file() for child in tmp_path.rglob("*"))


@pytest.mark.macos_only
def test_process_exit_closes_anonymous_inode_without_residue_macos(tmp_path: Path):
    _assert_process_exit_closes_anonymous_inode_without_residue(tmp_path)


@pytest.mark.linux_only
def test_process_exit_closes_anonymous_inode_without_residue_linux(tmp_path: Path):
    _assert_process_exit_closes_anonymous_inode_without_residue(tmp_path)
