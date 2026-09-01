from __future__ import annotations

import errno
import os
from pathlib import Path
import sys

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy
from tools import tts_publish
from tools.tts_publish import TTSPublishError, publish_durable
from tools.tts_staging import _create_anonymous_audio_stage_for_test
from tools.tts_transaction import TTSTransaction


pytestmark = pytest.mark.macos_only
VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00darwin-audio"


def _publish(tmp_path: Path, destination: Path):
    stage_parent = tmp_path / "stage"
    stage_parent.mkdir()
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, stage_parent)
    os.write(int(Path(stage.sink.path).name), VALID_MP3)
    sealed = stage.seal(stage.sink.path)
    published = None
    error = None
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            try:
                published = publish_durable(transaction.decide(), destination)
            except TTSPublishError as exc:
                error = exc
    if error is not None:
        raise error
    return published


def test_darwin_absent_uses_renameatx_np_required_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    if sys.platform != "darwin":
        pytest.skip("Darwin runtime required")
    calls: list[tuple[int, bytes, int, bytes, int]] = []
    real = tts_publish._darwin_renameatx_np

    def record(src_fd, src, dst_fd, dst, flags):
        calls.append((src_fd, src, dst_fd, dst, flags))
        return real(src_fd, src, dst_fd, dst, flags)

    monkeypatch.setattr(tts_publish, "_darwin_renameatx_np", record)
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        source_fd = os.open(
            "source",
            os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW,
            0o600,
            dir_fd=parent_fd,
        )
        os.close(source_fd)
        tts_publish._rename_absent_darwin(
            parent_fd, "source", parent_fd, "voice.mp3"
        )
    finally:
        os.close(parent_fd)
    assert len(calls) == 1
    assert calls[0][4] == (
        tts_publish.RENAME_EXCL
        | tts_publish.RENAME_NOFOLLOW_ANY
        | tts_publish.RENAME_RESOLVE_BENEATH
    )


def test_darwin_missing_flag_fails_closed_before_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    if sys.platform != "darwin":
        pytest.skip("Darwin runtime required")
    monkeypatch.setattr(tts_publish, "RENAME_RESOLVE_BENEATH", None)
    with pytest.raises(TTSPublishError):
        _publish(tmp_path, tmp_path / "voice.mp3")
    assert not list(tmp_path.glob(".hermes-tts-publish-*"))


def test_darwin_ctypes_conversion_hook_is_rejected_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    if sys.platform != "darwin":
        pytest.skip("Darwin runtime required")
    symbol = tts_publish._DARWIN_RENAMEATX_NP
    assert symbol is not None

    class PolicyHook:
        @classmethod
        def from_param(cls, value):
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                pass
            return value

    monkeypatch.setattr(
        symbol,
        "argtypes",
        [
            PolicyHook,
            tts_publish.ctypes.c_char_p,
            tts_publish.ctypes.c_int,
            tts_publish.ctypes.c_char_p,
            tts_publish.ctypes.c_uint,
        ],
    )
    destination = tmp_path / "voice.mp3"
    with pytest.raises(TTSPublishError):
        _publish(tmp_path, destination)
    assert not destination.exists()


def test_darwin_existing_uses_replace(tmp_path: Path):
    if sys.platform != "darwin":
        pytest.skip("Darwin runtime required")
    destination = tmp_path / "voice.mp3"
    destination.write_bytes(b"old")
    _publish(tmp_path, destination)
    assert destination.read_bytes() == VALID_MP3


def test_darwin_absent_collision_preserves_concurrent_destination(tmp_path: Path):
    if sys.platform != "darwin":
        pytest.skip("Darwin runtime required")
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        first = os.open("first", os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=parent_fd)
        os.close(first)
        (tmp_path / "final").write_bytes(b"concurrent")
        with pytest.raises(OSError) as exc_info:
            tts_publish._rename_absent_darwin(parent_fd, "first", parent_fd, "final")
        assert exc_info.value.errno == errno.EEXIST
        assert (tmp_path / "final").read_bytes() == b"concurrent"
    finally:
        os.close(parent_fd)
