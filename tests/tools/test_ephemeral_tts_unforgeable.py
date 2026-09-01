"""Public regressions for descriptor-only ephemeral TTS authority."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy


VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00private-audio"


class _Adapter:
    def __init__(self, observed: list[str], *, acknowledgement: object = None) -> None:
        self.observed = observed
        self.acknowledgement = acknowledgement

    def generate(self, _request, sink):
        self.observed.append(sink.path)
        os.write(int(Path(sink.path).name), VALID_MP3)
        if self.acknowledgement == "same":
            return sink.path
        return self.acknowledgement

    def finish_owned_work(self) -> None:
        return None

    def stop_owned_work(self) -> None:
        return None


def _install(monkeypatch: pytest.MonkeyPatch, adapter: object) -> None:
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool, "_anonymous_adapter_for_provider", lambda *_a, **_k: adapter
    )
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})


def test_no_importable_old_named_tts_authority_or_cleanup_helpers_remain() -> None:
    from tools import tts_tool

    for name in (
        "_EphemeralTTSState",
        "_EPHEMERAL_TTS_STATE",
        "_cleanup_ephemeral_tts_state",
        "_LateRebindArtifact",
        "_RequestedDestinationAttestation",
    ):
        assert not hasattr(tts_tool, name)


@pytest.mark.parametrize("entry", ["single", "public"])
def test_caller_destination_never_selects_ephemeral_sink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    from tools import tts_tool

    observed: list[str] = []
    _install(monkeypatch, _Adapter(observed, acknowledgement="same"))
    destination = tmp_path / "caller.mp3"
    forged = SimpleNamespace(path=destination, cleanup=destination.unlink)
    monkeypatch.setattr(tts_tool, "_EPHEMERAL_TTS_STATE", forged, raising=False)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = (
            tts_tool._text_to_speech_single("private", str(destination), provider="edge")
            if entry == "single"
            else tts_tool.text_to_speech_tool("private", str(destination), provider="edge")
        )

    result = json.loads(rendered)
    assert result["success"] is True
    assert "file_path" not in result
    assert destination.exists() is False
    assert observed and all(
        path.startswith(("/dev/fd/", "/proc/self/fd/")) for path in observed
    )


@pytest.mark.parametrize("ack", [None, "same"])
def test_returned_none_or_exact_sink_never_becomes_publication_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ack: object
) -> None:
    from tools import tts_tool

    observed: list[str] = []
    _install(monkeypatch, _Adapter(observed, acknowledgement=ack))
    destination = tmp_path / "caller.mp3"
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = json.loads(
            tts_tool._text_to_speech_single("private", str(destination), provider="edge")
        )
    assert result["success"] is True
    assert destination.exists() is False
    assert observed[0] != str(destination)


@pytest.mark.parametrize("ack", ["different", 7, Path("outside")])
def test_returned_different_or_non_string_is_categorical_and_scrubbed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ack: object
) -> None:
    from tools import tts_tool

    observed: list[str] = []
    _install(monkeypatch, _Adapter(observed, acknowledgement=ack))
    destination = tmp_path / "caller.mp3"
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = json.loads(
            tts_tool._text_to_speech_single("private", str(destination), provider="edge")
        )
    assert result == {"success": False, "error": "TTS generation failed"}
    assert destination.exists() is False


def test_durable_late_ephemeral_observation_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import tts_tool

    class RebindingAdapter(_Adapter):
        def generate(self, request, sink):
            acknowledgement = super().generate(request, sink)
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                pass
            return acknowledgement

    destination = tmp_path / "caller.mp3"
    _install(monkeypatch, RebindingAdapter([], acknowledgement="same"))
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        result = json.loads(
            tts_tool._text_to_speech_single("private", str(destination), provider="edge")
        )
    assert result == {"success": False, "error": "TTS generation failed"}
    assert destination.exists() is False
