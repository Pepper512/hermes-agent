from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy
from tools.tts_adapters import ProviderRequest, builtin_adapter
from tools.tts_staging import _create_anonymous_audio_stage_for_test


VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00private-audio"


class RecordingAdapter:
    def __init__(self, seen: list[str]) -> None:
        self.seen = seen

    def generate(self, request, sink):
        self.seen.append(sink.path)
        os.write(int(Path(sink.path).name), VALID_MP3)
        return sink.path

    def finish_owned_work(self) -> None:
        return None

    def stop_owned_work(self) -> None:
        return None


def _install_recording_adapter(monkeypatch: pytest.MonkeyPatch, seen: list[str]) -> None:
    from tools import tts_tool

    monkeypatch.setattr(
        tts_tool,
        "_anonymous_adapter_for_provider",
        lambda *_args, **_kwargs: RecordingAdapter(seen),
        raising=False,
    )
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})


@pytest.mark.parametrize("entry", ["single", "public"])
def test_entry_ephemeral_never_gives_provider_caller_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    from tools import tts_tool

    seen: list[str] = []
    _install_recording_adapter(monkeypatch, seen)
    destination = tmp_path / "caller.mp3"

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if entry == "single":
            raw = tts_tool._text_to_speech_single(
                "private speech", str(destination), provider="edge"
            )
        else:
            raw = tts_tool.text_to_speech_tool(
                "private speech", str(destination), provider="edge"
            )

    result = json.loads(raw)
    assert result["success"] is True, result
    assert "file_path" not in result
    assert (tmp_path / "caller.mp3").exists() is False
    assert len(seen) == 1
    assert seen[0] != str(destination)
    assert seen[0].startswith(("/dev/fd/", "/proc/self/fd/"))
    payload = base64.b64decode(result["audio"].split(",", 1)[1], validate=True)
    assert payload == VALID_MP3


def test_public_ephemeral_multichunk_uses_one_path_free_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import tts_tool

    seen: list[str] = []
    _install_recording_adapter(monkeypatch, seen)
    monkeypatch.setattr(tts_tool, "_split_text_for_tts", lambda *_args: ["one", "two"])

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = json.loads(
            tts_tool.text_to_speech_tool(
                "private speech", str(tmp_path / "caller.mp3"), provider="edge"
            )
        )

    assert result["success"] is True, result
    assert result["chunk_count"] == 2
    assert len(result["audio_parts"]) == 2
    assert "file_path" not in result
    assert len(seen) == 2
    assert all(path.startswith(("/dev/fd/", "/proc/self/fd/")) for path in seen)
    assert (tmp_path / "caller.mp3").exists() is False


@pytest.mark.parametrize("provider", ["neutts", "piper", "kittentts"])
def test_named_only_builtin_rejects_before_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    from tools import tts_tool

    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": provider})
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = json.loads(
            tts_tool._text_to_speech_single(
                "private speech", str(tmp_path / "caller.mp3"), provider=provider
            )
        )
    assert result == {"success": False, "error": "TTS generation failed"}
    assert (tmp_path / "caller.mp3").exists() is False


@pytest.mark.parametrize(
    ("provider", "helper", "format_keyword"),
    [
        ("openai", "_generate_openai_tts", "response_format_override"),
        ("deepinfra", "_generate_deepinfra_tts", "response_format_override"),
        ("xai", "_generate_xai_tts", "output_format_override"),
        ("minimax", "_generate_minimax_tts", None),
        ("mistral", "_generate_mistral_tts", "response_format_override"),
        ("gemini", "_generate_gemini_tts", "output_format_override"),
    ],
)
def test_builtin_writes_only_anonymous_sink_with_explicit_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: str,
    helper: str,
    format_keyword: str | None,
) -> None:
    from tools import tts_tool

    observed: dict[str, object] = {}

    def writer(_text, sink_path, _config, **kwargs):
        observed.update(path=sink_path, kwargs=kwargs)
        os.write(int(Path(sink_path).name), VALID_MP3)
        return sink_path

    monkeypatch.setattr(tts_tool, helper, writer)
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    request = ProviderRequest("hello", None, None, None, None)
    try:
        acknowledgement = builtin_adapter(provider, {}).generate(request, stage.sink)
        sealed = stage.seal(acknowledgement)
        assert sealed._size == len(VALID_MP3)
        assert observed["path"] == stage.sink.path
        assert str(observed["path"]).startswith(("/dev/fd/", "/proc/self/fd/"))
        if format_keyword is not None:
            assert observed["kwargs"][format_keyword] == "mp3"
    finally:
        stage.scrub_and_close()


@pytest.mark.parametrize("entry", ["single", "public"])
@pytest.mark.parametrize("preexisting", [False, True])
def test_durable_absent_create_and_existing_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    preexisting: bool,
) -> None:
    from tools import tts_tool

    _install_recording_adapter(monkeypatch, [])
    destination = tmp_path / "caller.mp3"
    if preexisting:
        destination.write_bytes(b"old")
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = (
            tts_tool._text_to_speech_single("hello", str(destination), provider="edge")
            if entry == "single"
            else tts_tool.text_to_speech_tool("hello", str(destination), provider="edge")
        )
    result = json.loads(rendered)
    assert result["success"] is True, result
    assert destination.read_bytes() == VALID_MP3


def test_public_durable_all_chunks_seal_before_first_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tools import tts_publish, tts_tool

    seen: list[str] = []
    _install_recording_adapter(monkeypatch, seen)
    monkeypatch.setattr(tts_tool, "_split_text_for_tts", lambda *_args: ["one", "two"])
    real_publish = tts_publish.publish_durable_many
    observed_counts: list[int] = []

    def publish_after_all(permit, destinations):
        observed_counts.append(len(seen))
        return real_publish(permit, destinations)

    monkeypatch.setattr(tts_publish, "publish_durable_many", publish_after_all)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        result = json.loads(
            tts_tool.text_to_speech_tool(
                "hello", str(tmp_path / "voice.mp3"), provider="edge"
            )
        )
    assert result["success"] is True, result
    assert observed_counts == [2]


class _ForeignDestination:
    """A caller-controlled object that must never be coerced or inspected."""

    def __bool__(self):
        raise AssertionError("foreign destination truthiness was inspected")

    def __fspath__(self):
        raise AssertionError("foreign destination path protocol was inspected")

    def __str__(self):
        raise AssertionError("foreign destination string form was inspected")


@pytest.mark.parametrize("entry", ["single", "public"])
@pytest.mark.parametrize("policy", [PersistencePolicy.DURABLE, PersistencePolicy.EPHEMERAL])
@pytest.mark.parametrize(
    "destination_kind",
    ["traversal", "protected", "nul", "foreign", "invalid-extension"],
)
def test_destination_preflight_is_policy_first_and_provider_free_when_durable_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry: str,
    policy: PersistencePolicy,
    destination_kind: str,
) -> None:
    from tools import tts_tool
    import agent.file_safety as file_safety

    provider_paths: list[str] = []
    adapter_resolutions: list[str] = []
    adapter = RecordingAdapter(provider_paths)

    def resolve_adapter(*_args, **_kwargs):
        adapter_resolutions.append("resolved")
        return adapter

    monkeypatch.setattr(tts_tool, "_anonymous_adapter_for_provider", resolve_adapter)
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(file_safety, "is_write_approval_required", lambda _path: False)
    monkeypatch.setattr(
        file_safety,
        "is_write_denied",
        lambda path: destination_kind == "protected" and "private-token" in path,
    )

    marker = f"private-destination-{destination_kind}"
    if destination_kind == "traversal":
        destination = str(tmp_path / marker / ".." / "voice.mp3")
    elif destination_kind == "protected":
        destination = str(tmp_path / f"{marker}-private-token.mp3")
    elif destination_kind == "nul":
        destination = str(tmp_path / marker) + "\x00.mp3"
    elif destination_kind == "foreign":
        destination = _ForeignDestination()
    else:
        destination = str(tmp_path / f"{marker}.txt")

    with bind_persistence_policy(policy):
        rendered = (
            tts_tool._text_to_speech_single(
                "private speech", destination, provider="edge"
            )
            if entry == "single"
            else tts_tool.text_to_speech_tool(
                "private speech", destination, provider="edge"
            )
        )

    result = json.loads(rendered)
    if policy is PersistencePolicy.EPHEMERAL:
        assert result["success"] is True, result
        assert "file_path" not in result
        assert "MEDIA:" not in rendered
        assert provider_paths and all(
            path.startswith(("/dev/fd/", "/proc/self/fd/"))
            for path in provider_paths
        )
        assert adapter_resolutions == ["resolved"]
    else:
        expected_error = {
            "traversal": "TTS destination contains a traversal component",
            "protected": (
                "TTS destination targets a protected credential or system location"
            ),
        }.get(destination_kind, "TTS destination is invalid")
        assert result == {"success": False, "error": expected_error}
        assert provider_paths == []
        assert adapter_resolutions == []
        assert marker not in rendered
