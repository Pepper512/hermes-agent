"""Contracts for the pre-cutover anonymous TTS provider adapters."""

from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError, fields
from types import MappingProxyType

import pytest

from agent.tts_provider import TTSProvider
from tools.tts_staging import ProviderAudioSink
from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    _descriptor_number_from_sink_path,
    _generate_edge_tts_to_sink,
    _generate_elevenlabs_to_sink,
    _render_command_tts_sink_template,
    _text_to_speech_single,
    text_to_speech_tool,
)
from tools.tts_adapters import (
    AnonymousSinkUnsupported,
    BUILTIN_SINK_DISPOSITIONS,
    ProviderAcknowledgementError,
    ProviderRequest,
    SinkDisposition,
    command_adapter,
    plugin_adapter,
)


SINK = ProviderAudioSink(
    path="/dev/fd/41",
    output_format="mp3",
    maximum_bytes=4096,
)
REQUEST = ProviderRequest(
    text="hello",
    voice="voice-a",
    model="model-a",
    speed=1.25,
    instructions="calm",
)


class LegacyNamedProvider(TTSProvider):
    synthesize_calls: list[tuple[object, ...]]

    def __init__(self) -> None:
        self.synthesize_calls = []

    @property
    def name(self) -> str:
        return "legacy"

    def synthesize(self, text: str, output_path: str, **kwargs: object) -> str:
        self.synthesize_calls.append((text, output_path, kwargs))
        return output_path


class SinkPlugin(LegacyNamedProvider):
    seen: tuple[object, ...] | None = None

    def synthesize_to_sink(
        self,
        text: str,
        sink_path: str,
        *,
        voice: str | None = None,
        model: str | None = None,
        speed: float | None = None,
        format: str,
        maximum_bytes: int,
        **extra: object,
    ) -> str:
        self.seen = (
            text,
            sink_path,
            voice,
            model,
            speed,
            format,
            maximum_bytes,
            extra,
        )
        return sink_path


def test_provider_request_is_immutable_and_contains_no_authority_fields():
    assert {field.name for field in fields(ProviderRequest)} == {
        "text",
        "voice",
        "model",
        "speed",
        "instructions",
    }
    with pytest.raises(FrozenInstanceError):
        REQUEST.text = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="invalid TTS provider request"):
        ProviderRequest("hello", None, None, float("nan"), None)


def test_every_builtin_has_explicit_sink_disposition():
    assert isinstance(BUILTIN_SINK_DISPOSITIONS, MappingProxyType)
    assert set(BUILTIN_SINK_DISPOSITIONS) == set(BUILTIN_TTS_PROVIDERS)
    assert {
        name
        for name, disposition in BUILTIN_SINK_DISPOSITIONS.items()
        if disposition is SinkDisposition.ADAPT
    } == {
        "edge",
        "elevenlabs",
        "openai",
        "deepinfra",
        "xai",
        "minimax",
        "mistral",
        "gemini",
    }
    assert {
        name
        for name, disposition in BUILTIN_SINK_DISPOSITIONS.items()
        if disposition is SinkDisposition.REJECT_NAMED_PATH
    } == {"neutts", "piper", "kittentts"}


def test_legacy_named_plugin_rejects_before_synthesize():
    provider = LegacyNamedProvider()
    with pytest.raises(AnonymousSinkUnsupported):
        plugin_adapter(provider).generate(REQUEST, SINK)
    assert provider.synthesize_calls == []


def test_sink_plugin_receives_only_fd_path_and_format():
    provider = SinkPlugin()
    ack = plugin_adapter(provider).generate(REQUEST, SINK)
    assert ack == SINK.path
    assert provider.seen == (
        REQUEST.text,
        SINK.path,
        REQUEST.voice,
        REQUEST.model,
        REQUEST.speed,
        SINK.output_format,
        SINK.maximum_bytes,
        {"instructions": REQUEST.instructions},
    )
    assert provider.synthesize_calls == []


def test_plugin_acknowledgement_has_no_path_authority():
    class BadAck(SinkPlugin):
        def synthesize_to_sink(self, *args: object, **kwargs: object) -> str:
            return "/tmp/other.mp3"

    with pytest.raises(ProviderAcknowledgementError):
        plugin_adapter(BadAck()).generate(REQUEST, SINK)

    class ForgedAck:
        def __eq__(self, other: object) -> bool:
            return other == SINK.path

    class NonStringAck(SinkPlugin):
        def synthesize_to_sink(self, *args: object, **kwargs: object) -> object:
            return ForgedAck()

    with pytest.raises(ProviderAcknowledgementError):
        plugin_adapter(NonStringAck()).generate(REQUEST, SINK)


def test_edge_adapter_receives_sink_not_destination(monkeypatch):
    seen: dict[str, object] = {}

    class Communicate:
        def __init__(self, text: str, **kwargs: object) -> None:
            seen["request"] = (text, kwargs)

        async def save(self, path: str) -> None:
            seen["path"] = path

    class EdgeModule:
        pass

    EdgeModule.Communicate = Communicate

    monkeypatch.setattr("tools.tts_tool._import_edge_tts", lambda: EdgeModule)
    monkeypatch.setattr("tools.tts_tool.os.path.getsize", lambda path: 12)

    ack = asyncio.run(
        _generate_edge_tts_to_sink(
            REQUEST.text,
            SINK.path,
            {},
            output_format=SINK.output_format,
            maximum_bytes=SINK.maximum_bytes,
            voice=REQUEST.voice,
            speed=REQUEST.speed,
        )
    )

    assert ack == SINK.path
    assert seen["path"] == SINK.path
    assert "/tmp" not in str(seen)

    from tools.tts_adapters import builtin_adapter

    assert builtin_adapter("edge", {}).generate(REQUEST, SINK) == SINK.path


def test_elevenlabs_adapter_writes_sink_with_explicit_format(monkeypatch):
    seen: dict[str, object] = {}
    written = bytearray()

    class Speech:
        def convert(self, **kwargs: object):
            seen["convert"] = kwargs
            return [b"one", b"two"]

    class Client:
        def __init__(self, **kwargs: object) -> None:
            self.text_to_speech = Speech()

    class Output:
        def __enter__(self):
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def write(self, chunk: bytes) -> None:
            written.extend(chunk)

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a: "key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: Client)
    monkeypatch.setattr("builtins.open", lambda path, mode: Output())

    ack = _generate_elevenlabs_to_sink(
        REQUEST.text,
        SINK.path,
        {},
        output_format="opus",
        maximum_bytes=SINK.maximum_bytes,
        voice=REQUEST.voice,
        model=REQUEST.model,
    )

    assert ack == SINK.path
    assert written == b"onetwo"
    assert seen["convert"]["output_format"] == "opus_48000_64"

    from tools.tts_adapters import builtin_adapter

    opus_sink = ProviderAudioSink(SINK.path, "opus", SINK.maximum_bytes)
    assert builtin_adapter("elevenlabs", {}).generate(REQUEST, opus_sink) == SINK.path


def test_command_template_uses_fd_path_and_format():
    rendered = _render_command_tts_sink_template(
        "writer --input {input_path} --output {output_path} "
        "--format {format} --cap {maximum_bytes}",
        input_path="/private/input.txt",
        sink=SINK,
        voice="voice-a",
        model="model-a",
        speed="1.25",
    )
    assert SINK.path in rendered
    assert "--format mp3" in rendered
    assert "--cap 4096" in rendered


def test_command_adapter_never_expands_destination():
    destination = "/private/caller/final.mp3"
    rendered = _render_command_tts_sink_template(
        "writer {output_path} {format}",
        input_path="/private/input.txt",
        sink=SINK,
        voice="",
        model="",
        speed="",
    )
    assert destination not in rendered
    assert SINK.path in rendered


def test_command_adapter_is_parallel_and_fails_closed_before_task4(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.tts_tool._generate_command_tts_to_sink",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    adapter = command_adapter(
        "local-writer",
        {"type": "command", "command": "writer {output_path}"},
        {},
    )
    with pytest.raises(AnonymousSinkUnsupported):
        adapter.generate(REQUEST, SINK)
    assert calls == []


def test_descriptor_number_is_derived_only_from_canonical_sink_path():
    assert _descriptor_number_from_sink_path("/dev/fd/41") == 41
    assert _descriptor_number_from_sink_path("/proc/self/fd/41") == 41
    for invalid in ("41", "/tmp/41", "/dev/fd/041", "/dev/fd/-1"):
        with pytest.raises(ValueError, match="invalid anonymous TTS sink"):
            _descriptor_number_from_sink_path(invalid)


def _assert_named_or_sibling_provider_rejected(provider: str) -> None:
    from tools.tts_adapters import builtin_adapter

    with pytest.raises(AnonymousSinkUnsupported):
        builtin_adapter(provider, {}).generate(REQUEST, SINK)


def test_neutts_named_path_rejected():
    _assert_named_or_sibling_provider_rejected("neutts")


def test_piper_sibling_path_rejected():
    _assert_named_or_sibling_provider_rejected("piper")


def test_kittentts_sibling_path_rejected():
    _assert_named_or_sibling_provider_rejected("kittentts")


def test_unknown_unclassified_provider_fails_closed():
    from tools.tts_adapters import builtin_adapter

    with pytest.raises(AnonymousSinkUnsupported):
        builtin_adapter("unknown", {})


def test_sink_adapters_are_unreachable_from_public_tts_before_task7():
    forbidden = {
        "ProviderAudioSink",
        "ProviderRequest",
        "builtin_adapter",
        "plugin_adapter",
        "command_adapter",
        "BUILTIN_SINK_DISPOSITIONS",
        "_generate_command_tts_to_sink",
        "_generate_edge_tts_to_sink",
        "_generate_elevenlabs_to_sink",
        "_render_command_tts_sink_template",
    }
    assert forbidden.isdisjoint(_text_to_speech_single.__code__.co_names)
    assert forbidden.isdisjoint(text_to_speech_tool.__code__.co_names)
