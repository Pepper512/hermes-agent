"""Contracts for the pre-cutover anonymous TTS provider adapters."""

from __future__ import annotations

import asyncio
import copy
from dataclasses import FrozenInstanceError, fields
import os
import pickle
from pathlib import Path
from types import MappingProxyType

import pytest

from agent.tts_provider import TTSProvider
from tools.tts_staging import ProviderAudioSink
from tools.tts_staging import (
    AnonymousAudioStageError,
    _create_anonymous_audio_stage_for_test,
)
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


REQUEST = ProviderRequest(
    text="hello",
    voice="voice-a",
    model="model-a",
    speed=1.25,
    instructions="calm",
)


@pytest.fixture
def sink_factory(tmp_path: Path):
    stages = []

    def create(
        output_format: str = "mp3", maximum_bytes: int = 4096
    ) -> ProviderAudioSink:
        stage = _create_anonymous_audio_stage_for_test(
            output_format, maximum_bytes, tmp_path
        )
        stages.append(stage)
        return stage.sink

    yield create

    for stage in reversed(stages):
        stage.scrub_and_close()


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


def test_stage_issued_sink_is_not_publicly_constructible(tmp_path: Path):
    with pytest.raises(TypeError):
        ProviderAudioSink("/dev/fd/41", "mp3", 4096)

    with pytest.raises(TypeError):
        class SinkSubclass(ProviderAudioSink):
            pass

    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    try:
        for reconstruct in (
            lambda: copy.copy(stage.sink),
            lambda: copy.deepcopy(stage.sink),
            lambda: pickle.loads(pickle.dumps(stage.sink)),
        ):
            with pytest.raises((TypeError, pickle.PickleError)):
                reconstruct()
    finally:
        stage.scrub_and_close()


def test_sink_issuer_is_not_importable_but_validator_copies_trusted_fields(
    tmp_path: Path,
):
    import tools.tts_staging as staging

    assert not hasattr(staging, "_temporary_issue_provider_audio_sink")
    assert not hasattr(staging, "_create_provider_audio_sink_boundary")
    assert not hasattr(staging, "_capture_provider_audio_sink_issuer")

    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    try:
        assert staging._validate_provider_audio_sink(stage.sink) == (
            stage.sink.path,
            "mp3",
            4096,
        )
    finally:
        stage.scrub_and_close()


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


def test_legacy_named_plugin_rejects_before_synthesize(sink_factory):
    sink = sink_factory()
    provider = LegacyNamedProvider()
    with pytest.raises(AnonymousSinkUnsupported):
        plugin_adapter(provider).generate(REQUEST, sink)
    assert provider.synthesize_calls == []


def test_sink_plugin_receives_only_fd_path_and_format(sink_factory):
    sink = sink_factory()
    provider = SinkPlugin()
    ack = plugin_adapter(provider).generate(REQUEST, sink)
    assert ack == sink.path
    assert provider.seen == (
        REQUEST.text,
        sink.path,
        REQUEST.voice,
        REQUEST.model,
        REQUEST.speed,
        sink.output_format,
        sink.maximum_bytes,
        {"instructions": REQUEST.instructions},
    )
    assert provider.synthesize_calls == []


def test_plugin_acknowledgement_has_no_path_authority(sink_factory):
    sink = sink_factory()
    class BadAck(SinkPlugin):
        def synthesize_to_sink(self, *args: object, **kwargs: object) -> str:
            return "/tmp/other.mp3"

    with pytest.raises(ProviderAcknowledgementError):
        plugin_adapter(BadAck()).generate(REQUEST, sink)

    class ForgedAck:
        def __eq__(self, other: object) -> bool:
            return other == sink.path

    class NonStringAck(SinkPlugin):
        def synthesize_to_sink(self, *args: object, **kwargs: object) -> object:
            return ForgedAck()

    with pytest.raises(ProviderAcknowledgementError):
        plugin_adapter(NonStringAck()).generate(REQUEST, sink)


def test_str_subclass_acknowledgement_rejects_without_equality(sink_factory):
    sink = sink_factory()
    equality_calls = []

    class SideEffectString(str):
        def __eq__(self, other: object) -> bool:
            equality_calls.append(other)
            return True

    class SubclassAck(SinkPlugin):
        def synthesize_to_sink(self, *args: object, **kwargs: object) -> str:
            return SideEffectString(sink.path)

    with pytest.raises(ProviderAcknowledgementError):
        plugin_adapter(SubclassAck()).generate(REQUEST, sink)
    assert equality_calls == []


def test_every_sink_consumer_rejects_unissued_object_before_field_access(
    tmp_path: Path,
    monkeypatch,
):
    field_reads = []
    provider_calls = []

    class Lookalike:
        @property
        def path(self):
            field_reads.append("path")
            return "/dev/fd/41"

        @property
        def output_format(self):
            field_reads.append("format")
            return "mp3"

        @property
        def maximum_bytes(self):
            field_reads.append("cap")
            return 4096

    class RecordingPlugin(SinkPlugin):
        def synthesize_to_sink(self, *args: object, **kwargs: object) -> str:
            provider_calls.append("plugin")
            return "/dev/fd/41"

    monkeypatch.setattr(
        "tools.tts_tool._generate_edge_tts_to_sink",
        lambda *args, **kwargs: provider_calls.append("edge"),
    )
    monkeypatch.setattr(
        "tools.tts_tool._generate_elevenlabs_to_sink",
        lambda *args, **kwargs: provider_calls.append("elevenlabs"),
    )

    command_config = {
        "type": "command",
        "command": "writer {output_path}",
        "output_format": "mp3",
    }
    consumers = (
        lambda sink: plugin_adapter(RecordingPlugin()).generate(REQUEST, sink),
        lambda sink: __import__("tools.tts_adapters", fromlist=["builtin_adapter"])
        .builtin_adapter("edge", {})
        .generate(REQUEST, sink),
        lambda sink: __import__("tools.tts_adapters", fromlist=["builtin_adapter"])
        .builtin_adapter("elevenlabs", {})
        .generate(REQUEST, sink),
        lambda sink: __import__("tools.tts_adapters", fromlist=["builtin_adapter"])
        .builtin_adapter("openai", {})
        .generate(REQUEST, sink),
        lambda sink: __import__("tools.tts_adapters", fromlist=["builtin_adapter"])
        .builtin_adapter("piper", {})
        .generate(REQUEST, sink),
        lambda sink: command_adapter("local", command_config, {}).generate(REQUEST, sink),
        lambda sink: _render_command_tts_sink_template(
            command_config["command"],
            input_path=str(tmp_path / "input.txt"),
            sink=sink,
            voice="",
            model="",
            speed="",
        ),
        lambda sink: __import__("tools.tts_tool", fromlist=["_generate_command_tts_to_sink"])
        ._generate_command_tts_to_sink(
            "hello", sink, "local", command_config, {}
        ),
    )
    forged_sinks = (object.__new__(ProviderAudioSink), Lookalike())
    before = set(tmp_path.iterdir())

    for consumer in consumers:
        for forged in forged_sinks:
            with pytest.raises(AnonymousAudioStageError):
                consumer(forged)

    assert field_reads == []
    assert provider_calls == []
    assert set(tmp_path.iterdir()) == before


def test_edge_adapter_receives_sink_not_destination(monkeypatch, sink_factory):
    sink = sink_factory()
    seen: dict[str, object] = {}

    class Communicate:
        def __init__(self, text: str, **kwargs: object) -> None:
            seen["request"] = (text, kwargs)

        async def stream(self):
            seen["path"] = sink.path
            yield {"type": "audio", "data": b"abc"}

    class EdgeModule:
        pass

    EdgeModule.Communicate = Communicate

    monkeypatch.setattr("tools.tts_tool._import_edge_tts", lambda: EdgeModule)
    ack = asyncio.run(
        _generate_edge_tts_to_sink(
            REQUEST.text,
            sink.path,
            {},
            output_format=sink.output_format,
            maximum_bytes=sink.maximum_bytes,
            voice=REQUEST.voice,
            speed=REQUEST.speed,
        )
    )

    assert ack == sink.path
    assert seen["path"] == sink.path
    assert "/tmp" not in str(seen)

    from tools.tts_adapters import builtin_adapter

    assert builtin_adapter("edge", {}).generate(REQUEST, sink) == sink.path


def test_edge_stops_before_writing_chunk_over_cap(tmp_path: Path, monkeypatch):
    stage = _create_anonymous_audio_stage_for_test("mp3", 5, tmp_path)

    class Communicate:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def stream(self):
            yield {"type": "WordBoundary", "text": "ignored"}
            yield {"type": "audio", "data": b"abc"}
            yield {"type": "audio", "data": b"def"}

    class EdgeModule:
        pass

    EdgeModule.Communicate = Communicate
    monkeypatch.setattr("tools.tts_tool._import_edge_tts", lambda: EdgeModule)
    try:
        with pytest.raises(ValueError, match="tts_anonymous_provider_failed"):
            asyncio.run(
                _generate_edge_tts_to_sink(
                    "hello",
                    stage.sink.path,
                    {},
                    output_format="mp3",
                    maximum_bytes=5,
                )
            )
        fd = int(Path(stage.sink.path).name)
        os.lseek(fd, 0, os.SEEK_SET)
        assert os.read(fd, 32) == b"abc"
    finally:
        stage.scrub_and_close()


def test_edge_rejects_malformed_audio_chunk_with_fixed_error(
    tmp_path: Path, monkeypatch
):
    stage = _create_anonymous_audio_stage_for_test("mp3", 32, tmp_path)

    class Communicate:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def stream(self):
            yield {"type": "audio", "data": "not-bytes"}

    class EdgeModule:
        pass

    EdgeModule.Communicate = Communicate
    monkeypatch.setattr("tools.tts_tool._import_edge_tts", lambda: EdgeModule)
    try:
        with pytest.raises(ValueError, match="^tts_anonymous_provider_failed$"):
            asyncio.run(
                _generate_edge_tts_to_sink(
                    "hello",
                    stage.sink.path,
                    {},
                    output_format="mp3",
                    maximum_bytes=32,
                )
            )
    finally:
        stage.scrub_and_close()


def test_elevenlabs_adapter_writes_sink_with_explicit_format(
    monkeypatch, sink_factory
):
    sink = sink_factory("opus")
    seen: dict[str, object] = {}

    class Speech:
        def convert(self, **kwargs: object):
            seen["convert"] = kwargs
            return [b"one", b"two"]

    class Client:
        def __init__(self, **kwargs: object) -> None:
            self.text_to_speech = Speech()

    monkeypatch.setattr("tools.tts_tool._resolve_provider_key", lambda *a: "key")
    monkeypatch.setattr("tools.tts_tool._import_elevenlabs", lambda: Client)

    ack = _generate_elevenlabs_to_sink(
        REQUEST.text,
        sink.path,
        {},
        output_format="opus",
        maximum_bytes=sink.maximum_bytes,
        voice=REQUEST.voice,
        model=REQUEST.model,
    )

    assert ack == sink.path
    fd = int(Path(sink.path).name)
    os.lseek(fd, 0, os.SEEK_SET)
    assert os.read(fd, 32) == b"onetwo"
    assert seen["convert"]["output_format"] == "opus_48000_64"

    from tools.tts_adapters import builtin_adapter

    assert builtin_adapter("elevenlabs", {}).generate(REQUEST, sink) == sink.path


def test_command_template_uses_fd_path_and_format(sink_factory):
    sink = sink_factory()
    rendered = _render_command_tts_sink_template(
        "writer --input {input_path} --output {output_path} "
        "--format {format} --cap {maximum_bytes}",
        input_path="/private/input.txt",
        sink=sink,
        voice="voice-a",
        model="model-a",
        speed="1.25",
    )
    assert sink.path in rendered
    assert "--format mp3" in rendered
    assert "--cap 4096" in rendered


def test_command_adapter_never_expands_destination(sink_factory):
    sink = sink_factory()
    destination = "/private/caller/final.mp3"
    rendered = _render_command_tts_sink_template(
        "writer {output_path} {format}",
        input_path="/private/input.txt",
        sink=sink,
        voice="",
        model="",
        speed="",
    )
    assert destination not in rendered
    assert sink.path in rendered


def test_command_adapter_is_parallel_and_fails_closed_before_task4(
    monkeypatch, sink_factory
):
    sink = sink_factory()
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
        adapter.generate(REQUEST, sink)
    assert calls == []


def test_descriptor_number_is_derived_only_from_canonical_sink_path():
    assert _descriptor_number_from_sink_path("/dev/fd/41") == 41
    assert _descriptor_number_from_sink_path("/proc/self/fd/41") == 41
    for invalid in ("41", "/tmp/41", "/dev/fd/041", "/dev/fd/-1"):
        with pytest.raises(ValueError, match="invalid anonymous TTS sink"):
            _descriptor_number_from_sink_path(invalid)


def _assert_named_or_sibling_provider_rejected(
    provider: str, sink: ProviderAudioSink
) -> None:
    from tools.tts_adapters import builtin_adapter

    with pytest.raises(AnonymousSinkUnsupported):
        builtin_adapter(provider, {}).generate(REQUEST, sink)


def test_neutts_named_path_rejected(sink_factory):
    _assert_named_or_sibling_provider_rejected("neutts", sink_factory())


def test_piper_sibling_path_rejected(sink_factory):
    _assert_named_or_sibling_provider_rejected("piper", sink_factory())


def test_kittentts_sibling_path_rejected(sink_factory):
    _assert_named_or_sibling_provider_rejected("kittentts", sink_factory())


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
