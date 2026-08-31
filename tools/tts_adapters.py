"""Pre-cutover provider adapters for anonymous TTS audio sinks.

These internals are intentionally unreachable from the public TTS entries
until the full GENERATE -> SEAL -> DECIDE -> PUBLISH transaction cuts over.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from agent.tts_provider import TTSProvider
from tools.tts_staging import ProviderAudioSink


_UNSUPPORTED = "tts_anonymous_sink_unsupported"
_PROTOCOL_ERROR = "tts_anonymous_sink_protocol_failed"


class AnonymousSinkUnsupported(RuntimeError):
    """The selected provider cannot honor the anonymous-sink contract."""


class ProviderAcknowledgementError(RuntimeError):
    """A provider returned data outside the acknowledgement contract."""


class SinkDisposition(Enum):
    ADAPT = "adapt"
    REJECT_NAMED_PATH = "reject_named_path"


@dataclass(frozen=True)
class ProviderRequest:
    """Immutable provider input with no destination or cleanup authority."""

    text: str
    voice: str | None
    model: str | None
    speed: float | None
    instructions: str | None

    def __post_init__(self) -> None:
        optional_strings = (self.voice, self.model, self.instructions)
        valid_speed = (
            self.speed is None
            or (
                isinstance(self.speed, (int, float))
                and not isinstance(self.speed, bool)
                and math.isfinite(float(self.speed))
                and float(self.speed) > 0
            )
        )
        if (
            not isinstance(self.text, str)
            or any(value is not None and not isinstance(value, str) for value in optional_strings)
            or not valid_speed
        ):
            raise ValueError("invalid TTS provider request")


class TTSProviderAdapter(Protocol):
    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        """Generate only into *sink* and return a non-authoritative ack."""


BUILTIN_SINK_DISPOSITIONS: Mapping[str, SinkDisposition] = MappingProxyType({
    "edge": SinkDisposition.ADAPT,
    "elevenlabs": SinkDisposition.ADAPT,
    "openai": SinkDisposition.ADAPT,
    "deepinfra": SinkDisposition.ADAPT,
    "xai": SinkDisposition.ADAPT,
    "minimax": SinkDisposition.ADAPT,
    "mistral": SinkDisposition.ADAPT,
    "gemini": SinkDisposition.ADAPT,
    "neutts": SinkDisposition.REJECT_NAMED_PATH,
    "piper": SinkDisposition.REJECT_NAMED_PATH,
    "kittentts": SinkDisposition.REJECT_NAMED_PATH,
})


def _validate_acknowledgement(acknowledgement: object, sink: ProviderAudioSink) -> object:
    if acknowledgement is None or (
        isinstance(acknowledgement, str) and acknowledgement == sink.path
    ):
        return acknowledgement
    raise ProviderAcknowledgementError(_PROTOCOL_ERROR)


@dataclass(frozen=True)
class _PluginAdapter:
    provider: TTSProvider

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        if type(self.provider).synthesize_to_sink is TTSProvider.synthesize_to_sink:
            raise AnonymousSinkUnsupported(_UNSUPPORTED)
        acknowledgement = self.provider.synthesize_to_sink(
            request.text,
            sink.path,
            voice=request.voice,
            model=request.model,
            speed=request.speed,
            format=sink.output_format,
            maximum_bytes=sink.maximum_bytes,
            instructions=request.instructions,
        )
        return _validate_acknowledgement(acknowledgement, sink)


def plugin_adapter(provider: TTSProvider) -> TTSProviderAdapter:
    if not isinstance(provider, TTSProvider):
        raise AnonymousSinkUnsupported(_UNSUPPORTED)
    return _PluginAdapter(provider)


@dataclass(frozen=True)
class _RejectedBuiltInAdapter:
    provider_name: str

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        raise AnonymousSinkUnsupported(_UNSUPPORTED)


@dataclass(frozen=True)
class _DeferredBuiltInAdapter:
    provider_name: str

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        # Task 7 supplies the remaining audited built-in writers.  Until then,
        # this parallel boundary is deliberately fail-closed and unreachable.
        raise AnonymousSinkUnsupported(_UNSUPPORTED)


@dataclass(frozen=True)
class _EdgeAdapter:
    tts_config: Mapping[str, Any]

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        from tools.tts_tool import _generate_edge_tts_to_sink

        acknowledgement = asyncio.run(
            _generate_edge_tts_to_sink(
                request.text,
                sink.path,
                dict(self.tts_config),
                output_format=sink.output_format,
                maximum_bytes=sink.maximum_bytes,
                voice=request.voice,
                speed=request.speed,
            )
        )
        return _validate_acknowledgement(acknowledgement, sink)


@dataclass(frozen=True)
class _ElevenLabsAdapter:
    tts_config: Mapping[str, Any]

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        from tools.tts_tool import _generate_elevenlabs_to_sink

        acknowledgement = _generate_elevenlabs_to_sink(
            request.text,
            sink.path,
            dict(self.tts_config),
            output_format=sink.output_format,
            maximum_bytes=sink.maximum_bytes,
            voice=request.voice,
            model=request.model,
        )
        return _validate_acknowledgement(acknowledgement, sink)


@dataclass(frozen=True)
class _CommandAdapter:
    provider_name: str
    config: Mapping[str, Any]
    tts_config: Mapping[str, Any]

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        # Task 4 owns fd inheritance and process-tree lifetime.  Keeping the
        # adapter categorical here prevents accidental pre-cutover execution.
        raise AnonymousSinkUnsupported(_UNSUPPORTED)


def command_adapter(
    provider_name: str,
    config: Mapping[str, Any],
    tts_config: Mapping[str, Any],
) -> TTSProviderAdapter:
    if (
        not isinstance(provider_name, str)
        or not provider_name.strip()
        or not isinstance(config, Mapping)
        or str(config.get("type") or "command").strip().lower() != "command"
        or not isinstance(config.get("command"), str)
        or not str(config.get("command")).strip()
        or not isinstance(tts_config, Mapping)
    ):
        raise AnonymousSinkUnsupported(_UNSUPPORTED)
    return _CommandAdapter(provider_name.strip().lower(), config, tts_config)


def builtin_adapter(provider_name: str, tts_config: Mapping[str, Any]) -> TTSProviderAdapter:
    if not isinstance(provider_name, str):
        raise AnonymousSinkUnsupported(_UNSUPPORTED)
    key = provider_name.strip().lower()
    disposition = BUILTIN_SINK_DISPOSITIONS.get(key)
    if disposition is None:
        raise AnonymousSinkUnsupported(_UNSUPPORTED)
    if disposition is SinkDisposition.REJECT_NAMED_PATH:
        return _RejectedBuiltInAdapter(key)
    if key == "edge":
        return _EdgeAdapter(tts_config)
    if key == "elevenlabs":
        return _ElevenLabsAdapter(tts_config)
    return _DeferredBuiltInAdapter(key)
