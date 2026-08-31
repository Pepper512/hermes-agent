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
from typing import Any, Mapping, Protocol, TYPE_CHECKING

from agent.tts_provider import TTSProvider
from tools.tts_staging import ProviderAudioSink, _validate_provider_audio_sink


_UNSUPPORTED = "tts_anonymous_sink_unsupported"
_PROTOCOL_ERROR = "tts_anonymous_sink_protocol_failed"
_LIFECYCLE_ERROR = "tts_provider_lifecycle_failed"
_SCRUB_ERROR = "tts_anonymous_scrub_failed"

if TYPE_CHECKING:
    from tools.tts_staging import AnonymousAudioStage, SealedAudio


class AnonymousSinkUnsupported(RuntimeError):
    """The selected provider cannot honor the anonymous-sink contract."""


class ProviderAcknowledgementError(RuntimeError):
    """A provider returned data outside the acknowledgement contract."""


class ProviderLifecycleError(RuntimeError):
    """Fixed, provider-output-free lifecycle failure."""


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

    def finish_owned_work(self) -> None:
        """Synchronously finish all adapter-owned work."""

    def stop_owned_work(self) -> None:
        """Idempotently stop all adapter-owned work."""


def generate_and_seal(
    adapter: TTSProviderAdapter,
    request: ProviderRequest,
    stage: "AnonymousAudioStage",
) -> "SealedAudio":
    """Generate, finish provider work, then seal one genuine stage.

    Every failure path makes one destruction attempt.  Provider exceptions are
    deliberately replaced with fixed categories; destruction failure takes
    precedence because non-persistence can no longer be asserted.
    """
    try:
        _validate_provider_audio_sink(stage.sink)
        acknowledgement = adapter.generate(request, stage.sink)
        adapter.finish_owned_work()
        return stage.seal(acknowledgement)
    except BaseException:
        try:
            try:
                adapter.stop_owned_work()
            except BaseException:
                pass
            stage.scrub_and_close()
        except BaseException:
            raise ProviderLifecycleError(_SCRUB_ERROR) from None
        raise ProviderLifecycleError(_LIFECYCLE_ERROR) from None


class _SynchronousLifecycle:
    """Reviewed adapters complete all owned work in ``generate``."""

    def finish_owned_work(self) -> None:
        return None

    def stop_owned_work(self) -> None:
        return None


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


def _validate_acknowledgement(acknowledgement: object, trusted_path: str) -> object:
    if acknowledgement is None or (
        type(acknowledgement) is str and acknowledgement == trusted_path
    ):
        return acknowledgement
    raise ProviderAcknowledgementError(_PROTOCOL_ERROR)


@dataclass(frozen=True)
class _PluginAdapter(_SynchronousLifecycle):
    provider: TTSProvider

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        path, output_format, maximum_bytes = _validate_provider_audio_sink(sink)
        if any(
            getattr(self.provider, marker, False) is True
            for marker in (
                "anonymous_sink_background",
                "anonymous_sink_named_path",
                "returns_before_complete",
            )
        ):
            raise AnonymousSinkUnsupported(_UNSUPPORTED)
        if type(self.provider).synthesize_to_sink is TTSProvider.synthesize_to_sink:
            raise AnonymousSinkUnsupported(_UNSUPPORTED)
        acknowledgement = self.provider.synthesize_to_sink(
            request.text,
            path,
            voice=request.voice,
            model=request.model,
            speed=request.speed,
            format=output_format,
            maximum_bytes=maximum_bytes,
            instructions=request.instructions,
        )
        return _validate_acknowledgement(acknowledgement, path)


def plugin_adapter(provider: TTSProvider) -> TTSProviderAdapter:
    if not isinstance(provider, TTSProvider):
        raise AnonymousSinkUnsupported(_UNSUPPORTED)
    return _PluginAdapter(provider)


@dataclass(frozen=True)
class _RejectedBuiltInAdapter(_SynchronousLifecycle):
    provider_name: str

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        _validate_provider_audio_sink(sink)
        raise AnonymousSinkUnsupported(_UNSUPPORTED)


@dataclass(frozen=True)
class _DeferredBuiltInAdapter(_SynchronousLifecycle):
    provider_name: str

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        _validate_provider_audio_sink(sink)
        # Task 7 supplies the remaining audited built-in writers.  Until then,
        # this parallel boundary is deliberately fail-closed and unreachable.
        raise AnonymousSinkUnsupported(_UNSUPPORTED)


@dataclass(frozen=True)
class _EdgeAdapter(_SynchronousLifecycle):
    tts_config: Mapping[str, Any]

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        from tools.tts_tool import _generate_edge_tts_to_sink

        path, output_format, maximum_bytes = _validate_provider_audio_sink(sink)
        acknowledgement = asyncio.run(
            _generate_edge_tts_to_sink(
                request.text,
                path,
                dict(self.tts_config),
                output_format=output_format,
                maximum_bytes=maximum_bytes,
                voice=request.voice,
                speed=request.speed,
            )
        )
        return _validate_acknowledgement(acknowledgement, path)


@dataclass(frozen=True)
class _ElevenLabsAdapter(_SynchronousLifecycle):
    tts_config: Mapping[str, Any]

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        from tools.tts_tool import _generate_elevenlabs_to_sink

        path, output_format, maximum_bytes = _validate_provider_audio_sink(sink)
        acknowledgement = _generate_elevenlabs_to_sink(
            request.text,
            path,
            dict(self.tts_config),
            output_format=output_format,
            maximum_bytes=maximum_bytes,
            voice=request.voice,
            model=request.model,
        )
        return _validate_acknowledgement(acknowledgement, path)


@dataclass(frozen=True)
class _CommandAdapter(_SynchronousLifecycle):
    provider_name: str
    config: Mapping[str, Any]
    tts_config: Mapping[str, Any]

    def generate(self, request: ProviderRequest, sink: ProviderAudioSink) -> object:
        from tools.tts_tool import _run_command_tts_to_sink

        path, _, _ = _validate_provider_audio_sink(sink)
        try:
            acknowledgement = _run_command_tts_to_sink(
                request.text,
                sink,
                self.provider_name,
                dict(self.config),
                dict(self.tts_config),
                voice=request.voice,
                model=request.model,
                speed=request.speed,
            )
        except Exception:
            raise AnonymousSinkUnsupported(_UNSUPPORTED) from None
        return _validate_acknowledgement(acknowledgement, path)


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
        or config.get("background") is True
        or config.get("daemon") is True
        or not isinstance(tts_config, Mapping)
    ):
        raise AnonymousSinkUnsupported(_UNSUPPORTED)
    return _CommandAdapter(
        provider_name.strip().lower(),
        MappingProxyType(dict(config)),
        MappingProxyType(dict(tts_config)),
    )


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
