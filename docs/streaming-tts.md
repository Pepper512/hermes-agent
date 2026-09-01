# Streaming TTS

Hermes can stream TTS audio as it arrives from the provider, instead of waiting
for the full audio before playing. This is used by voice mode (CLI/TUI live
conversation), the dashboard speak-stream WebSocket, and — via the gateway
`StreamingTTSConsumer` — any platform adapter that opts into streaming audio.
Voice replies start speaking after the first clause instead of after full
generation + synthesis.

## Architecture

The streaming pipeline has four parts:

1. **Producer** — the LLM emits text deltas as it generates a response
2. **Sentence chunker** — `tools.tts_streaming.SentenceChunker` accumulates
   deltas, strips `<think>` blocks (even split across deltas), and flushes
   complete sentences
3. **TTS provider** — a registered `StreamingTTSProvider` turns each sentence
   into raw PCM chunks (int16 mono at the provider's declared `sample_rate`)
4. **Audio sink** — `sounddevice.OutputStream` for local playback
   (`tools.tts_tool.stream_tts_to_speaker`), or a gateway platform adapter's
   `write_streaming_tts` seam (`gateway/streaming_tts_consumer.py`)

Providers with no chunked API may get per-*sentence* playback through the
synchronous `text_to_speech_tool` path, provided that their adapter supports
the anonymous audio sink described below. Edge (the default) supports that
path and is conversational too. NeuTTS, Piper, and KittenTTS do not currently
support it and fail closed before synthesis; Hermes does not fall back to a
named temporary file. All spoken text is cleaned by
`tools.tts_text_normalize.prepare_spoken_text` (one cleaner, all paths).

## How to pick a provider

By default the dispatcher streams with the provider you already configured
(`tts.provider`) when that provider has a chunked API — it never silently
swaps your voice for a different provider just to get streaming.

To override, set `tts.streaming.provider` in your `config.yaml`:

- a provider name (`elevenlabs`, `gemini`, `openai`, `xai`) pins that streamer
- `auto` walks the priority list `elevenlabs → gemini → openai → xai` and uses
  the first one whose credentials resolve — an explicit opt-in to "best
  chunked voice available"

```yaml
tts:
  provider: gemini
  streaming:
    provider: gemini      # or "auto"
  gemini:
    model: gemini-2.5-flash-preview-tts
    voice: Kore
```

## Capability matrix

| Provider    | Transport                             | Chunked PCM | Credentials |
|-------------|---------------------------------------|-------------|-------------|
| elevenlabs  | chunked HTTP (`pcm_24000`)            | yes         | `ELEVENLABS_API_KEY` / `tts.elevenlabs` |
| openai      | chunked HTTP (`with_streaming_response`, `pcm`) | yes | `tts.openai.api_key` → env → managed gateway |
| gemini      | SSE (`streamGenerateContent?alt=sse`) | yes         | `GEMINI_API_KEY` / `GOOGLE_API_KEY` |
| xai         | WebSocket (`wss://api.x.ai/v1/tts`)   | yes         | xAI OAuth or `XAI_API_KEY` |
| edge, mistral, minimax, deepinfra | — | no (anonymous-sink sync fallback) | as usual |
| NeuTTS, Piper, KittenTTS | — | no (unavailable on the fallback path) | as usual |

All credential lookups go through `resolve_provider_secret()`
(config > env/.env > credential pool) — never bare env reads. Streamed bodies
are capped at 16 MiB per sentence, mirroring the sync providers' bounded
upstream-body invariant.

## Synchronous fallback and anonymous staging

The synchronous fallback is a separate, whole-audio contract. It uses one
transaction for all sentence chunks:

> **GENERATE → SEAL → DECIDE → PUBLISH**

Hermes creates a mode-`0600` regular file inside an invocation-private
mode-`0700` root, opens it, and unlinks its name before calling the provider.
The provider receives only the platform descriptor path, an explicit format,
and a byte cap. It never receives the caller's destination or a publication
temporary name. The accepted staging formats are exactly `mp3`, `wav`, `ogg`,
`flac`, `m4a`, `aac`, `amr`, and `opus`; individual providers may support a
stricter subset and reject other combinations.

| Synchronous provider boundary | Status |
| --- | --- |
| Edge, ElevenLabs, OpenAI, DeepInfra, xAI, MiniMax, Mistral, Gemini | Reviewed anonymous-sink adapter |
| Command provider | Supported only through the descriptor command contract |
| Plugin provider | Supported only when it implements `synthesize_to_sink` |
| NeuTTS, Piper, KittenTTS | Rejected before dispatch; named-path-only |

A provider must finish all work before returning. Hermes joins adapter-owned
tasks and threads and terminates and reaps command-provider process trees
before it seals the bytes. A command provider receives transcript text only on
stdin and the one sink descriptor through `pass_fds=(sink_fd,)` with
`close_fds=True`; one absolute bounded deadline covers spawn, input, output
drain, process exit, and cleanup, and output activity cannot extend it. Daemon
and background modes are not compatible. A plugin's
`synthesize_to_sink` method may acknowledge success with `None` or the exact
issued sink string. Its legacy `synthesize(..., output_path, ...)` method does
not opt it into this path.

On Darwin the descriptor form is `/dev/fd/<n>`; on Linux it is
`/proc/self/fd/<n>`. Hermes fails closed on another platform or when the
required anonymous-sink or atomic-publication primitive is unavailable. There
is no fallback flag, named staging path, persist-then-delete repair, or manual
cleanup step.

An ephemeral invocation returns bounded audio as base64 data-URI fields and
does not return `file_path`, `MEDIA:`, a cleanup token, or any staging path.
The URI media type comes only from the sealed explicit format: `mp3` uses
`audio/mpeg`, `wav` uses `audio/wav`, `ogg` and Ogg-contained `opus` use
`audio/ogg`, `flac` uses `audio/flac`, `m4a` uses `audio/mp4`, `aac` uses
`audio/aac`, and `amr` uses `audio/amr`. A missing or unknown format fails
closed instead of returning mislabeled audio. A durable invocation publishes
only after the final persistence decision. A destination proven absent is
created with the platform's exclusive no-replace
primitive; a destination proven to exist is atomically overwritten under the
caller's existing durable authority. See [Ephemeral sessions](ephemeral-session.md)
for the policy and failure semantics.

## Adding a new streaming provider

1. Subclass `StreamingTTSProvider` in `tools/tts_streaming.py`
2. Set `sample_rate` (and `channels` / `sample_width` if not int16 mono)
3. Implement `available()` (a pure probe — never install anything) and
   `stream(self, text) -> Iterator[bytes]` yielding raw PCM chunks
4. Decorate with `@register("yourname")`
5. Add tests in `tests/tools/test_tts_streaming.py`

The ABC enforces the contract; the registry makes the provider discoverable;
the dispatcher (`stream_tts_to_speaker`) and the gateway consumer handle the
sentence buffer, stop events, and audio sink for free.

## Gateway streaming (platform adapters)

`gateway/streaming_tts_consumer.py` bridges agent deltas to an adapter's
streaming-audio seam. Adapters opt in by overriding, on
`BasePlatformAdapter`:

- `supports_streaming_tts(chat_id, audio_format) -> bool`
- `begin_streaming_tts / write_streaming_tts / finish_streaming_tts /
  abort_streaming_tts`

All default to unsupported/no-op, so existing adapters are untouched. When a
turn's streaming audio completes, the whole-file auto-TTS reply for that turn
is suppressed (no double playback); when streaming fails before any audio was
audible, the gateway uses the anonymous-staged whole-file voice reply.
