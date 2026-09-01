# Ephemeral sessions

Hermes one-shot sessions are durable by default. `--ephemeral-session` selects
a closed, invocation-scoped mode that prevents the prompt, response, usage,
memory, hook transcript, debug output, and session artifacts from being
written to Hermes' local durable sinks.

Use one of these two public forms:

```bash
hermes --ephemeral-session -z "one text prompt"
printf '%s' "one text prompt" \
  | hermes chat --query-file - --oneshot --ephemeral-session
```

For `chat`, exactly one text source (`-q` or `--query-file`) is required. The
flag cannot be combined with resume/continue, `--usage-file`, an image,
interactive CLI/TUI use, or another built-in or plugin command. Long-option
abbreviations are rejected. A bare `--ephemeral-session` is invalid.

## Persistence boundary

The policy is bound before plugin discovery, session database construction,
memory setup, hooks, or the agent. It remains active through delegated work,
error handling, finalization, and cleanup. Under the ephemeral policy Hermes:

- does not create or reopen session SQLite/WAL files, session JSON, request
  dumps, transcript logs, usage files, memory stores, or transcript-bearing
  hook output;
- suppresses lifecycle, command, delegation, diagnostic, debug, and recording
  sinks that could retain invocation content;
- disables Browser Use execution rather than returning a screenshot,
  recording, or browser-created result path; and
- returns fixed, path-free failures instead of raw provider or filesystem
  errors.

This boundary controls Hermes' local persistence. It does not promise physical
media erasure and cannot control retention by an external model or TTS
provider. Reviewed Python code, the Python runtime, and the operating system
remain part of the trusted computing base.

## Text-to-speech behavior

Every public and internal TTS call uses one anonymous-staging transaction:

> **GENERATE → SEAL → DECIDE → PUBLISH**

During GENERATE, Hermes creates a mode-`0700` invocation root and a mode-`0600`
regular file, holds its descriptor, and unlinks the only filesystem name before
provider dispatch. The provider receives only a descriptor path, an explicit
format, the configured byte cap, and ordinary synthesis inputs. It receives no
caller destination, destination parent, publication name, or cleanup
authority.

The accepted staging formats are exactly `mp3`, `wav`, `ogg`, `flac`, `m4a`,
`aac`, `amr`, and `opus`. A provider may support a stricter subset. Unsupported
provider/format combinations fail categorically before named audio is created.

| Provider kind | Anonymous-sink compatibility |
| --- | --- |
| Built-in | Edge, ElevenLabs, OpenAI, DeepInfra, xAI, MiniMax, Mistral, and Gemini have reviewed adapters. |
| Named-path-only built-in | NeuTTS, Piper, and KittenTTS are unavailable until separately reviewed descriptor-safe adapters exist. |
| Plugin | The provider must override `TTSProvider.synthesize_to_sink(text, sink_path, ..., format, maximum_bytes, ...)`. The legacy named `synthesize` method is not accepted. |
| Command | The command must consume text on stdin, write only to the issued output descriptor, finish synchronously, and not daemonize or leave background work. |

Plugins may acknowledge success with `None` or the exact issued sink string;
another return value has no path authority and is rejected. Command providers
run in an owned process group with `close_fds=True`, inherit only the sink via
`pass_fds=(sink_fd,)`, and are terminated and reaped on success, error,
timeout, or cancellation before SEAL. The stdin and captured stdout/stderr
streams are closed or drained, and their reader/writer threads are joined,
before SEAL or final scrub. One absolute bounded deadline covers spawn, input,
output drain, process exit, and cleanup; continuous output cannot extend it.
Captured process output is bounded and is never included in the public result.

At SEAL, Hermes first proves that all provider-owned work has stopped. It then
validates the held descriptor's identity, ownership, exact mode, regular-file
type, zero link count, bounded size, digest, and audio structure. The provider
cannot select the bytes by returning a path.

At DECIDE, one transaction-local observation records whether the invocation
was ever ephemeral. The observation is monotonic: a durable → ephemeral →
durable transition still forbids publication. A late durable → ephemeral
transition, including durable → ephemeral → durable, scrubs the held stages
and returns the fixed path-free `TTS generation failed` result; it does not
become an entry-ephemeral audio delivery. Entry-ephemeral audio becomes a
bounded, in-memory result and the anonymous inode is truncated, synced, and
closed. The result contains base64 data-URI audio fields and provider/chunk
metadata only; it contains no `file_path`, `MEDIA:`, descriptor number,
staging name, cleanup handle, or transaction identifier.

The data URI is labeled from the transaction's sealed explicit format, never
from a caller path, provider acknowledgement, or untrusted sniffed label. The
closed mapping is `mp3` → `audio/mpeg`, `wav` → `audio/wav`, `ogg` and
Ogg-contained `opus` → `audio/ogg`, `flac` → `audio/flac`, `m4a` →
`audio/mp4`, `aac` → `audio/aac`, and `amr` → `audio/amr`. A missing or
unknown format stops with the same path-free generation failure.

## Durable publication

A call that remains durable throughout DECIDE may PUBLISH. Hermes copies the
sealed bytes into a provider-invisible, same-filesystem, mode-`0600` temporary
in the authorized destination parent, verifies the copy, syncs it, and makes a
final persistence check immediately before the atomic publication syscall.

- If the destination was absent when authorized, Darwin uses
  `renameatx_np` with exclusive/no-follow/beneath flags and Linux uses
  `renameat2(RENAME_NOREPLACE)`. A concurrent creator is preserved.
- If the destination was an existing authorized regular file, Hermes uses
  atomic replacement and preserves the prior durable overwrite behavior.
- After publication, Hermes syncs the held destination parent. If that sync
  fails, it reports durability as uncertain and does not try an unsafe
  pathname rollback.

The supported descriptor forms are `/dev/fd/<n>` on Darwin and
`/proc/self/fd/<n>` on Linux. Missing descriptor namespaces, signal-mask
support, required atomic primitives, or exact Darwin flags are hard failures.
Hermes does not fall back to plain rename, named provider staging, copying
directly to the final path, or publish-then-delete.

## Failure, interruption, and cleanup

Provider errors, malformed acknowledgements, timeout, cancellation, invalid
format, size overflow, metadata drift, and policy transitions stop publication
and scrub the exact held inode. Catchable signals are deferred only across the
small synchronous current-thread descriptor-custody and publication
boundaries. No await, provider callback, logger callback, or user/plugin code
runs in those regions; pending signals are restored after ownership is
recoverable or the publication outcome is monotonic.

A process crash closes an anonymous staging inode without leaving an audio
pathname. During an already-authorized durable publication copy, a crash can
leave an owner-only unpublished temporary that may still contain audio; no
scrub claim is made for a process that did not complete cleanup. A handled
substitution or cleanup failure may instead leave only the exact owner-held,
scrubbed residue whose pathname identity can no longer be proved. Hermes does
not automatically delete or reuse an unproved pathname, because doing so could
delete another process's replacement. If exact descriptor scrubbing or syncing
fails, Hermes returns a high-severity categorical failure and makes no
nonpersistence claim.

There is no fallback option and no normal manual cleanup procedure. Do not
restore named staging, delete suspected residues by pattern, or retry an
unsupported provider/platform by weakening the gate. Resolve the adapter or
platform capability and rerun through the same public contract.
