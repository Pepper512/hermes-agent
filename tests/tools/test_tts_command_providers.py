"""
Tests for custom command-type TTS providers.

These tests cover the ``tts.providers.<name>`` registry: built-in
precedence, command resolution, placeholder rendering, shell-quote
context handling, timeout / failure cleanup, voice_compatible opt-in,
and max_text_length lookup.

Nothing here talks to a real TTS engine. The shell command itself is
portable: we write bytes to ``{output_path}`` using ``python -c`` so
the tests run identically on Linux, macOS, and (with minor quoting
differences) Windows.
"""

import json
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tools.tts_tool import (
    BUILTIN_TTS_PROVIDERS,
    COMMAND_TTS_OUTPUT_FORMATS,
    DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH,
    DEFAULT_COMMAND_TTS_OUTPUT_FORMAT,
    DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS,
    _generate_command_tts,
    _get_command_tts_output_format,
    _get_command_tts_timeout,
    _get_named_provider_config,
    _has_any_command_tts_provider,
    _is_command_provider_config,
    _is_command_tts_voice_compatible,
    _iter_command_providers,
    _render_command_tts_template,
    _render_command_tts_sink_template,
    _resolve_command_provider_config,
    _resolve_max_text_length,
    _run_command_tts,
    _shell_quote_context,
    check_tts_requirements,
    text_to_speech_tool,
)
from tools.tts_adapters import (
    AnonymousSinkUnsupported,
    ProviderLifecycleError,
    ProviderRequest,
    command_adapter,
    generate_and_seal,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _python_copy_command(output_placeholder: str = "{output_path}") -> str:
    """Return a cross-platform shell command that copies {input_path} -> output."""
    interpreter = sys.executable
    return (
        f'"{interpreter}" -c "import shutil, sys; '
        f'shutil.copyfile(sys.argv[1], sys.argv[2])" '
        f'{{input_path}} {output_placeholder}'
    )


def _shell_command(*args: str) -> str:
    """Return a shell command string for subprocess.Popen(shell=True)."""
    if os.name == "nt":
        return subprocess.list2cmdline(list(args))
    return " ".join(shlex.quote(str(arg)) for arg in args)


# ---------------------------------------------------------------------------
# _resolve_command_provider_config / built-in precedence
# ---------------------------------------------------------------------------

class TestResolveCommandProviderConfig:
    def test_builtin_names_are_never_command_providers(self):
        cfg = {
            "providers": {
                "openai": {"type": "command", "command": "echo hi"},
                "edge": {"type": "command", "command": "echo hi"},
            },
        }
        for name in BUILTIN_TTS_PROVIDERS:
            assert _resolve_command_provider_config(name, cfg) is None

    def test_missing_provider_returns_none(self):
        cfg = {"providers": {}}
        assert _resolve_command_provider_config("nope", cfg) is None


    def test_case_insensitive_lookup(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "x"}}}
        assert _resolve_command_provider_config("PIPER-CLI", cfg) is not None

    def test_native_piper_cannot_be_shadowed_by_command_entry(self):
        """Regression guard for PR that added native Piper as a built-in.
        A user's ``tts.providers.piper`` must not override the built-in."""
        cfg = {
            "providers": {
                "piper": {"type": "command", "command": "some-script"},
            },
        }
        assert _resolve_command_provider_config("piper", cfg) is None


class TestCommandTtsEnv:
    def test_command_provider_uses_sanitized_child_env(self, monkeypatch):
        """Salvage of #56332: command TTS must not inherit Hermes secrets."""
        monkeypatch.setenv("AUXILIARY_VISION_API_KEY", "sk-vision")
        monkeypatch.setenv("GATEWAY_RELAY_SECRET", "relay-secret")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
        monkeypatch.setenv("MY_SAFE_TTS_VAR", "keep")

        captured = {}

        class _Stream:
            def read(self, size):
                return ""

        class Proc:
            returncode = 0
            stdout = _Stream()
            stderr = _Stream()

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return Proc()

        monkeypatch.setattr("tools.tts_tool.subprocess.Popen", fake_popen)

        result = _run_command_tts("echo hi", timeout=1)

        assert result.returncode == 0
        env = captured["env"]
        assert "AUXILIARY_VISION_API_KEY" not in env
        assert "GATEWAY_RELAY_SECRET" not in env
        assert "OPENAI_API_KEY" not in env
        assert env["MY_SAFE_TTS_VAR"] == "keep"


class TestGetNamedProviderConfig:
    def test_providers_block_wins(self):
        cfg = {"providers": {"voxcpm": {"command": "new"}},
               "voxcpm": {"command": "legacy"}}
        assert _get_named_provider_config(cfg, "voxcpm") == {"command": "new"}

    def test_legacy_tts_name_block_still_resolves(self):
        cfg = {"voxcpm": {"type": "command", "command": "legacy"}}
        assert _get_named_provider_config(cfg, "voxcpm") == {
            "type": "command", "command": "legacy"
        }

    def test_builtin_names_do_not_leak_through_legacy_path(self):
        """``tts.openai`` must never be mistaken for a command provider."""
        cfg = {"openai": {"command": "oops", "type": "command"}}
        assert _get_named_provider_config(cfg, "openai") == {}


class TestIsCommandProviderConfig:
    def test_empty_dict_is_false(self):
        assert _is_command_provider_config({}) is False


    def test_type_mismatch_is_false(self):
        assert _is_command_provider_config({"type": "native", "command": "x"}) is False


# ---------------------------------------------------------------------------
# _iter_command_providers / _has_any_command_tts_provider
# ---------------------------------------------------------------------------

class TestIterCommandProviders:
    def test_iterates_only_user_command_providers(self):
        cfg = {
            "providers": {
                "openai": {"type": "command", "command": "shouldnt show up"},
                "piper-cli": {"type": "command", "command": "piper-cli"},
                "voxcpm": {"type": "command", "command": "voxcpm"},
                "broken": {"type": "command", "command": ""},
            },
        }
        names = sorted(name for name, _ in _iter_command_providers(cfg))
        assert names == ["piper-cli", "voxcpm"]


    def test_has_any_command_provider_when_none(self):
        assert _has_any_command_tts_provider({"providers": {}}) is False
        assert _has_any_command_tts_provider({}) is False


# ---------------------------------------------------------------------------
# config getters
# ---------------------------------------------------------------------------

class TestConfigGetters:
    def test_timeout_defaults(self):
        assert _get_command_tts_timeout({}) == float(DEFAULT_COMMAND_TTS_TIMEOUT_SECONDS)


    def test_output_format_defaults(self):
        assert _get_command_tts_output_format({}) == DEFAULT_COMMAND_TTS_OUTPUT_FORMAT


    def test_voice_compatible_boolean(self):
        assert _is_command_tts_voice_compatible({"voice_compatible": True}) is True
        assert _is_command_tts_voice_compatible({"voice_compatible": False}) is False


    def test_voice_compatible_default_off(self):
        assert _is_command_tts_voice_compatible({}) is False


# ---------------------------------------------------------------------------
# _resolve_max_text_length for command providers
# ---------------------------------------------------------------------------

class TestMaxTextLengthForCommandProviders:
    def test_default_for_command_provider(self):
        cfg = {"providers": {"piper-cli": {"type": "command", "command": "x"}}}
        assert _resolve_max_text_length("piper-cli", cfg) == DEFAULT_COMMAND_TTS_MAX_TEXT_LENGTH


    def test_override_under_legacy_tts_name_block(self):
        cfg = {"piper-cli": {"type": "command", "command": "x", "max_text_length": 7777}}
        assert _resolve_max_text_length("piper-cli", cfg) == 7777

    def test_non_command_unknown_provider_still_falls_back(self):
        assert _resolve_max_text_length("unknown", {}) > 0


# ---------------------------------------------------------------------------
# _shell_quote_context / template rendering
# ---------------------------------------------------------------------------

class TestShellQuoteContext:
    def test_bare_context(self):
        tpl = 'tts {output_path}'
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) is None


    def test_inside_double_quotes(self):
        tpl = 'tts "{output_path}"'
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) == '"'

    def test_escaped_double_quote_inside_double(self):
        tpl = r'tts "foo \" {output_path}"'
        pos = tpl.index("{output_path}")
        assert _shell_quote_context(tpl, pos) == '"'


class TestRenderCommandTtsTemplate:
    def test_substitutes_all_placeholders(self):
        placeholders = {
            "input_path": "/tmp/in.txt",
            "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out.mp3",
            "format": "mp3",
            "voice": "af_sky",
            "model": "tiny",
            "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "tts --voice {voice} --in {input_path} --out {output_path}",
            placeholders,
        )
        assert "af_sky" in rendered
        assert "/tmp/out.mp3" in rendered


    def test_literal_braces_survive(self):
        placeholders = {
            "input_path": "/tmp/in.txt", "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out.mp3", "format": "mp3",
            "voice": "", "model": "", "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "echo '{{not a placeholder}}' && tts --in {input_path}",
            placeholders,
        )
        assert "{not a placeholder}" in rendered

    def test_injection_is_neutralized(self):
        """Embedded shell metacharacters in a placeholder value must be quoted."""
        placeholders = {
            "input_path": "/tmp/in.txt", "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out; rm -rf /",
            "format": "mp3",
            "voice": "$(whoami)", "model": "", "speed": "1.0",
        }
        rendered = _render_command_tts_template(
            "tts --voice {voice} --out {output_path}",
            placeholders,
        )
        # The injection payload must not appear unquoted in the rendered
        # command. On POSIX shlex.quote wraps the value in single quotes.
        if os.name != "nt":
            assert "'$(whoami)'" in rendered or "'\\''" in rendered
            assert "; rm -rf /" not in rendered.replace(
                "'/tmp/out; rm -rf /'", "",
            )

    def test_preserves_shell_quoting_style(self):
        placeholders = {
            "input_path": "/tmp/in.txt", "text_path": "/tmp/in.txt",
            "output_path": "/tmp/out.mp3", "format": "mp3",
            "voice": "bob's voice", "model": "", "speed": "1.0",
        }
        # When the template wraps the placeholder in double quotes we must
        # escape for that context, not collapse to single-quoted form.
        rendered = _render_command_tts_template(
            'tts --voice "{voice}"',
            placeholders,
        )
        assert '"bob\'s voice"' in rendered


# ---------------------------------------------------------------------------
# _run_command_tts idle/progress timeout behavior
# ---------------------------------------------------------------------------

class TestRunCommandTts:
    def test_sink_command_uses_absolute_total_deadline_despite_continuous_output(self):
        from tools.tts_tool import CommandSinkLifecycleError

        command = _shell_command(
            sys.executable,
            "-u",
            "-c",
            (
                "import sys,time; end=time.monotonic()+.9; "
                "exec(\"while time.monotonic()<end:\\n "
                " sys.stdout.write('x'); sys.stdout.flush(); time.sleep(.02)\")"
            ),
        )
        started = time.monotonic()
        with pytest.raises(CommandSinkLifecycleError) as excinfo:
            _run_command_tts(command, 0.1, input_text="deadline transcript")
        elapsed = time.monotonic() - started
        assert elapsed < 0.8
        assert str(excinfo.value) == "tts_command_sink_lifecycle_failed"

    def test_sink_command_deadline_starts_before_spawn(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import CommandSinkLifecycleError

        real_popen = subprocess.Popen

        def delayed_popen(*args, **kwargs):
            time.sleep(0.08)
            return real_popen(*args, **kwargs)

        monkeypatch.setattr(tts_tool.subprocess, "Popen", delayed_popen)
        command = _shell_command(
            sys.executable,
            "-c",
            "import sys,time; sys.stdin.read(); time.sleep(.08)",
        )
        started = time.monotonic()
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts(command, 0.1, input_text="x")
        assert time.monotonic() - started < 0.17

    def test_sink_command_deadline_includes_reader_drain(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import CommandSinkLifecycleError

        real_popen = subprocess.Popen

        class SlowReader:
            def __init__(self, stream):
                self._stream = stream
                self.buffer = self
                self.encoding = stream.encoding

            def read1(self, size):
                time.sleep(0.12)
                return self._stream.buffer.read1(size)

            def close(self):
                self._stream.close()

            @property
            def closed(self):
                return self._stream.closed

        def delayed_reader_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            proc.stdout = SlowReader(proc.stdout)
            return proc

        monkeypatch.setattr(tts_tool.subprocess, "Popen", delayed_reader_popen)
        command = _shell_command(sys.executable, "-c", "print('drain me')")
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts(command, 0.1, input_text="x")

    def test_sink_command_pid_access_failure_still_reaps_spawned_child(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import CommandSinkLifecycleError

        real_popen = subprocess.Popen
        captured = {}

        class PidFailsOnce:
            def __init__(self, proc):
                object.__setattr__(self, "_proc", proc)
                object.__setattr__(self, "_failed", False)

            def __getattr__(self, name):
                if name == "pid" and not object.__getattribute__(self, "_failed"):
                    object.__setattr__(self, "_failed", True)
                    raise KeyboardInterrupt
                return getattr(object.__getattribute__(self, "_proc"), name)

        def failing_pid_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            captured["proc"] = proc
            return PidFailsOnce(proc)

        monkeypatch.setattr(tts_tool.subprocess, "Popen", failing_pid_popen)
        command = "exec " + _shell_command(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        )
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts(command, 1, input_text="x")
        assert captured["proc"].poll() is not None

    def test_sink_command_lock_failure_never_leaks_spawned_child(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import CommandSinkLifecycleError

        real_popen = subprocess.Popen
        captured = {}

        def recording_popen(*args, **kwargs):
            proc = real_popen(*args, **kwargs)
            captured["proc"] = proc
            return proc

        monkeypatch.setattr(tts_tool.subprocess, "Popen", recording_popen)
        class FailingThreading:
            Thread = threading.Thread

            @staticmethod
            def Lock():
                raise KeyboardInterrupt

        monkeypatch.setattr(tts_tool, "threading", FailingThreading)
        command = "exec " + _shell_command(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        )
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts(command, 1, input_text="x")
        if "proc" in captured:
            assert captured["proc"].poll() is not None

    def test_sink_command_fixed_error_reaps_live_captured_child(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import CommandSinkLifecycleError

        captured = {}

        def fixed_error_after_spawn(command, timeout, env_passthrough, **kwargs):
            del timeout, env_passthrough
            proc = kwargs["popen"](
                command,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            captured["proc"] = proc
            raise CommandSinkLifecycleError("tts_command_sink_lifecycle_failed")

        monkeypatch.setattr(
            tts_tool,
            "_run_command_tts_sink_process_impl",
            fixed_error_after_spawn,
        )
        command = "exec " + _shell_command(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        )
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts(command, 1, input_text="x")
        assert captured["proc"].poll() is not None

    def test_sink_command_fixed_error_closes_exited_child_streams(self, monkeypatch):
        from tools import tts_tool
        from tools.tts_tool import CommandSinkLifecycleError

        captured = {}

        def fixed_error_after_exit(command, timeout, env_passthrough, **kwargs):
            del timeout, env_passthrough
            proc = kwargs["popen"](
                command,
                shell=True,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            proc.wait(timeout=1)
            captured["proc"] = proc
            raise CommandSinkLifecycleError("tts_command_sink_lifecycle_failed")

        monkeypatch.setattr(
            tts_tool,
            "_run_command_tts_sink_process_impl",
            fixed_error_after_exit,
        )
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts("/usr/bin/true", 1, input_text="x")
        proc = captured["proc"]
        assert proc.stdin.closed and proc.stdout.closed and proc.stderr.closed

    def test_sink_command_boundary_discards_secret_command_and_output(self):
        from tools.tts_tool import (
            CommandSinkLifecycleError,
            _run_command_tts_sink_process,
        )

        secret = "sink-secret-f26c"
        command = _shell_command(
            sys.executable,
            "-c",
            f"import sys; print({secret!r}); print({secret!r}, file=sys.stderr); sys.exit(7)",
        )
        with pytest.raises(CommandSinkLifecycleError) as excinfo:
            _run_command_tts_sink_process(
                command,
                1,
                inherited_sink_fd=None,
                input_text="private transcript",
            )
        exc = excinfo.value
        exposed = " ".join((str(exc), repr(exc), repr(exc.args), repr(vars(exc))))
        assert secret not in exposed
        assert "private transcript" not in exposed
        assert not any(hasattr(exc, name) for name in ("cmd", "command", "output", "stdout", "stderr"))

    def test_sink_command_success_returns_no_diagnostic_payload(self):
        from tools.tts_tool import _run_command_tts_sink_process

        command = _shell_command(
            sys.executable,
            "-c",
            "import sys; sys.stdin.read(); print('provider diagnostic')",
        )
        assert (
            _run_command_tts_sink_process(
                command,
                1,
                inherited_sink_fd=None,
                input_text="private transcript",
            )
            is None
        )

    @pytest.mark.macos_only
    def test_command_passes_only_sink_fd(self, monkeypatch):
        captured = {}

        class Stream:
            def read(self, size):
                return ""

            def close(self):
                pass

        class Proc:
            pid = 12345
            returncode = 0
            stdin = Stream()
            stdout = Stream()
            stderr = Stream()

            def wait(self, timeout=None):
                return 0

            def poll(self):
                return self.returncode

        def fake_popen(command, **kwargs):
            captured.update(kwargs)
            return Proc()

        monkeypatch.setattr("tools.tts_tool.subprocess.Popen", fake_popen)
        monkeypatch.setattr("tools.tts_tool._stop_command_tts_process_group", lambda *a, **k: None)
        _run_command_tts("writer", 1, inherited_sink_fd=17, input_text="hello")
        assert captured["pass_fds"] == (17,)
        assert captured["close_fds"] is True

    @pytest.mark.linux_only
    def test_command_passes_only_sink_fd_linux(self, monkeypatch):
        self.test_command_passes_only_sink_fd(monkeypatch)

    def test_command_text_streams_only_through_stdin(self, tmp_path):
        secret = "transcript-only-via-stdin-6af3"
        observed = tmp_path / "stdin-length"
        command = _shell_command(
            sys.executable,
            "-c",
            f"import sys; data=sys.stdin.read(); open({str(observed)!r},'w').write(str(len(data)))",
        )
        result = _run_command_tts(command, 2, input_text=secret)
        assert result is None
        assert observed.read_text() == str(len(secret))
        assert secret not in command

    @pytest.mark.macos_only
    def test_command_input_placeholders_are_fd_zero_path(self, tmp_path):
        from tools.tts_staging import _create_anonymous_audio_stage_for_test

        stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
        try:
            rendered = _render_command_tts_sink_template(
                "writer {input_path} {text_path} {output_path}",
                sink=stage.sink,
                voice="",
                model="",
                speed="1",
            )
            expected = "/dev/fd/0" if sys.platform == "darwin" else "/proc/self/fd/0"
            assert rendered.count(expected) == 2
        finally:
            stage.scrub_and_close()

    @pytest.mark.linux_only
    def test_command_input_placeholders_are_fd_zero_path_linux(self, tmp_path):
        self.test_command_input_placeholders_are_fd_zero_path(tmp_path)

    def test_command_stdout_stderr_capture_is_bounded_while_fully_drained(self):
        command = _shell_command(
            sys.executable,
            "-c",
            "import sys; sys.stdout.write('o'*200000); sys.stderr.write('e'*200000)",
        )
        result = _run_command_tts(command, 3, input_text="")
        assert result is None

    @pytest.mark.macos_only
    def test_command_sink_path_creates_no_named_text_file(self, tmp_path, monkeypatch):
        from tools.tts_staging import _create_anonymous_audio_stage_for_test

        stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
        command = _shell_command(
            sys.executable,
            "-c",
            "import sys; open(sys.argv[1],'wb').write(b'ID3\\x04\\0\\0\\0\\0\\0\\0')",
        ) + " {output_path}"
        monkeypatch.setattr(
            "tools.tts_tool.tempfile.TemporaryDirectory",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("named text file")),
        )
        adapter = command_adapter(
            "writer",
            {"type": "command", "command": command, "timeout": 2},
            {},
        )
        sealed = generate_and_seal(
            adapter,
            ProviderRequest("private transcript", None, None, None, None),
            stage,
        )
        assert sealed is not None
        stage.scrub_and_close()

    @pytest.mark.linux_only
    def test_command_sink_path_creates_no_named_text_file_linux(self, tmp_path, monkeypatch):
        self.test_command_sink_path_creates_no_named_text_file(tmp_path, monkeypatch)

    @pytest.mark.macos_only
    def test_command_argv_env_and_fixed_errors_exclude_text(self, tmp_path, monkeypatch):
        from tools import tts_tool
        from tools.tts_staging import _create_anonymous_audio_stage_for_test

        secret = "private-transcript-51cd"
        captured = {}
        real_popen = subprocess.Popen

        def recording_popen(command, **kwargs):
            captured["command"] = command
            captured["env"] = dict(kwargs["env"])
            return real_popen(command, **kwargs)

        monkeypatch.setattr(tts_tool.subprocess, "Popen", recording_popen)
        stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
        adapter = command_adapter(
            "failed-writer",
            {
                "type": "command",
                "command": _shell_command(sys.executable, "-c", "import sys; sys.exit(9)"),
                "timeout": 2,
            },
            {},
        )
        with pytest.raises(ProviderLifecycleError) as excinfo:
            generate_and_seal(
                adapter,
                ProviderRequest(secret, None, None, None, None),
                stage,
            )
        assert str(excinfo.value) == "tts_provider_lifecycle_failed"
        assert secret not in captured["command"]
        assert all(secret not in str(value) for value in captured["env"].values())

    @pytest.mark.macos_only
    def test_direct_sink_adapter_boundary_is_fixed_and_payload_free(self, tmp_path):
        from tools.tts_staging import _create_anonymous_audio_stage_for_test
        from tools.tts_tool import CommandSinkLifecycleError, _run_command_tts_to_sink

        secret = "direct-sink-secret-0a91"
        stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
        try:
            failing = {
                "type": "command",
                "command": _shell_command(
                    sys.executable,
                    "-c",
                    f"import sys; print({secret!r}); sys.exit(4)",
                ),
                "output_format": "mp3",
            }
            with pytest.raises(CommandSinkLifecycleError) as excinfo:
                _run_command_tts_to_sink(
                    "private transcript",
                    stage.sink,
                    "writer",
                    failing,
                    {},
                    voice=None,
                    model=None,
                    speed=None,
                )
            exposed = " ".join(
                (str(excinfo.value), repr(excinfo.value), repr(excinfo.value.args), repr(vars(excinfo.value)))
            )
            assert secret not in exposed
            assert "private transcript" not in exposed
            assert not any(
                hasattr(excinfo.value, name)
                for name in ("cmd", "command", "output", "stdout", "stderr")
            )
        finally:
            stage.scrub_and_close()

    @pytest.mark.linux_only
    def test_direct_sink_adapter_boundary_is_fixed_and_payload_free_linux(self, tmp_path):
        self.test_direct_sink_adapter_boundary_is_fixed_and_payload_free(tmp_path)

    @pytest.mark.macos_only
    def test_direct_sink_adapter_success_returns_none(self, tmp_path):
        from tools.tts_staging import _create_anonymous_audio_stage_for_test
        from tools.tts_tool import _run_command_tts_to_sink

        stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
        command = _shell_command(
            sys.executable,
            "-c",
            "import sys; sys.stdin.read(); open(sys.argv[1],'wb').write(b'ID3\\x04\\0\\0\\0\\0\\0\\0')",
        ) + " {output_path}"
        try:
            result = _run_command_tts_to_sink(
                "private transcript",
                stage.sink,
                "writer",
                {"type": "command", "command": command, "output_format": "mp3"},
                {},
                voice=None,
                model=None,
                speed=None,
            )
            assert result is None
            assert stage.seal(result) is not None
        finally:
            stage.scrub_and_close()

    @pytest.mark.linux_only
    def test_direct_sink_adapter_success_returns_none_linux(self, tmp_path):
        self.test_direct_sink_adapter_success_returns_none(tmp_path)

    @pytest.mark.linux_only
    def test_command_argv_env_and_fixed_errors_exclude_text_linux(self, tmp_path, monkeypatch):
        self.test_command_argv_env_and_fixed_errors_exclude_text(tmp_path, monkeypatch)

    def test_command_background_claim_rejects_before_launch(self):
        with pytest.raises(AnonymousSinkUnsupported):
            command_adapter(
                "background-writer",
                {"type": "command", "command": "writer", "background": True},
                {},
            )

    def test_reads_process_output_in_large_chunks(self):
        read_sizes: dict[str, list[int]] = {"stdout": [], "stderr": []}

        class FakeStream:
            def __init__(self, name: str, chunks: list[str]):
                self.name = name
                self.chunks = chunks

            def read(self, size: int) -> str:
                read_sizes[self.name].append(size)
                if self.chunks:
                    return self.chunks.pop(0)
                return ""

        class FakeProcess:
            def __init__(self):
                self.pid = 12345
                self.returncode = 0
                self.stdout = FakeStream("stdout", ["done"])
                self.stderr = FakeStream("stderr", ["tick"])

            def wait(self, timeout=None):
                return self.returncode

        with patch("tools.tts_tool.subprocess.Popen", return_value=FakeProcess()):
            result = _run_command_tts("fake tts", timeout=0.25)

        assert result.returncode == 0
        assert result.stdout == "done"
        assert result.stderr == "tick"
        assert read_sizes["stdout"][0] == 65536
        assert read_sizes["stderr"][0] == 65536


    def test_silent_after_progress_still_times_out_with_stderr(self, tmp_path):
        script = tmp_path / "progress_then_hang.py"
        script.write_text(
            "\n".join([
                "import sys, time",
                "print('starting tier 1', file=sys.stderr, flush=True)",
                "time.sleep(1.0)",
            ]),
            encoding="utf-8",
        )

        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            _run_command_tts(
                _shell_command(sys.executable, "-u", str(script)),
                timeout=0.2,
            )

        assert "starting tier 1" in (excinfo.value.stderr or "")
        assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)


# ---------------------------------------------------------------------------
# End-to-end: _generate_command_tts
# ---------------------------------------------------------------------------

class TestGenerateCommandTts:
    def test_writes_output_file(self, tmp_path):
        out = tmp_path / "clip.mp3"
        config = {"command": _python_copy_command()}
        result = _generate_command_tts(
            "hello world",
            str(out),
            "py-copy",
            config,
            {},
        )
        assert result == str(out)
        assert out.exists()
        # The command copied the input text file over to output, so it
        # contains the original UTF-8 text.
        assert out.read_text(encoding="utf-8") == "hello world"


    @pytest.mark.skipif(os.name == "nt", reason="POSIX-only timeout semantics")
    def test_timeout_raises_runtime(self, tmp_path):
        config = {
            "command": f'"{sys.executable}" -c "import time; time.sleep(10)"',
            "timeout": 1,
        }
        with pytest.raises(RuntimeError, match="timed out"):
            _generate_command_tts(
                "hello",
                str(tmp_path / "x.mp3"),
                "slow",
                config,
                {},
            )


# ---------------------------------------------------------------------------
# text_to_speech_tool integration
# ---------------------------------------------------------------------------

class TestTextToSpeechToolWithCommandProvider:
    def test_command_provider_dispatches_end_to_end(self, tmp_path):
        cfg = {
            "tts": {
                "provider": "py-copy",
                "providers": {
                    "py-copy": {
                        "type": "command",
                        "command": _python_copy_command(),
                        "output_format": "mp3",
                    },
                },
            },
        }
        out = tmp_path / "clip.mp3"

        # Patch the config loader used by the tool so we don't touch disk.
        def fake_load():
            return cfg["tts"]

        with patch("tools.tts_tool._load_tts_config", fake_load):
            result = text_to_speech_tool(text="hi", output_path=str(out))
        data = json.loads(result)
        assert data["success"] is True, data
        assert data["provider"] == "py-copy"
        assert data["voice_compatible"] is False
        assert Path(data["file_path"]).exists()

    def test_voice_compatible_opt_in_toggles_flag(self, tmp_path):
        """voice_compatible=true is reflected in the response when the
        file is already .ogg (no ffmpeg needed)."""
        cfg = {
            "provider": "py-copy-ogg",
            "providers": {
                "py-copy-ogg": {
                    "type": "command",
                    "command": _python_copy_command(),
                    "output_format": "ogg",
                    "voice_compatible": True,
                },
            },
        }
        out = tmp_path / "clip.ogg"

        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result = text_to_speech_tool(text="hi", output_path=str(out))
        data = json.loads(result)
        assert data["success"] is True
        assert data["voice_compatible"] is True
        assert data["media_tag"].startswith("[[audio_as_voice]]")

    def test_missing_command_falls_through_to_builtin(self, tmp_path):
        """A provider entry with an empty command is not a command
        provider; the tool should not raise a "command not configured"
        error but fall through to the built-in resolution path."""
        cfg = {
            "provider": "broken",
            "providers": {
                "broken": {"type": "command", "command": "   "},
            },
        }
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            result = text_to_speech_tool(text="hi", output_path=str(tmp_path / "x.mp3"))
        data = json.loads(result)
        # The response should not carry the command-provider error text.
        err = (data.get("error") or "").lower()
        assert "tts.providers.broken.command is not configured" not in err


class TestCheckTtsRequirements:
    def test_configured_command_provider_satisfies_requirement(self):
        cfg = {
            "provider": "x",
            "providers": {"x": {"type": "command", "command": "echo x"}},
        }
        with patch("tools.tts_tool._load_tts_config", return_value=cfg):
            assert check_tts_requirements() is True


class TestCommandTtsEnvPassthrough:
    def test_env_passthrough_restores_named_keys(self, monkeypatch):
        """A provider's env_passthrough allowlist re-adds its own API key
        without unscrubbing everything else."""
        monkeypatch.setenv("MY_TTS_API_KEY", "sk-provider")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

        captured = {}

        class _Stream:
            def read(self, size):
                return ""

        class Proc:
            returncode = 0
            stdout = _Stream()
            stderr = _Stream()

            def wait(self, timeout=None):
                return 0

        def fake_popen(command, **kwargs):
            captured["env"] = kwargs["env"]
            return Proc()

        monkeypatch.setattr("tools.tts_tool.subprocess.Popen", fake_popen)

        result = _run_command_tts(
            "echo hi", timeout=1, env_passthrough=["MY_TTS_API_KEY"]
        )

        assert result.returncode == 0
        env = captured["env"]
        assert env["MY_TTS_API_KEY"] == "sk-provider"
        assert "OPENAI_API_KEY" not in env

    def test_allowlist_parsed_from_provider_config(self):
        from tools.tts_tool import _command_provider_env_passthrough

        assert _command_provider_env_passthrough(
            {"env_passthrough": ["A_KEY", " B_KEY ", ""]}
        ) == ["A_KEY", "B_KEY"]
        assert _command_provider_env_passthrough({}) == []
        assert _command_provider_env_passthrough({"env_passthrough": "A_KEY"}) == []
