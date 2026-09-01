"""Provider lifecycle barriers for anonymous TTS staging."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.tts_adapters import (
    ProviderLifecycleError,
    ProviderRequest,
    generate_and_seal,
    plugin_adapter,
)
from agent.tts_provider import TTSProvider
from tools.tts_staging import _create_anonymous_audio_stage_for_test


REQUEST = ProviderRequest("hello", None, None, 1.0, None)


class RecordingStage:
    def __init__(
        self,
        events: list[str],
        backing_stage: object,
        *,
        scrub_error: BaseException | None = None,
    ):
        self.events = events
        self.backing_stage = backing_stage
        self.sink = backing_stage.sink
        self.scrub_calls = 0
        self.scrub_error = scrub_error

    def seal(self, acknowledgement: object) -> str:
        self.events.append("seal")
        return "sealed"

    def scrub_and_close(self) -> None:
        self.scrub_calls += 1
        self.events.append("scrub")
        self.backing_stage.scrub_and_close()
        if self.scrub_error is not None:
            raise self.scrub_error

    def cleanup(self) -> None:
        self.backing_stage.scrub_and_close()


class RecordingAdapter:
    def __init__(
        self,
        events: list[str],
        *,
        generate_error: BaseException | None = None,
        finish_error: BaseException | None = None,
        stop_error: BaseException | None = None,
    ):
        self.events = events
        self.generate_error = generate_error
        self.finish_error = finish_error
        self.stop_error = stop_error

    def generate(self, request: ProviderRequest, sink: object) -> None:
        self.events.append("provider-start")
        if self.generate_error is not None:
            raise self.generate_error

    def finish_owned_work(self) -> None:
        self.events.append("provider-stop")
        if self.finish_error is not None:
            raise self.finish_error

    def stop_owned_work(self) -> None:
        self.events.append("provider-abort")
        if self.stop_error is not None:
            raise self.stop_error


def test_seal_runs_after_adapter_completion(tmp_path: Path):
    events: list[str] = []
    stage = RecordingStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    try:
        result = generate_and_seal(RecordingAdapter(events), REQUEST, stage)
        assert result == "sealed"
        assert events == ["provider-start", "provider-stop", "seal"]
    finally:
        stage.cleanup()


@pytest.mark.parametrize("failure_site", ["generate", "finish"])
def test_failure_stops_then_scrubs_exactly_once(failure_site: str, tmp_path: Path):
    events: list[str] = []
    adapter = RecordingAdapter(
        events,
        generate_error=RuntimeError("provider secret") if failure_site == "generate" else None,
        finish_error=RuntimeError("provider secret") if failure_site == "finish" else None,
    )
    stage = RecordingStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    with pytest.raises(ProviderLifecycleError, match="^tts_provider_lifecycle_failed$"):
        generate_and_seal(adapter, REQUEST, stage)
    assert events[-2:] == ["provider-abort", "scrub"]
    assert stage.scrub_calls == 1


def test_stop_owned_work_failure_cannot_skip_stage_scrub(tmp_path: Path):
    events: list[str] = []
    stage = RecordingStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    adapter = RecordingAdapter(
        events,
        generate_error=RuntimeError("provider secret"),
        stop_error=RuntimeError("stop secret"),
    )
    with pytest.raises(ProviderLifecycleError, match="^tts_provider_lifecycle_failed$"):
        generate_and_seal(adapter, REQUEST, stage)
    assert stage.scrub_calls == 1


def test_scrub_failure_takes_precedence_and_stays_categorical(tmp_path: Path):
    events: list[str] = []
    stage = RecordingStage(
        events,
        _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path),
        scrub_error=OSError("private path"),
    )
    adapter = RecordingAdapter(events, generate_error=RuntimeError("provider secret"))
    with pytest.raises(ProviderLifecycleError, match="^tts_anonymous_scrub_failed$"):
        generate_and_seal(adapter, REQUEST, stage)
    assert stage.scrub_calls == 1


def test_seal_failure_stops_then_scrubs(tmp_path: Path):
    events: list[str] = []

    class FailingSealStage(RecordingStage):
        def seal(self, acknowledgement: object) -> str:
            self.events.append("seal")
            raise OSError("held fd detail")

    stage = FailingSealStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    with pytest.raises(ProviderLifecycleError, match="^tts_provider_lifecycle_failed$"):
        generate_and_seal(RecordingAdapter(events), REQUEST, stage)
    assert events == ["provider-start", "provider-stop", "seal", "provider-abort", "scrub"]
    assert stage.scrub_calls == 1


def test_reader_threads_join_before_seal(tmp_path: Path):
    events: list[str] = []

    class ThreadAdapter(RecordingAdapter):
        def generate(self, request: ProviderRequest, sink: object) -> None:
            self.worker = threading.Thread(
                target=lambda: (time.sleep(0.02), events.append("reader-done"))
            )
            self.worker.start()
            events.append("provider-start")

        def finish_owned_work(self) -> None:
            self.worker.join()
            events.append("provider-stop")

    stage = RecordingStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    try:
        generate_and_seal(ThreadAdapter(events), REQUEST, stage)
        assert events == ["provider-start", "reader-done", "provider-stop", "seal"]
    finally:
        stage.cleanup()


class _LifecyclePlugin(TTSProvider):
    def __init__(self, *, raises: bool = False, background: bool = False):
        self.raises = raises
        self.calls = 0
        self.anonymous_sink_background = background

    @property
    def name(self) -> str:
        return "lifecycle-test"

    def synthesize(self, text: str, output_path: str, **kwargs: object) -> str:
        raise AssertionError("legacy named callback must stay unreachable")

    def synthesize_to_sink(self, text: str, sink_path: str, **kwargs: object) -> str:
        self.calls += 1
        if self.raises:
            raise RuntimeError("provider transcript and path")
        return sink_path


def test_plugin_exception_scrubs_without_seal(tmp_path: Path):
    events: list[str] = []
    stage = RecordingStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    with pytest.raises(ProviderLifecycleError, match="^tts_provider_lifecycle_failed$"):
        generate_and_seal(plugin_adapter(_LifecyclePlugin(raises=True)), REQUEST, stage)
    assert events == ["scrub"]
    assert stage.scrub_calls == 1


def test_provider_background_contract_violation_rejects_before_callback(tmp_path: Path):
    events: list[str] = []
    provider = _LifecyclePlugin(background=True)
    stage = RecordingStage(
        events, _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    )
    with pytest.raises(ProviderLifecycleError, match="^tts_provider_lifecycle_failed$"):
        generate_and_seal(plugin_adapter(provider), REQUEST, stage)
    assert provider.calls == 0
    assert events == ["scrub"]


def _tree_command(
    pid_file: Path,
    sink_fd: int,
    *,
    exit_code: int = 0,
    sleep: float = 0.05,
) -> str:
    held_file = pid_file.with_suffix(".held")
    child = (
        "import os,sys,time; "
        "os.fstat(int(sys.argv[1])); "
        f"open({str(held_file)!r},'w').write('held'); "
        "time.sleep(30)"
    )
    script = "\n".join(
        (
            "import os,subprocess,sys,time",
            f"fd={sink_fd}",
            "os.fstat(fd)",
            f"child={child!r}",
            "c=subprocess.Popen([sys.executable,'-c',child,str(fd)],pass_fds=(fd,))",
            f"deadline=time.monotonic()+2; held={str(held_file)!r}",
            "while not os.path.exists(held) and time.monotonic() < deadline: time.sleep(.005)",
            "if not os.path.exists(held): sys.exit(98)",
            f"open({str(pid_file)!r},'w').write(str(os.getpid())+' '+str(c.pid))",
            f"time.sleep({sleep!r})",
            f"sys.exit({exit_code})",
        )
    )
    return " ".join(shlex.quote(value) for value in (sys.executable, "-c", script))


def _assert_pids_gone(pid_file: Path) -> None:
    parent_pid, child_pid = (int(value) for value in pid_file.read_text().split())
    for pid in (parent_pid, child_pid):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def _assert_command_success_reaps_tree(tmp_path: Path) -> None:
    from tools.tts_tool import _run_command_tts

    pid_file = tmp_path / "pids"
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    sink_fd = int(Path(stage.sink.path).name)
    try:
        result = _run_command_tts(
            _tree_command(pid_file, sink_fd),
            timeout=3,
            inherited_sink_fd=sink_fd,
            input_text="hello",
        )
        assert result is None
        _assert_pids_gone(pid_file)
    finally:
        stage.scrub_and_close()


@pytest.mark.macos_only
def test_command_success_reaps_child_and_grandchild_before_return(tmp_path: Path):
    _assert_command_success_reaps_tree(tmp_path)


@pytest.mark.linux_only
def test_command_success_reaps_child_and_grandchild_before_return_linux(tmp_path: Path):
    _assert_command_success_reaps_tree(tmp_path)


def _assert_command_nonzero_reaps_tree(tmp_path: Path) -> None:
    from tools.tts_tool import _run_command_tts

    pid_file = tmp_path / "pids"
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    sink_fd = int(Path(stage.sink.path).name)
    try:
        from tools.tts_tool import CommandSinkLifecycleError

        with pytest.raises(CommandSinkLifecycleError):
            _run_command_tts(
                _tree_command(pid_file, sink_fd, exit_code=7),
                timeout=3,
                inherited_sink_fd=sink_fd,
                input_text="x",
            )
        _assert_pids_gone(pid_file)
    finally:
        stage.scrub_and_close()


@pytest.mark.macos_only
def test_command_nonzero_reaps_tree_before_error(tmp_path: Path):
    _assert_command_nonzero_reaps_tree(tmp_path)


@pytest.mark.linux_only
def test_command_nonzero_reaps_tree_before_error_linux(tmp_path: Path):
    _assert_command_nonzero_reaps_tree(tmp_path)


def _assert_command_timeout_reaps_tree(tmp_path: Path) -> None:
    from tools.tts_tool import _run_command_tts

    pid_file = tmp_path / "pids"
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    sink_fd = int(Path(stage.sink.path).name)
    try:
        from tools.tts_tool import CommandSinkLifecycleError

        with pytest.raises(CommandSinkLifecycleError):
            _run_command_tts(
                _tree_command(pid_file, sink_fd, sleep=30),
                timeout=0.2,
                inherited_sink_fd=sink_fd,
                input_text="x",
            )
        _assert_pids_gone(pid_file)
    finally:
        stage.scrub_and_close()


@pytest.mark.macos_only
def test_command_timeout_reaps_tree_before_error(tmp_path: Path):
    _assert_command_timeout_reaps_tree(tmp_path)


@pytest.mark.linux_only
def test_command_timeout_reaps_tree_before_error_linux(tmp_path: Path):
    _assert_command_timeout_reaps_tree(tmp_path)


def _assert_command_cancel_reaps_tree(tmp_path: Path, monkeypatch) -> None:
    from tools import tts_tool
    from tools.tts_tool import CommandSinkLifecycleError

    pid_file = tmp_path / "pids"
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    sink_fd = int(Path(stage.sink.path).name)
    real_popen = subprocess.Popen

    class CancelPopen(real_popen):
        def __init__(self, *args, **kwargs):
            real_popen.__init__(self, *args, **kwargs)

        def poll(self):
            deadline = time.monotonic() + 2
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            raise KeyboardInterrupt()

        def wait(self, timeout=None):
            return real_popen.wait(self, timeout=timeout)

        def kill(self):
            return real_popen.kill(self)

    monkeypatch.setattr(tts_tool.subprocess, "Popen", CancelPopen)
    try:
        with pytest.raises(CommandSinkLifecycleError) as excinfo:
            tts_tool._run_command_tts(
                _tree_command(pid_file, sink_fd, sleep=30),
                timeout=3,
                inherited_sink_fd=sink_fd,
                input_text="x",
            )
        assert str(excinfo.value) == "tts_command_sink_lifecycle_failed"
        _assert_pids_gone(pid_file)
    finally:
        stage.scrub_and_close()


@pytest.mark.macos_only
def test_command_cancel_reaps_tree_before_reraising(tmp_path: Path, monkeypatch):
    _assert_command_cancel_reaps_tree(tmp_path, monkeypatch)


@pytest.mark.linux_only
def test_command_cancel_reaps_tree_before_reraising_linux(tmp_path: Path, monkeypatch):
    _assert_command_cancel_reaps_tree(tmp_path, monkeypatch)


def _assert_forced_stop_failure_runs_all_cleanup(tmp_path: Path, monkeypatch) -> None:
    from tools import tts_tool
    from tools.tts_tool import CommandSinkLifecycleError

    pid_file = tmp_path / "pids"
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, tmp_path)
    sink_fd = int(Path(stage.sink.path).name)
    real_popen = subprocess.Popen
    captured = {}
    fallback_calls = []
    before_threads = {thread.ident for thread in threading.enumerate()}

    class RecordingPopen(real_popen):
        def __init__(self, *args, **kwargs):
            real_popen.__init__(self, *args, **kwargs)
            captured["proc"] = self
            deadline = time.monotonic() + 2
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.005)
            captured["pid_marker_ready"] = pid_file.exists()

    real_fallback = tts_tool._fallback_stop_command_tts_process_group

    def recording_fallback(proc, pgid):
        fallback_calls.append(pgid)
        return real_fallback(proc, pgid)

    monkeypatch.setattr(tts_tool.subprocess, "Popen", RecordingPopen)
    monkeypatch.setattr(
        tts_tool,
        "_stop_command_tts_process_group",
        lambda *args: (_ for _ in ()).throw(RuntimeError("forced high-level stop")),
    )
    monkeypatch.setattr(tts_tool, "_fallback_stop_command_tts_process_group", recording_fallback)
    try:
        with pytest.raises(CommandSinkLifecycleError):
            tts_tool._run_command_tts(
                _tree_command(pid_file, sink_fd, sleep=30),
                timeout=0.1,
                inherited_sink_fd=sink_fd,
                input_text="x",
            )
        assert captured["pid_marker_ready"] is True
        _assert_pids_gone(pid_file)
        proc = captured["proc"]
        assert fallback_calls == [proc.pid]
        assert proc.poll() is not None
        assert proc.stdin.closed and proc.stdout.closed and proc.stderr.closed
        assert {thread.ident for thread in threading.enumerate()} == before_threads
    finally:
        stage.scrub_and_close()


@pytest.mark.macos_only
def test_forced_stop_failure_runs_all_cleanup(tmp_path: Path, monkeypatch):
    _assert_forced_stop_failure_runs_all_cleanup(tmp_path, monkeypatch)


@pytest.mark.linux_only
def test_forced_stop_failure_runs_all_cleanup_linux(tmp_path: Path, monkeypatch):
    _assert_forced_stop_failure_runs_all_cleanup(tmp_path, monkeypatch)


@pytest.mark.macos_only
def test_fallback_probe_failure_still_attempts_kill_and_wait(monkeypatch):
    from tools import tts_tool

    events = []

    class Proc:
        def poll(self):
            events.append("poll")
            return None

        def kill(self):
            events.append("kill")

        def wait(self, timeout):
            events.append(("wait", timeout))
            return 0

    monkeypatch.setattr(
        tts_tool,
        "_posix_process_group_exists",
        lambda _pgid: (_ for _ in ()).throw(OSError("forced probe failure")),
    )
    monkeypatch.setattr(
        tts_tool.os,
        "killpg",
        lambda _pgid, sig: events.append(("killpg", sig)),
    )
    with pytest.raises(RuntimeError):
        tts_tool._fallback_stop_command_tts_process_group(Proc(), 1234)
    assert ("killpg", signal.SIGTERM) in events
    assert ("killpg", signal.SIGKILL) in events
    assert "poll" in events
    assert "kill" in events
    assert any(isinstance(event, tuple) and event[0] == "wait" for event in events)


@pytest.mark.linux_only
def test_fallback_probe_failure_still_attempts_kill_and_wait_linux(monkeypatch):
    test_fallback_probe_failure_still_attempts_kill_and_wait(monkeypatch)


@pytest.mark.macos_only
def test_fallback_poll_failure_still_attempts_kill_and_wait(monkeypatch):
    from tools import tts_tool

    events = []

    class Proc:
        def poll(self):
            events.append("poll")
            raise KeyboardInterrupt

        def kill(self):
            events.append("kill")

        def wait(self, timeout):
            events.append(("wait", timeout))
            raise subprocess.TimeoutExpired("fixed", timeout)

    with pytest.raises(RuntimeError):
        tts_tool._fallback_stop_command_tts_process_group(Proc(), None)
    assert "kill" in events
    assert any(isinstance(event, tuple) and event[0] == "wait" for event in events)


@pytest.mark.linux_only
def test_fallback_poll_failure_still_attempts_kill_and_wait_linux(monkeypatch):
    test_fallback_poll_failure_still_attempts_kill_and_wait(monkeypatch)


@pytest.mark.macos_only
def test_fallback_clock_failure_still_attempts_direct_kill_and_wait(monkeypatch):
    from tools import tts_tool

    events = []

    class Proc:
        def poll(self):
            return None

        def kill(self):
            events.append("kill")

        def wait(self, timeout):
            events.append(("wait", timeout))
            return 0

    monkeypatch.setattr(tts_tool.os, "killpg", lambda _pgid, _sig: None)
    monkeypatch.setattr(
        tts_tool.time,
        "monotonic",
        lambda: (_ for _ in ()).throw(KeyboardInterrupt),
    )
    with pytest.raises(RuntimeError):
        tts_tool._fallback_stop_command_tts_process_group(Proc(), 1234)
    assert "kill" in events
    assert any(isinstance(event, tuple) and event[0] == "wait" for event in events)


@pytest.mark.linux_only
def test_fallback_clock_failure_still_attempts_direct_kill_and_wait_linux(monkeypatch):
    test_fallback_clock_failure_still_attempts_direct_kill_and_wait(monkeypatch)


def _cleanup_partial_init_test_process(pid: int | None) -> None:
    if not isinstance(pid, int):
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        os.waitpid(pid, 0)
    except (ChildProcessError, ProcessLookupError):
        pass


def _fixed_error_text(exc: BaseException) -> str:
    return " ".join((str(exc), repr(exc), repr(exc.args), repr(vars(exc))))


def _assert_callable_popen_boundary_rejects_before_invocation(monkeypatch) -> None:
    from tools import tts_tool
    from tools.tts_tool import CommandSinkLifecycleError

    invoked = []

    def forbidden_wrapper(*_args, **_kwargs):
        invoked.append(True)
        raise KeyboardInterrupt("wrapper-secret")

    monkeypatch.setattr(tts_tool.subprocess, "Popen", forbidden_wrapper)
    with pytest.raises(CommandSinkLifecycleError) as excinfo:
        tts_tool._run_command_tts("wrapper-command-secret", 1, input_text="secret")
    assert invoked == []
    assert "secret" not in _fixed_error_text(excinfo.value)


@pytest.mark.macos_only
def test_callable_popen_boundary_rejects_before_invocation(monkeypatch):
    _assert_callable_popen_boundary_rejects_before_invocation(monkeypatch)


@pytest.mark.linux_only
def test_callable_popen_boundary_rejects_before_invocation_linux(monkeypatch):
    _assert_callable_popen_boundary_rejects_before_invocation(monkeypatch)


def _assert_partial_popen_init_failure_reaps_real_child(monkeypatch) -> None:
    from tools import tts_tool
    from tools.tts_tool import CommandSinkLifecycleError

    real_popen = subprocess.Popen
    assert issubclass(real_popen, tts_tool._REVIEWED_POPEN_CLASS)

    class RaisesAfterSpawn(real_popen):
        spawned_pid = None

        def __init__(self, *args, **kwargs):
            real_popen.__init__(self, *args, **kwargs)
            type(self).spawned_pid = self.pid
            raise KeyboardInterrupt("initializer-secret")

    monkeypatch.setattr(tts_tool.subprocess, "Popen", RaisesAfterSpawn)
    try:
        with pytest.raises(CommandSinkLifecycleError) as excinfo:
            tts_tool._run_command_tts("exec /bin/sleep 30", 1, input_text="secret")
        pid = RaisesAfterSpawn.spawned_pid
        assert isinstance(pid, int)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        assert "secret" not in _fixed_error_text(excinfo.value)
    finally:
        _cleanup_partial_init_test_process(RaisesAfterSpawn.spawned_pid)


@pytest.mark.macos_only
def test_partial_popen_init_failure_reaps_real_child(monkeypatch):
    _assert_partial_popen_init_failure_reaps_real_child(monkeypatch)


@pytest.mark.linux_only
def test_partial_popen_init_failure_reaps_real_child_linux(monkeypatch):
    _assert_partial_popen_init_failure_reaps_real_child(monkeypatch)


def _assert_partial_popen_without_methods_uses_pid_fallback(monkeypatch) -> None:
    from tools import tts_tool
    from tools.tts_tool import CommandSinkLifecycleError

    real_popen = subprocess.Popen
    assert issubclass(real_popen, tts_tool._REVIEWED_POPEN_CLASS)

    class BrokenMethodsAfterSpawn(real_popen):
        spawned_pid = None

        def __init__(self, *args, **kwargs):
            real_popen.__init__(self, *args, **kwargs)
            type(self).spawned_pid = self.pid
            raise KeyboardInterrupt("partial-secret")

        def poll(self):
            raise AttributeError("poll unavailable")

        def wait(self, timeout=None):
            raise AttributeError("wait unavailable")

        def kill(self):
            raise AttributeError("kill unavailable")

    monkeypatch.setattr(tts_tool.subprocess, "Popen", BrokenMethodsAfterSpawn)
    try:
        with pytest.raises(CommandSinkLifecycleError) as excinfo:
            tts_tool._run_command_tts("exec /bin/sleep 30", 1, input_text="secret")
        pid = BrokenMethodsAfterSpawn.spawned_pid
        assert isinstance(pid, int)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)
        assert "secret" not in _fixed_error_text(excinfo.value)
    finally:
        _cleanup_partial_init_test_process(BrokenMethodsAfterSpawn.spawned_pid)


@pytest.mark.macos_only
def test_partial_popen_without_methods_uses_pid_fallback(monkeypatch):
    _assert_partial_popen_without_methods_uses_pid_fallback(monkeypatch)


@pytest.mark.linux_only
def test_partial_popen_without_methods_uses_pid_fallback_linux(monkeypatch):
    _assert_partial_popen_without_methods_uses_pid_fallback(monkeypatch)


def _assert_initializer_failure_before_spawn_creates_no_process(monkeypatch) -> None:
    from tools import tts_tool
    from tools.tts_tool import CommandSinkLifecycleError

    real_popen = subprocess.Popen
    assert issubclass(real_popen, tts_tool._REVIEWED_POPEN_CLASS)

    class FailsBeforeSpawn(real_popen):
        initialized = False

        def __init__(self, *_args, **_kwargs):
            type(self).initialized = True
            raise KeyboardInterrupt("before-spawn-secret")

    monkeypatch.setattr(tts_tool.subprocess, "Popen", FailsBeforeSpawn)
    with pytest.raises(CommandSinkLifecycleError) as excinfo:
        tts_tool._run_command_tts("command-secret", 1, input_text="secret")
    assert FailsBeforeSpawn.initialized is True
    assert not hasattr(excinfo.value, "pid")
    assert "secret" not in _fixed_error_text(excinfo.value)


@pytest.mark.macos_only
def test_initializer_failure_before_spawn_creates_no_process(monkeypatch):
    _assert_initializer_failure_before_spawn_creates_no_process(monkeypatch)


@pytest.mark.linux_only
def test_initializer_failure_before_spawn_creates_no_process_linux(monkeypatch):
    _assert_initializer_failure_before_spawn_creates_no_process(monkeypatch)


def _assert_invalid_partial_pid_never_reaches_process_control(monkeypatch) -> None:
    from tools import tts_tool

    events = []

    class Partial:
        pid = -1
        _child_created = True
        returncode = None
        stdin = None
        stdout = None
        stderr = None

    monkeypatch.setattr(
        tts_tool,
        "_stop_command_tts_process_group",
        lambda *_args: events.append("stop"),
    )
    monkeypatch.setattr(
        tts_tool,
        "_fallback_stop_command_tts_process_group",
        lambda *_args: events.append("fallback"),
    )
    monkeypatch.setattr(tts_tool.os, "killpg", lambda *_args: events.append("killpg"))
    monkeypatch.setattr(tts_tool.os, "kill", lambda *_args: events.append("kill"))
    monkeypatch.setattr(tts_tool.os, "waitpid", lambda *_args: events.append("waitpid"))
    tts_tool._emergency_cleanup_spawned_command_tts_process(Partial())
    assert events == []


@pytest.mark.macos_only
def test_invalid_partial_pid_never_reaches_process_control(monkeypatch):
    _assert_invalid_partial_pid_never_reaches_process_control(monkeypatch)


@pytest.mark.linux_only
def test_invalid_partial_pid_never_reaches_process_control_linux(monkeypatch):
    _assert_invalid_partial_pid_never_reaches_process_control(monkeypatch)


def _assert_unconfirmed_partial_pid_never_reaches_process_control(monkeypatch) -> None:
    from tools import tts_tool

    events = []

    class Partial:
        pid = 424242
        _child_created = False
        returncode = None
        stdin = None
        stdout = None
        stderr = None

    monkeypatch.setattr(
        tts_tool,
        "_stop_command_tts_process_group",
        lambda *_args: events.append("stop"),
    )
    monkeypatch.setattr(tts_tool.os, "killpg", lambda *_args: events.append("killpg"))
    monkeypatch.setattr(tts_tool.os, "kill", lambda *_args: events.append("kill"))
    monkeypatch.setattr(tts_tool.os, "waitpid", lambda *_args: events.append("waitpid"))
    tts_tool._emergency_cleanup_spawned_command_tts_process(Partial())
    assert events == []


@pytest.mark.macos_only
def test_unconfirmed_partial_pid_never_reaches_process_control(monkeypatch):
    _assert_unconfirmed_partial_pid_never_reaches_process_control(monkeypatch)


@pytest.mark.linux_only
def test_unconfirmed_partial_pid_never_reaches_process_control_linux(monkeypatch):
    _assert_unconfirmed_partial_pid_never_reaches_process_control(monkeypatch)


def _assert_successful_high_level_cleanup_skips_raw_pid_fallback(monkeypatch) -> None:
    from tools import tts_tool

    events = []

    class Partial:
        pid = 424242
        _child_created = True
        returncode = 0
        stdin = None
        stdout = None
        stderr = None

    monkeypatch.setattr(
        tts_tool,
        "_stop_command_tts_process_group",
        lambda *_args: events.append("stop"),
    )
    monkeypatch.setattr(tts_tool.os, "killpg", lambda *_args: events.append("killpg"))
    monkeypatch.setattr(tts_tool.os, "kill", lambda *_args: events.append("kill"))
    monkeypatch.setattr(tts_tool.os, "waitpid", lambda *_args: events.append("waitpid"))
    tts_tool._emergency_cleanup_spawned_command_tts_process(Partial())
    assert events == ["stop"]


@pytest.mark.macos_only
def test_successful_high_level_cleanup_skips_raw_pid_fallback(monkeypatch):
    _assert_successful_high_level_cleanup_skips_raw_pid_fallback(monkeypatch)


@pytest.mark.linux_only
def test_successful_high_level_cleanup_skips_raw_pid_fallback_linux(monkeypatch):
    _assert_successful_high_level_cleanup_skips_raw_pid_fallback(monkeypatch)
