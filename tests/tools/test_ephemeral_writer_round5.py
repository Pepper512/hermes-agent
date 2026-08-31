"""Round-5 regressions for remaining browser, debug, and media writers."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy


def _invoke(execution: str, call):
    if execution == "direct":
        return call()
    if execution == "copied":
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, call).result()

    async def run():
        return await asyncio.to_thread(call)

    return asyncio.run(run())


@pytest.mark.parametrize("execution", ["direct", "copied", "async"])
def test_ephemeral_browser_recording_never_starts_profile_webm(
    tmp_path, monkeypatch, execution
):
    from tools import browser_tool

    home = tmp_path / ".hermes"
    calls = []
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.config.read_raw_config",
        lambda: {"browser": {"record_sessions": True}},
    )

    def run(_task_id, command, args, **_kwargs):
        calls.append((command, args))
        Path(args[-1]).write_bytes(b"private-browser-pixels")
        return {"success": True}

    monkeypatch.setattr(browser_tool, "_run_browser_command", run)
    monkeypatch.setattr(browser_tool, "_recording_sessions", set())

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        _invoke(execution, lambda: browser_tool._maybe_start_recording("private-task"))

    assert calls == []
    assert not (home / "browser_recordings").exists()


def test_ephemeral_browser_recording_stop_never_finalizes_browser_file(monkeypatch):
    from tools import browser_tool

    calls = []
    monkeypatch.setattr(browser_tool, "_recording_sessions", {"private-task"})
    monkeypatch.setattr(
        browser_tool,
        "_run_browser_command",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"success": True},
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        browser_tool._maybe_stop_recording("private-task")

    assert calls == []
    assert "private-task" not in browser_tool._recording_sessions


def test_browser_cleanup_worker_inherits_ephemeral_context(monkeypatch):
    from hermes_cli.persistence import persistence_disabled
    from tools import browser_tool

    observed = []
    finished = threading.Event()

    def worker():
        observed.append(persistence_disabled())
        finished.set()

    monkeypatch.setattr(browser_tool, "_browser_cleanup_thread_worker", worker)
    monkeypatch.setattr(browser_tool, "_cleanup_thread", None)
    monkeypatch.setattr(browser_tool, "_cleanup_running", False)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        browser_tool._start_browser_cleanup_thread()
        assert finished.wait(2)
        browser_tool._cleanup_thread.join(timeout=2)

    assert observed == [True]


def _configure_browser_exec(monkeypatch, home: Path, outcome: str, observed: dict):
    from tools import browser_use_cli

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use_cli, "_resolve_real_profile_cdp", lambda *_a, **_k: None)
    monkeypatch.setattr(browser_use_cli, "_resolve_backend_cdp", lambda *_a, **_k: None)
    monkeypatch.setattr(browser_use_cli, "_read_browser_cfg", lambda: {})
    monkeypatch.setattr(browser_use_cli, "is_legacy_browser_use_cloud_config", lambda _c: False)

    def run(*_args, **kwargs):
        workspace = Path(kwargs["env"]["BH_AGENT_WORKSPACE"])
        observed["workspace"] = workspace
        workspace.mkdir(parents=True, exist_ok=True)
        (workspace / "private-page.json").write_text("private-browser-payload")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("browser-use", 5)
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return SimpleNamespace(
            returncode=0 if outcome == "success" else 1,
            stdout=f"private output {workspace}/private-page.json",
            stderr=f"private stderr {workspace}" if outcome == "error" else "",
        )

    monkeypatch.setattr(browser_use_cli.subprocess, "run", run)
    return browser_use_cli


@pytest.mark.parametrize(
    ("execution", "outcome"),
    [("direct", "success"), ("copied", "error"), ("async", "timeout"), ("direct", "cancel")],
)
def test_ephemeral_browser_exec_rejects_before_workspace_or_process(
    tmp_path, monkeypatch, execution, outcome
):
    home = tmp_path / ".hermes"
    observed = {}
    browser_use_cli = _configure_browser_exec(monkeypatch, home, outcome, observed)

    def call():
        return browser_use_cli.browser_exec("print('private')", task_id="private-task")

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = _invoke(execution, call)

    assert json.loads(rendered) == {
        "error": "browser-use is unavailable in ephemeral mode",
        "success": False,
    }
    assert observed == {}
    assert not (home / "cache" / "browser-use").exists()
    assert str(home) not in rendered
    assert "workspace" not in json.loads(rendered)


def test_ephemeral_browser_workspace_internal_helper_creates_nothing(tmp_path, monkeypatch):
    from tools.browser_use_cli import _workspace_dir

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert _workspace_dir("private-task") is None
    assert not home.exists()


def test_ephemeral_browser_exec_rejects_before_result_rendering(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    observed = {}
    browser_use_cli = _configure_browser_exec(monkeypatch, home, "success", observed)
    monkeypatch.setattr(
        "tools.registry.tool_result",
        lambda _result: (_ for _ in ()).throw(RuntimeError("private renderer path")),
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = browser_use_cli.browser_exec("print('private')", task_id="private-task")

    assert observed == {}
    assert json.loads(rendered) == {
        "error": "browser-use is unavailable in ephemeral mode",
        "success": False,
    }
    assert str(home) not in rendered
    assert "private renderer path" not in rendered


@pytest.mark.parametrize("env_var", ["WEB_TOOLS_DEBUG", "VISION_TOOLS_DEBUG", "IMAGE_TOOLS_DEBUG"])
def test_ephemeral_debug_session_is_disabled_before_log_path(tmp_path, monkeypatch, env_var):
    from tools.debug_helpers import DebugSession

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv(env_var, "true")

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        session = DebugSession("private_tool", env_var=env_var)
        session.log_call("private_call", {"prompt": "private-prompt"})
        session.save()
        info = session.get_session_info()

    assert session.active is False
    assert info == {"enabled": False, "session_id": None, "log_path": None, "total_calls": 0}
    assert not (home / "logs").exists()


def test_ephemeral_debug_save_stays_closed_after_late_policy_change(tmp_path, monkeypatch):
    from tools.debug_helpers import DebugSession

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("WEB_TOOLS_DEBUG", "true")
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        session = DebugSession("web_tools", env_var="WEB_TOOLS_DEBUG")
    session.log_call("private", {"prompt": "private-prompt"})
    session.save()
    assert not (home / "logs").exists()


@pytest.mark.parametrize("execution", ["direct", "copied", "async"])
def test_ephemeral_inline_media_materializers_return_bounded_data_not_paths(
    tmp_path, monkeypatch, execution
):
    from agent.image_gen_provider import save_b64_image
    from agent.video_gen_provider import save_b64_video, save_bytes_video

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    image_b64 = base64.b64encode(b"private-image-bytes").decode()
    video_b64 = base64.b64encode(b"private-video-bytes").decode()

    def materialize():
        return (
            str(save_b64_image(image_b64)),
            str(save_b64_video(video_b64)),
            str(save_bytes_video(b"private-video-raw")),
        )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        image, video, raw_video = _invoke(execution, materialize)

    assert image.startswith("data:image/png;base64,")
    assert video.startswith("data:video/mp4;base64,")
    assert raw_video.startswith("data:video/mp4;base64,")
    assert all(len(value) < 1024 for value in (image, video, raw_video))
    assert not (home / "cache").exists()


def test_ephemeral_url_media_materializers_stream_in_memory(tmp_path, monkeypatch):
    from agent.image_gen_provider import save_url_image
    from agent.video_gen_provider import save_url_video

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    class Response:
        headers = {"Content-Type": "image/png"}
        content = b""

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            del chunk_size
            yield b"private-media-bytes"

    monkeypatch.setattr("requests.get", lambda *_a, **_k: Response())
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        image = str(save_url_image("https://example.test/private.png"))
        Response.headers = {"Content-Type": "video/mp4"}
        video = str(save_url_video("https://example.test/private.mp4"))

    assert image.startswith("data:image/png;base64,")
    assert video.startswith("data:video/mp4;base64,")
    assert not (home / "cache").exists()


def test_ephemeral_media_cache_constructors_fail_before_profile_paths(tmp_path, monkeypatch):
    from agent.image_gen_provider import _images_cache_dir
    from agent.video_gen_provider import _videos_cache_dir

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(RuntimeError, match="unavailable in ephemeral mode"):
            _images_cache_dir()
        with pytest.raises(RuntimeError, match="unavailable in ephemeral mode"):
            _videos_cache_dir()
    assert not home.exists()


def _configure_tts(monkeypatch, home: Path, outcome: str):
    from tools import tts_tool

    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(tts_tool, "DEFAULT_OUTPUT_DIR", str(home / "audio"))
    monkeypatch.setattr(tts_tool, "_load_tts_config", lambda: {"provider": "edge"})
    monkeypatch.setattr(tts_tool, "_get_provider", lambda _cfg: "edge")
    monkeypatch.setattr(tts_tool, "_resolve_max_text_length", lambda *_a: 10000)

    def synth(text, output_path=None, **_kwargs):
        del text
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private-audio-bytes")
        if outcome == "error":
            return json.dumps({"success": False, "error": f"private failure {path}"})
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return json.dumps(
            {"success": True, "file_path": str(path), "provider": "edge", "voice_compatible": False}
        )

    monkeypatch.setattr(tts_tool, "_text_to_speech_single", synth)
    monkeypatch.setattr(
        tts_tool,
        "_build_audio_delivery_files",
        lambda paths, *_a, **_k: (list(paths), False),
    )
    return tts_tool


@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
def test_ephemeral_tts_uses_cleaned_in_memory_audio_envelope(
    tmp_path, monkeypatch, outcome
):
    home = tmp_path / ".hermes"
    tts_tool = _configure_tts(monkeypatch, home, outcome)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                tts_tool.text_to_speech_tool("private speech")
            result = {}
        else:
            result = json.loads(tts_tool.text_to_speech_tool("private speech"))

    if outcome == "success":
        assert result["audio"].startswith("data:audio/mpeg;base64,")
        assert "file_path" not in result
        assert "file_paths" not in result
        assert "MEDIA:" not in json.dumps(result)
    else:
        assert str(home) not in json.dumps(result)
    assert not (home / "audio").exists()


def test_ephemeral_tts_internal_materializer_cannot_write_requested_path(
    tmp_path, monkeypatch
):
    from tools import tts_tool

    requested = tmp_path / ".hermes" / "audio" / "private.mp3"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(tts_tool, "_import_edge_tts", lambda: object())

    async def generate(_text, output_path, _config):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"private-internal-audio")

    monkeypatch.setattr(tts_tool, "_generate_edge_tts", generate)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = json.loads(
            tts_tool._text_to_speech_single(
                "private internal speech",
                str(requested),
                provider="edge",
                tts_config_override={"provider": "edge"},
            )
        )

    assert result["audio"].startswith("data:audio/mpeg;base64,")
    assert "file_path" not in result
    assert not requested.exists()


def test_durable_media_and_browser_workspace_remain_profile_backed(tmp_path, monkeypatch):
    from agent.image_gen_provider import save_b64_image
    from tools.browser_use_cli import _workspace_dir

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = Path(_workspace_dir("durable-task"))
    image = save_b64_image(base64.b64encode(b"durable-image").decode())

    assert workspace == home / "cache" / "browser-use" / "workspace" / "durable-task"
    assert workspace.is_dir()
    assert Path(image).is_file()
