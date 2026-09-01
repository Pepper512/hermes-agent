"""Round-4 regressions for invocation-scoped ephemeral writer boundaries."""

from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy


def _invoke_in_context(execution: str, call):
    if execution == "direct":
        return call()
    if execution == "copied_worker":
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, call).result()

    async def invoke_async():
        return await asyncio.to_thread(call)

    return asyncio.run(invoke_async())


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_process_checkpoint_never_creates_profile_file(
    tmp_path, monkeypatch, execution
):
    from tools import process_registry as process_registry_mod

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    checkpoint = home / "processes.json"
    monkeypatch.setattr(process_registry_mod, "CHECKPOINT_PATH", checkpoint)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        registry = process_registry_mod.ProcessRegistry()
        session = process_registry_mod.ProcessSession(
            id="private-process",
            command="printf private-process-command",
            task_id="private-task",
            session_key="private-session",
            pid=12345,
            started_at=time.time(),
        )
        registry._running[session.id] = session
        _invoke_in_context(execution, registry._write_checkpoint)

    assert not checkpoint.exists()


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_verification_result_never_opens_sqlite(
    tmp_path, monkeypatch, execution
):
    from agent.verification_evidence import record_terminal_result

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")

    def record():
        return record_terminal_result(
            command="pytest -k private-verification",
            cwd=tmp_path,
            session_id="private-verification-session",
            exit_code=1,
            output="private-verification-output",
        )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert _invoke_in_context(execution, record) is None

    assert not list(home.glob("verification_evidence.db*"))


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_approval_stage_stays_in_memory(
    tmp_path, monkeypatch, execution
):
    from tools.write_approval import stage_write

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    def stage():
        return stage_write(
            "skills",
            {"action": "create", "content": "private-proposed-skill"},
            summary="private staged skill",
            origin="foreground",
        )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        record = _invoke_in_context(execution, stage)

    assert record["payload"]["content"] == "private-proposed-skill"
    assert not (home / "pending").exists()


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_moa_trace_ignores_enabled_config(
    tmp_path, monkeypatch, execution
):
    from agent.moa_trace import save_moa_turn

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {"moa": {"save_traces": True}},
    )

    def save():
        return save_moa_turn(
            session_id="private-moa-session",
            preset_name="private-preset",
            reference_outputs=[],
            aggregator_label="private-aggregator",
            aggregator_model="private-model",
            aggregator_provider="private-provider",
            aggregator_temperature=0,
            aggregator_input_messages=[
                {"role": "user", "content": "private-moa-prompt"}
            ],
            aggregator_output="private-moa-output",
            aggregator_streamed=False,
        )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert _invoke_in_context(execution, save) is None

    assert not (home / "moa-traces").exists()


def _computer_capture():
    from tools.computer_use.backend import CaptureResult, UIElement

    png_b64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76L"
        "AAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
    )
    return CaptureResult(
        mode="vision",
        width=800,
        height=600,
        png_b64=png_b64,
        image_mime_type="image/png",
        png_bytes_len=len(base64.b64decode(png_b64)),
        elements=[
            UIElement(
                index=index,
                role="Text",
                label=f"private-element-{index}-" + ("x" * 300),
                bounds=(0, index, 20, 20),
                app="private-app",
            )
            for index in range(3)
        ],
        app="private-app",
        window_title="private-window",
    )


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_computer_capture_is_in_memory_and_path_free(
    tmp_path, monkeypatch, execution
):
    from tools.computer_use import tool as computer_tool

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(computer_tool, "_image_dimensions_from_b64", lambda _b64: None)
    monkeypatch.setattr(computer_tool, "_should_route_through_aux_vision", lambda: False)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = _invoke_in_context(
            execution,
            lambda: computer_tool._capture_response(_computer_capture(), max_elements=1),
        )

    rendered = json.dumps(result) if isinstance(result, dict) else result
    assert "private-element-0" in rendered
    assert "screenshot_path" not in rendered
    assert "elements_file" not in rendered
    assert str(home) not in rendered
    assert not (home / "cache").exists()


def _configure_browser_vision(monkeypatch, *, outcome: str):
    from tools import browser_tool
    from tools import vision_tools

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAYAAADED76L"
        "AAAADUlEQVR4nGNgGAUgAAABCAABgukLHQAAAABJRU5ErkJggg=="
    )

    def run_browser(_task_id, command, args, **_kwargs):
        assert command == "screenshot"
        target = args[-1]
        with open(target, "wb") as handle:
            handle.write(png)
        return {"success": True, "data": {"path": target}}

    monkeypatch.setattr(browser_tool, "_is_camofox_mode", lambda: False)
    monkeypatch.setattr(browser_tool, "_is_local_backend", lambda: True)
    monkeypatch.setattr(browser_tool, "_get_browser_engine", lambda: "chrome")
    monkeypatch.setattr(browser_tool, "_run_browser_command", run_browser)
    monkeypatch.setattr(browser_tool, "_get_cloud_provider", lambda: None)

    if outcome == "success":
        monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: True)
        monkeypatch.setattr(
            vision_tools,
            "_resize_image_for_vision",
            lambda *_args, **_kwargs: "data:image/png;base64,private-image",
        )
    else:
        monkeypatch.setattr(vision_tools, "_should_use_native_vision_fast_path", lambda: False)

        def fail_vision(**_kwargs):
            if outcome == "cancel":
                raise asyncio.CancelledError()
            raise RuntimeError("private vision failure /private/browser/path")

        monkeypatch.setattr(browser_tool, "_lazy_call_llm", fail_vision)

    return browser_tool


@pytest.mark.parametrize(
    ("execution", "outcome"),
    [("direct", "success"), ("copied_worker", "error"), ("async_worker", "cancel")],
)
def test_ephemeral_browser_vision_uses_disposable_path_and_returns_no_path(
    tmp_path, monkeypatch, execution, outcome
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    browser_tool = _configure_browser_vision(monkeypatch, outcome=outcome)

    def invoke():
        return browser_tool.browser_vision("private browser question", task_id="private")

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                _invoke_in_context(execution, invoke)
            result = ""
        else:
            result = _invoke_in_context(execution, invoke)

    rendered = json.dumps(result) if isinstance(result, dict) else result
    assert "screenshot_path" not in rendered
    assert str(home) not in rendered
    assert "/private/browser/path" not in rendered
    assert not (home / "cache" / "screenshots").exists()


def test_ephemeral_browser_internal_capture_cannot_bypass_public_guard(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    browser_tool = _configure_browser_vision(monkeypatch, outcome="success")

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = browser_tool._browser_vision_impl(
            "private browser question", task_id="private"
        )

    rendered = json.dumps(result) if isinstance(result, dict) else result
    assert "screenshot_path" not in rendered
    assert str(home) not in rendered
    assert not (home / "cache" / "screenshots").exists()


def test_ephemeral_camofox_vision_uses_response_bytes_without_profile_file(
    tmp_path, monkeypatch
):
    from tools import browser_camofox

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        browser_camofox,
        "_get_session",
        lambda _task_id: {"tab_id": "private-tab", "user_id": "private-user"},
    )
    monkeypatch.setattr(
        browser_camofox,
        "_camofox_private_page_block",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        browser_camofox,
        "_get_raw",
        lambda *_args, **_kwargs: SimpleNamespace(content=b"private-image-bytes"),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="private analysis"))]
    )
    monkeypatch.setattr("agent.auxiliary_client.call_llm", lambda **_kwargs: response)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = browser_camofox.camofox_vision(
            "private camofox question", task_id="private-task"
        )

    parsed = json.loads(result)
    assert parsed["success"] is True
    assert parsed["analysis"] == "private analysis"
    assert "screenshot_path" not in parsed
    assert not (home / "browser_screenshots").exists()


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_skills_snapshot_writer_creates_no_cold_cache(
    tmp_path, monkeypatch, execution
):
    from agent import prompt_builder

    home = tmp_path / ".hermes"
    skills_dir = home / "skills"
    monkeypatch.setenv("HERMES_HOME", str(home))

    def write():
        return prompt_builder._write_skills_snapshot(
            skills_dir,
            {"private/SKILL.md": [1, 2]},
            [{"skill_name": "private-skill", "description": "private-description"}],
            {},
        )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert _invoke_in_context(execution, write) is None

    assert not (home / ".skills_prompt_snapshot.json").exists()


def test_terminal_cleanup_worker_inherits_ephemeral_policy_before_first_import(
    tmp_path, monkeypatch
):
    import tools.terminal_tool as terminal_tool
    from hermes_cli.persistence import persistence_disabled

    observed = []
    finished = threading.Event()

    def worker():
        observed.append(persistence_disabled())
        finished.set()

    monkeypatch.setattr(terminal_tool, "_cleanup_thread_worker", worker)
    monkeypatch.setattr(terminal_tool, "_cleanup_thread", None)
    monkeypatch.setattr(terminal_tool, "_cleanup_running", False)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        terminal_tool._start_cleanup_thread()
        assert finished.wait(2)
        terminal_tool._cleanup_thread.join(timeout=2)

    assert observed == [True]


def test_real_terminal_cleanup_first_import_creates_no_state_db(tmp_path):
    home = tmp_path / ".hermes"
    script = r'''\
import sys
import time
from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy

with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
    import tools.terminal_tool as terminal_tool
    terminal_tool._start_cleanup_thread()
    deadline = time.time() + 5
    while "tools.process_registry" not in sys.modules and time.time() < deadline:
        time.sleep(0.01)
    terminal_tool._stop_cleanup_thread()
'''
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(__import__("pathlib").Path(__file__).resolve().parents[2])

    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert not (home / "state.db").exists()
    assert not list(home.glob("state.db-*"))


def test_real_terminal_dispatch_writes_no_checkpoint_or_verification_db(
    tmp_path, monkeypatch
):
    from tools import process_registry as process_registry_mod
    from tools.terminal_tool import terminal_tool

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        process_registry_mod,
        "CHECKPOINT_PATH",
        home / "processes.json",
    )
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        foreground = json.loads(
            terminal_tool(
                f"{sys.executable} -m pytest --version",
                task_id="private-terminal-foreground",
                session_id="private-terminal-session",
                workdir=str(tmp_path),
            )
        )
        background = json.loads(
            terminal_tool(
                f"{sys.executable} -c \"import time; time.sleep(0.2); print('private-background')\"",
                task_id="private-terminal-background",
                session_id="private-terminal-session",
                workdir=str(tmp_path),
                background=True,
            )
        )
        session_id = background["session_id"]

    # The reader finishes after the caller has rebound to durable. Its copied
    # invocation context must keep final checkpoint cleanup fail-closed.
    deadline = time.time() + 10
    while time.time() < deadline:
        status = process_registry_mod.process_registry.poll(session_id)
        if status["status"] == "exited":
            break
        time.sleep(0.02)

    assert foreground["exit_code"] == 0
    assert "verification_evidence" not in foreground
    assert status["status"] == "exited"
    assert not (home / "processes.json").exists()
    assert not list(home.glob("verification_evidence.db*"))
