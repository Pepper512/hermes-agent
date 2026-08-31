"""Security contract for invocation-scoped ephemeral one-shot sessions."""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import os
import subprocess
import sys
import threading
import types
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


_SUBPROCESS_SITE_CUSTOMIZE = r'''\
import importlib.abc
import importlib.machinery
import sys
from unittest.mock import MagicMock
import asyncio

def fake_run_conversation(self, prompt):
    if "model-failure" in prompt:
        raise RuntimeError("private-model-failure /private/provider/path")
    if "cancelled" in prompt:
        raise asyncio.CancelledError("private-cancelled /private/cancel/path")
    if "interrupted" in prompt:
        raise KeyboardInterrupt("private-interrupted /private/interrupt/path")
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "subprocess-ok"},
    ]
    self._session_messages = messages
    from hermes_cli.persistence import persistence_disabled
    if persistence_disabled(self):
        # Exercise the real recall/session-search lazy-open, final flush,
        # session JSON, activity, and request-dump fences from inside the
        # subprocess.  oneshot's real finally block then exercises memory
        # shutdown, compression finalization, agent close, and outer cleanup.
        assert self._get_session_db_for_recall() is None
        self._flush_messages_to_session_db(messages)
        self._save_session_log(messages)
        self._persist_session_activity_if_due()
        from agent.agent_runtime_helpers import dump_api_request_debug
        assert dump_api_request_debug(
            self,
            {"messages": messages},
            reason="private-request-dump",
        ) is None
    self._persist_session(messages)
    return {
        "final_response": "subprocess-ok",
        "messages": messages,
        "api_calls": 1,
    }

class RunAgentLoader(importlib.abc.Loader):
    def __init__(self, wrapped):
        self.wrapped = wrapped

    def create_module(self, spec):
        creator = getattr(self.wrapped, "create_module", None)
        return creator(spec) if creator else None

    def exec_module(self, module):
        self.wrapped.exec_module(module)
        module.OpenAI = lambda **_kwargs: MagicMock()
        module.AIAgent.run_conversation = fake_run_conversation

class RunAgentFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "run_agent":
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        finally:
            sys.meta_path.insert(0, self)
        if spec is not None and spec.loader is not None:
            spec.loader = RunAgentLoader(spec.loader)
        return spec

sys.meta_path.insert(0, RunAgentFinder())
'''


def _subprocess_env(home: Path, injection_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(injection_dir), str(Path(__file__).resolve().parents[2])]
    )
    env["HERMES_DISABLE_FAST_CHAT_LAUNCH"] = "1"
    return env


def _run_ephemeral_subprocess(
    home: Path,
    injection_dir: Path,
    argv: list[str],
    *,
    stdin: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *argv],
        input=stdin,
        text=True,
        capture_output=True,
        env=_subprocess_env(home, injection_dir),
        timeout=30,
        check=False,
    )


def _prepare_subprocess_home(home: Path) -> None:
    home.mkdir()
    (home / "config.yaml").write_text(
        "model:\n"
        "  default: local-model\n"
        "  provider: custom\n"
        "  base_url: http://127.0.0.1:1/v1\n"
        "  api_key: test-only\n",
        encoding="utf-8",
    )


def _assert_no_transcript_sinks(home: Path, private_marker: str) -> None:
    forbidden_names = {
        "state.db",
        "state.db-wal",
        "state.db-shm",
        "sessions",
        "memories",
        "logs",
        "hooks",
        "delegation",
    }
    forbidden_suffixes = {".db", ".db-wal", ".db-shm", ".json", ".jsonl", ".log"}
    marker_forms = {
        private_marker.encode(),
        private_marker.upper().encode(),
        private_marker.encode().hex().encode(),
        base64.b64encode(private_marker.encode()),
        urllib.parse.quote(private_marker).encode(),
    }
    for path in home.rglob("*"):
        relative_parts = {part.lower() for part in path.relative_to(home).parts}
        assert not (relative_parts & forbidden_names), path.relative_to(home)
        assert not any(path.name.lower().endswith(suffix) for suffix in forbidden_suffixes), path.relative_to(home)
        if path.is_file():
            payload = path.read_bytes()
            for marker_form in marker_forms:
                assert marker_form not in payload


def _parse(argv: list[str]):
    from hermes_cli._parser import build_top_level_parser

    return build_top_level_parser()[0].parse_args(argv)


def test_parser_binds_one_explicit_policy_for_both_public_forms() -> None:
    from hermes_cli.persistence import PersistencePolicy

    chat = _parse(
        ["chat", "--query-file", "-", "--oneshot", "--ephemeral-session"]
    )
    top_level = _parse(["--ephemeral-session", "-z", "hello"])

    assert chat.persistence_policy is PersistencePolicy.EPHEMERAL
    assert top_level.persistence_policy is PersistencePolicy.EPHEMERAL


def test_parser_defaults_to_durable_policy() -> None:
    from hermes_cli.persistence import PersistencePolicy

    assert _parse(["-z", "hello"]).persistence_policy is PersistencePolicy.DURABLE
    assert (
        _parse(["chat", "--query-file", "-", "--oneshot"]).persistence_policy
        is PersistencePolicy.DURABLE
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--ephemeral-session"],
        ["--ephemeral-session", "--resume", "latest", "-z", "hello"],
        ["--ephemeral-session", "--continue", "-z", "hello"],
        ["--ephemeral-session", "--usage-file", "usage.json", "-z", "hello"],
        ["chat", "--ephemeral-session"],
        ["chat", "--ephemeral-session", "-q", "hello"],
        ["chat", "--ephemeral-session", "-q", "hello", "--oneshot", "--resume", "latest"],
        ["chat", "--ephemeral-session", "-q", "hello", "--oneshot", "--continue"],
        ["chat", "--ephemeral-session", "--image", "image.png", "--oneshot"],
    ],
)
def test_invalid_ephemeral_combinations_fail_validation(argv: list[str]) -> None:
    from hermes_cli.persistence import validate_invocation_policy

    args = _parse(argv)
    with pytest.raises(ValueError, match="ephemeral-session"):
        validate_invocation_policy(args)


def test_ephemeral_help_is_public_on_both_surfaces() -> None:
    parser = __import__("hermes_cli._parser", fromlist=["build_top_level_parser"])
    top, _subparsers, chat = parser.build_top_level_parser()

    assert "--ephemeral-session" in top.format_help()
    assert "--ephemeral-session" in chat.format_help()


def test_chat_ephemeral_dispatches_before_cli_construction(monkeypatch) -> None:
    import hermes_cli.main as main_mod
    from hermes_cli.persistence import PersistencePolicy

    args = _parse(
        ["chat", "--query-file", "-", "--oneshot", "--ephemeral-session"]
    )
    monkeypatch.setattr(sys, "stdin", io.StringIO("private prompt"))
    seen: dict[str, object] = {}

    class Dispatched(Exception):
        pass

    def fake_dispatch(prompt, **kwargs):
        seen["prompt"] = prompt
        seen.update(kwargs)
        raise Dispatched

    monkeypatch.setattr(main_mod, "_run_and_exit_oneshot", fake_dispatch)
    monkeypatch.setitem(
        sys.modules,
        "cli",
        types.SimpleNamespace(main=lambda **_kwargs: pytest.fail("HermesCLI constructed")),
    )

    with pytest.raises(Dispatched):
        main_mod.cmd_chat(args)

    assert seen["prompt"] == "private prompt"
    assert seen["persistence_policy"] is PersistencePolicy.EPHEMERAL


def test_oneshot_ephemeral_never_constructs_session_db_and_binds_agent_policy(
    monkeypatch,
) -> None:
    import hermes_cli.oneshot as oneshot
    import run_agent
    from hermes_cli.persistence import PersistencePolicy

    monkeypatch.setattr(
        oneshot,
        "_create_session_db_for_oneshot",
        lambda: pytest.fail("SessionDB constructed"),
    )
    monkeypatch.setattr(
        sys.modules["hermes_cli.config"],
        "load_config",
        lambda: {"model": {"default": "local-model"}},
    )
    monkeypatch.setattr(
        importlib.import_module("hermes_cli.runtime_provider"),
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": None,
            "base_url": "http://127.0.0.1:1/v1",
            "provider": "custom",
            "requested_provider": "custom",
            "api_mode": "chat_completions",
            "credential_pool": None,
        },
    )
    monkeypatch.setattr(
        importlib.import_module("hermes_cli.tools_config"),
        "_get_platform_tools",
        lambda *_args: set(),
    )
    monkeypatch.setattr(
        importlib.import_module("hermes_cli.mcp_startup"),
        "ensure_mcp_discovery_before_agent_build",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(oneshot, "get_fallback_chain", lambda _cfg: [])

    seen: dict[str, object] = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            self._session_messages = []

        def run_conversation(self, prompt):
            return {"final_response": "ok", "prompt": prompt}

        def shutdown_memory_provider(self, *_args):
            return None

        def close(self):
            return None

    monkeypatch.setattr(run_agent, "AIAgent", FakeAgent)

    response, _result = oneshot._run_agent(
        "private prompt",
        persistence_policy=PersistencePolicy.EPHEMERAL,
    )

    assert response == "ok"
    assert seen["session_db"] is None
    assert seen["persistence_policy"] is PersistencePolicy.EPHEMERAL
    assert seen["skip_memory"] is True
    assert seen["skip_background_review"] is True


def test_ephemeral_policy_suppresses_all_lifecycle_hooks(monkeypatch) -> None:
    from hermes_cli import lifecycle
    from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy

    monkeypatch.setattr(
        "hermes_cli.plugins.invoke_hook",
        lambda *_args, **_kwargs: pytest.fail("plugin hook received transcript"),
    )
    monkeypatch.setattr(
        "hermes_cli.observability.observe_lifecycle",
        lambda *_args, **_kwargs: pytest.fail("observer received transcript"),
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert lifecycle.has_hook("pre_api_request") is False
        assert lifecycle.invoke_hook("pre_api_request", user_message="private") == []
        assert lifecycle.finalize_session(session_id="private") == []


def test_delegated_child_inherits_ephemeral_policy_without_opening_db() -> None:
    import hermes_state
    from hermes_cli.persistence import PersistencePolicy
    from tests.tools.test_delegate import _make_mock_parent
    from tools.delegate_tool import _build_child_agent

    parent = _make_mock_parent(depth=0)
    parent.persistence_policy = PersistencePolicy.EPHEMERAL
    parent._session_db = MagicMock()

    with (
        patch("run_agent.AIAgent") as agent_cls,
        patch.object(hermes_state, "SessionDB") as session_db_cls,
    ):
        agent_cls.return_value = MagicMock()
        _build_child_agent(
            task_index=0,
            goal="private delegated task",
            context=None,
            toolsets=None,
            model="local-model",
            max_iterations=5,
            parent_agent=parent,
            task_count=1,
        )

    session_db_cls.assert_not_called()
    assert agent_cls.call_args.kwargs["session_db"] is None
    assert (
        agent_cls.call_args.kwargs["persistence_policy"]
        is PersistencePolicy.EPHEMERAL
    )


def test_ephemeral_agent_cannot_lazy_open_or_write_request_dump(tmp_path: Path) -> None:
    from agent.agent_runtime_helpers import dump_api_request_debug
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._persist_disabled = True
    agent.logs_dir = tmp_path
    agent.session_id = "private-session"

    assert agent._get_session_db_for_recall() is None
    assert dump_api_request_debug(agent, {"messages": ["private"]}, reason="test") is None
    assert not list(tmp_path.glob("request_dump_*.json"))


def test_ephemeral_policy_cannot_be_rearmed_by_late_sink_rebinding(tmp_path: Path) -> None:
    import run_agent
    from hermes_cli.persistence import PersistencePolicy
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent.persistence_policy = PersistencePolicy.EPHEMERAL
    # The typed policy is the authority.  A stale/hostile compatibility flag
    # must never be able to reopen persistence after construction.
    agent._persist_disabled = False
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path
    agent.session_id = "private-session"
    agent._session_messages = [{"role": "user", "content": "private"}]
    agent._session_db = MagicMock()
    agent._end_session_on_close = True
    agent._owns_session_db = True
    agent._memory_manager = MagicMock()
    agent.context_compressor = MagicMock()
    agent.save_trajectories = True
    agent.model = "local-model"

    with patch.object(
        run_agent,
        "_save_trajectory_to_file",
        side_effect=AssertionError("trajectory sink rearmed"),
    ):
        agent._save_trajectory(agent._session_messages, "private", True)

    agent._save_session_log()
    agent._flush_messages_to_session_db(agent._session_messages)
    agent._persist_session_activity_if_due()
    agent.shutdown_memory_provider(agent._session_messages)
    agent.close()

    assert not list(tmp_path.glob("session_*.json"))
    agent._session_db.end_session.assert_not_called()
    agent._session_db.close.assert_not_called()
    agent._memory_manager.on_session_end.assert_not_called()
    agent._memory_manager.sync_all.assert_not_called()
    agent.context_compressor.on_session_end.assert_not_called()


def test_ambient_ephemeral_policy_cannot_be_rearmed_by_owner_policy() -> None:
    from hermes_cli.persistence import (
        PersistencePolicy,
        bind_persistence_policy,
        persistence_disabled,
    )

    owner = types.SimpleNamespace(
        persistence_policy=PersistencePolicy.DURABLE,
        _persist_disabled=False,
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert persistence_disabled(owner) is True


def test_ephemeral_startup_callbacks_run_under_bound_policy(monkeypatch) -> None:
    import hermes_cli.main as main_mod
    from hermes_cli.persistence import PersistencePolicy, current_persistence_policy

    args = _parse(["--ephemeral-session", "-z", "private"])
    observed: list[PersistencePolicy] = []

    def observe(*_args, **_kwargs):
        observed.append(current_persistence_policy())

    monkeypatch.setattr(
        "hermes_cli.plugins.start_background_plugin_discovery", observe
    )
    monkeypatch.setattr(
        "hermes_cli.mcp_startup.start_background_mcp_discovery", observe
    )
    monkeypatch.setattr("hermes_cli.config.load_config", lambda: observe() or {})
    monkeypatch.setattr("agent.shell_hooks.register_from_config", observe)
    monkeypatch.setattr("agent.outbound_webhooks.register_from_config", observe)

    main_mod._prepare_agent_startup(args)

    assert observed
    assert set(observed) == {PersistencePolicy.EPHEMERAL}
    assert current_persistence_policy() is PersistencePolicy.DURABLE


def test_background_discovery_threads_inherit_ephemeral_policy(monkeypatch) -> None:
    from hermes_cli import mcp_startup, plugins
    from hermes_cli.persistence import (
        PersistencePolicy,
        bind_persistence_policy,
        current_persistence_policy,
    )

    observed: list[PersistencePolicy] = []
    plugin_done = threading.Event()
    mcp_done = threading.Event()

    class Manager:
        _discovered = False

        def discover_and_load(self):
            observed.append(current_persistence_policy())
            plugin_done.set()

    monkeypatch.setattr(plugins, "get_plugin_manager", lambda: Manager())
    monkeypatch.setattr(plugins, "_persist_plugin_toolset_keys", lambda: None)
    monkeypatch.setattr(plugins, "_background_discovery_thread", None)
    monkeypatch.setattr(mcp_startup, "_has_configured_mcp_servers", lambda: True)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_started", False)
    monkeypatch.setattr(mcp_startup, "_mcp_discovery_thread", None)

    def discover_mcp():
        observed.append(current_persistence_policy())
        mcp_done.set()

    monkeypatch.setattr(
        mcp_startup, "_discover_mcp_tools_without_interactive_oauth", discover_mcp
    )
    monkeypatch.setattr("tools.mcp_tool.get_mcp_status", lambda: [])

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        plugins.start_background_plugin_discovery()
        mcp_startup.start_background_mcp_discovery(
            logger=MagicMock(), thread_name="ephemeral-mcp-test"
        )

    assert plugin_done.wait(2)
    assert mcp_done.wait(2)
    assert observed == [PersistencePolicy.EPHEMERAL, PersistencePolicy.EPHEMERAL]


def test_invalid_ephemeral_plugin_command_rejects_before_discovery(monkeypatch) -> None:
    import hermes_cli.main as main_mod

    plugin_cli = MagicMock(return_value=[])
    discovery = MagicMock()
    monkeypatch.setattr(sys, "argv", ["hermes", "--ephemeral-session", "private-plugin"])
    monkeypatch.setenv("HERMES_DISABLE_FAST_CHAT_LAUNCH", "1")
    monkeypatch.setattr("plugins.memory.discover_plugin_cli_commands", plugin_cli)
    monkeypatch.setattr("hermes_cli.plugins.discover_plugins", discovery)

    with pytest.raises(SystemExit) as exc_info:
        main_mod.main()

    assert exc_info.value.code == 2
    plugin_cli.assert_not_called()
    discovery.assert_not_called()


def test_ephemeral_delegation_suppresses_diagnostics_spills_and_stop_hooks(
    tmp_path: Path, monkeypatch
) -> None:
    from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy
    from tools import delegate_tool

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    hook = MagicMock()
    monkeypatch.setattr("hermes_cli.plugins.invoke_hook", hook)
    monkeypatch.setattr(
        delegate_tool,
        "_load_config",
        lambda: {"max_summary_chars": 64},
    )
    memory_manager = MagicMock()
    parent = types.SimpleNamespace(
        persistence_policy=PersistencePolicy.EPHEMERAL,
        _memory_manager=memory_manager,
        session_id="private-parent",
        _current_turn_id="private-turn",
        context_compressor=None,
        session_estimated_cost_usd=0.0,
    )
    child = types.SimpleNamespace(
        persistence_policy=PersistencePolicy.EPHEMERAL,
        _subagent_id="private-child",
        session_id="private-child-session",
    )
    results = [
        {
            "task_index": 0,
            "summary": "private-canary-" * 200,
            "status": "success",
            "_child_role": "worker",
            "_child_cost_usd": 0.0,
        }
    ]

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        diagnostic = delegate_tool._dump_subagent_timeout_diagnostic(
            child=child,
            task_index=0,
            timeout_seconds=0.1,
            duration_seconds=0.2,
            worker_thread=None,
            goal="private goal",
        )
        delegate_tool._finalize_child_results(
            results,
            [{"goal": "private goal"}],
            [(0, {"goal": "private goal"}, child)],
            parent,
        )

    assert diagnostic is None
    assert "summary_full_path" not in results[0]
    hook.assert_not_called()
    memory_manager.on_delegation.assert_not_called()
    assert not (tmp_path / "logs").exists()
    assert not (tmp_path / "cache").exists()


def test_ephemeral_string_system_exit_is_categorical_and_cleanup_stays_bound(
    monkeypatch, capsys
) -> None:
    import hermes_cli.main as main_mod
    from hermes_cli.persistence import PersistencePolicy, current_persistence_policy

    monkeypatch.setattr(
        "hermes_cli.oneshot.run_oneshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            SystemExit("private provider and /private/path")
        ),
    )
    cleanup_policies: list[PersistencePolicy] = []
    monkeypatch.setattr(
        main_mod,
        "_cleanup_oneshot_runtime",
        lambda: cleanup_policies.append(current_persistence_policy()),
    )

    class Exited(Exception):
        pass

    monkeypatch.setattr(
        main_mod,
        "_exit_after_oneshot",
        lambda rc: (_ for _ in ()).throw(Exited(rc)),
    )

    with pytest.raises(Exited) as exc_info:
        main_mod._run_and_exit_oneshot(
            "private prompt", persistence_policy=PersistencePolicy.EPHEMERAL
        )

    assert exc_info.value.args == (1,)
    assert cleanup_policies == [PersistencePolicy.EPHEMERAL]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "hermes -z: agent failed\n"


def test_ephemeral_model_failure_is_categorical(monkeypatch, capsys) -> None:
    import hermes_cli.oneshot as oneshot
    from hermes_cli.persistence import PersistencePolicy

    monkeypatch.setattr(
        oneshot,
        "_validate_explicit_toolsets",
        lambda _toolsets: (None, None),
    )
    monkeypatch.setattr(
        oneshot,
        "_run_agent",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("private canary and raw provider exception")
        ),
    )

    assert (
        oneshot.run_oneshot(
            "private prompt",
            persistence_policy=PersistencePolicy.EPHEMERAL,
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "hermes -z: agent failed\n"


def test_temp_home_ephemeral_construction_creates_no_session_or_memory_sinks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import run_agent
    from hermes_cli.persistence import PersistencePolicy

    ephemeral_home = tmp_path / "ephemeral-home"
    ephemeral_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(ephemeral_home))
    config = {
        "sessions": {"write_json_snapshots": True},
        "memory": {
            "enabled": True,
            "user_profile": True,
            "provider": "untrusted-provider",
        },
        "context": {"engine": "untrusted-engine"},
    }

    from hermes_cli.persistence import bind_persistence_policy

    with (
        bind_persistence_policy(PersistencePolicy.EPHEMERAL),
        patch("hermes_cli.config.load_config_readonly", return_value=config),
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = run_agent.AIAgent(
            api_key="test-key",
            base_url="http://127.0.0.1:1/v1",
            model="local-model",
            provider="custom",
            enabled_toolsets=["memory", "session_search"],
            verbose_logging=True,
            quiet_mode=True,
            skip_context_files=True,
            persistence_policy=PersistencePolicy.EPHEMERAL,
        )

    try:
        assert agent.persistence_policy is PersistencePolicy.EPHEMERAL
        assert agent._persist_disabled is True
        assert agent._session_db is None
        assert agent._memory_store is None
        assert agent._memory_manager is None
        assert agent.verbose_logging is False
        assert agent._session_json_enabled is False
        assert not (ephemeral_home / "state.db").exists()
        assert not (ephemeral_home / "sessions").exists()
        assert not (ephemeral_home / "memories").exists()
        assert not (ephemeral_home / "logs").exists()
    finally:
        agent.close()


@pytest.mark.parametrize(
    ("argv", "stdin", "private_marker"),
    [
        (
            [
                "chat",
                "--query-file",
                "-",
                "--oneshot",
                "--ephemeral-session",
                "--model",
                "local-model",
                "--provider",
                "custom",
            ],
            "private-chat-form",
            "private-chat-form",
        ),
        (
            [
                "--ephemeral-session",
                "--model",
                "local-model",
                "--provider",
                "custom",
                "-z",
                "private-top-form",
            ],
            None,
            "private-top-form",
        ),
    ],
)
def test_both_real_cli_forms_leave_temp_home_sink_free(
    tmp_path: Path,
    argv: list[str],
    stdin: str | None,
    private_marker: str,
) -> None:
    injection = tmp_path / "injection"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(
        _SUBPROCESS_SITE_CUSTOMIZE, encoding="utf-8"
    )
    home = tmp_path / "ephemeral-home"
    _prepare_subprocess_home(home)

    result = _run_ephemeral_subprocess(home, injection, argv, stdin=stdin)

    assert result.returncode == 0
    assert result.stdout == "subprocess-ok\n"
    assert result.stderr == ""
    _assert_no_transcript_sinks(home, private_marker)


@pytest.mark.parametrize(
    ("prompt", "expected_returncode"),
    [
        ("private-model-failure", 1),
        ("private-cancelled", 1),
        ("private-interrupted", 130),
    ],
)
def test_real_ephemeral_subprocess_failures_are_categorical_and_sink_free(
    tmp_path: Path, prompt: str, expected_returncode: int
) -> None:
    injection = tmp_path / "injection"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(
        _SUBPROCESS_SITE_CUSTOMIZE, encoding="utf-8"
    )
    home = tmp_path / "ephemeral-home"
    _prepare_subprocess_home(home)
    argv = [
        "--ephemeral-session",
        "--model",
        "local-model",
        "--provider",
        "custom",
        "-z",
        prompt,
    ]

    result = _run_ephemeral_subprocess(home, injection, argv)

    assert result.returncode == expected_returncode
    assert result.stdout == ""
    if expected_returncode == 1:
        assert result.stderr == "hermes -z: agent failed\n"
    else:
        assert result.stderr == ""
    assert prompt not in result.stderr
    assert "/private/" not in result.stderr
    _assert_no_transcript_sinks(home, prompt)


@pytest.mark.parametrize(
    ("argv", "stdin", "exact_output"),
    [
        (
            [
                "chat",
                "--query-file",
                "-",
                "--oneshot",
                "--model",
                "local-model",
                "--provider",
                "custom",
            ],
            "durable-chat-comparison",
            False,
        ),
        (
            [
                "--model",
                "local-model",
                "--provider",
                "custom",
                "-z",
                "durable-top-comparison",
            ],
            None,
            True,
        ),
    ],
)
def test_both_real_durable_forms_preserve_output_and_create_session_db(
    tmp_path: Path, argv: list[str], stdin: str | None, exact_output: bool
) -> None:
    injection = tmp_path / "injection"
    injection.mkdir()
    (injection / "sitecustomize.py").write_text(
        _SUBPROCESS_SITE_CUSTOMIZE, encoding="utf-8"
    )
    durable_home = tmp_path / "durable-home"
    _prepare_subprocess_home(durable_home)
    result = _run_ephemeral_subprocess(durable_home, injection, argv, stdin=stdin)

    assert result.returncode == 0
    if exact_output:
        assert result.stdout == "subprocess-ok\n"
    else:
        # Durable chat intentionally retains its existing progress/decorated
        # output; only the public ephemeral form is narrowed to one line.
        assert result.stdout.startswith(
            "Query: durable-chat-comparison\nInitializing agent...\n"
        )
        assert result.stdout.endswith("Goodbye! ⚕\n")
    assert result.stderr == ""
    assert (durable_home / "state.db").is_file()
