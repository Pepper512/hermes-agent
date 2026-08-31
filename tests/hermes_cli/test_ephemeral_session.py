"""Security contract for invocation-scoped ephemeral one-shot sessions."""

from __future__ import annotations

import argparse
import importlib
import io
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


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
    from run_agent import AIAgent

    agent = object.__new__(AIAgent)
    agent._persist_disabled = True
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
