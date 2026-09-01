"""Ephemeral policy regressions for implicit core-tool persistence writers."""

from __future__ import annotations

import asyncio
import contextvars
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy


def _run_in_copied_worker(call):
    context = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(context.run, call).result()


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_terminal_overflow_stays_bounded_in_memory(
    tmp_path, monkeypatch, execution
):
    from tools.terminal_tool import terminal_tool
    import tools.tool_output_limits as limits

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(
        limits,
        "_cached_limits",
        {"max_bytes": 2_000, "max_lines": 2_000, "max_line_length": 2_000},
    )

    def invoke():
        return json.loads(
            terminal_tool(
                "python3 -c \"print('private-terminal-head'); "
                "[print('private-terminal-row', 'x'*80) for _ in range(200)]; "
                "print('private-terminal-tail')\"",
                task_id="ephemeral-terminal-overflow",
            )
        )

    async def invoke_async():
        return await asyncio.to_thread(invoke)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if execution == "copied_worker":
            result = _run_in_copied_worker(invoke)
        elif execution == "async_worker":
            result = asyncio.run(invoke_async())
        else:
            result = invoke()

    assert result["exit_code"] == 0
    assert "OUTPUT TRUNCATED" in result["output"]
    assert "full_output_path" not in result
    assert "output_total_chars" not in result
    assert not (home / "cache" / "terminal-output").exists()


def test_ephemeral_execute_code_stdout_overflow_returns_no_spill_path(
    tmp_path, monkeypatch
):
    from tools.code_execution_tool import MAX_STDOUT_BYTES, _truncate_stdout_text

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    content = "private-exec-head\n" + ("x" * MAX_STDOUT_BYTES) + "\nprivate-exec-tail"

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        visible, metadata = _truncate_stdout_text(content)

    assert "private-exec-head" in visible
    assert "private-exec-tail" in visible
    assert metadata["stdout_truncated"] is True
    assert "stdout_spill_path" not in metadata
    assert not (home / "cache" / "exec").exists()


def test_ephemeral_web_cache_and_truncation_create_no_files(tmp_path, monkeypatch):
    from tools import web_result_cache, web_tools

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(web_result_cache, "_web_config", lambda: {})
    content = "private-web-head\n" + ("row\n" * 2_000) + "private-web-tail"

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        web_result_cache.extract_cache_put(
            "https://example.invalid/private",
            "private-successful-extract",
            title="private-title",
        )
        assert (
            web_result_cache.extract_cache_get("https://example.invalid/private")
            is None
        )
        visible, truncated = web_tools._truncate_with_footer(
            content,
            "https://example.invalid/private",
            1_000,
        )

    assert truncated is True
    assert "private-web-head" in visible
    assert "private-web-tail" in visible
    assert "Full text saved to:" not in visible
    assert str(home) not in visible
    assert not (home / "cache" / "web").exists()


def test_ephemeral_browser_snapshot_truncation_returns_no_spill_path(
    tmp_path, monkeypatch
):
    from tools.browser_tool import _truncate_snapshot

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    snapshot = "\n".join(
        f'- item "private-browser-{index}" [ref=e{index}]' for index in range(400)
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        visible = _truncate_snapshot(snapshot, max_chars=500)

    assert "truncated" in visible.lower()
    assert "read_file path=" not in visible
    assert str(home) not in visible
    assert not (home / "cache" / "web").exists()


@pytest.mark.parametrize("execution", ["direct", "copied_worker", "async_worker"])
def test_ephemeral_generic_tool_result_spill_stays_in_memory(
    tmp_path, monkeypatch, execution
):
    from tools.tool_result_storage import PERSISTED_OUTPUT_TAG, maybe_persist_tool_result

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    remote_env = MagicMock()
    content = "private-generic-tool-result-" * 3_000

    def invoke():
        return maybe_persist_tool_result(
            content=content,
            tool_name="private-tool",
            tool_use_id="private-tool-call",
            env=remote_env,
            threshold=1_000,
        )

    async def invoke_async():
        return await asyncio.to_thread(invoke)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if execution == "copied_worker":
            result = _run_in_copied_worker(invoke)
        elif execution == "async_worker":
            result = asyncio.run(invoke_async())
        else:
            result = invoke()

    assert len(result) < len(content)
    assert PERSISTED_OUTPUT_TAG not in result
    assert str(home) not in result
    assert "Full output saved to:" not in result
    remote_env.execute.assert_not_called()
    assert not (home / "cache" / "spillover").exists()


def test_ephemeral_parser_limit_recovery_does_not_save_model_command(
    tmp_path, monkeypatch
):
    from tools.approval import _PARSER_LIMIT_DESCRIPTION, _hardline_block_result

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    command = "python3 -c '" + ("private-command; " * 900) + "'"

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = _hardline_block_result(_PARSER_LIMIT_DESCRIPTION, command)

    assert result["approved"] is False
    assert "write_file" in result["message"]
    assert "saved to" not in result["message"]
    assert str(home) not in result["message"]
    assert not (home / "cache" / "blocked-scripts").exists()


def test_ephemeral_mcp_schema_write_through_creates_no_profile_cache(
    tmp_path, monkeypatch
):
    from tools import mcp_schema_cache

    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        mcp_schema_cache.write_cache_entry(
            "private-mcp-server",
            "private-fingerprint",
            tools=[{"name": "private-tool", "inputSchema": {}}],
        )

    assert not (home / "cache" / "mcp_schema_cache.json").exists()
