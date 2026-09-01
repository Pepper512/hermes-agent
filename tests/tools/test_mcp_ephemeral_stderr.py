"""Configured stdio MCP stderr must remain invocation-local in ephemeral mode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy
from tools import mcp_tool

pytestmark = pytest.mark.skipif(
    not mcp_tool._MCP_AVAILABLE, reason="MCP SDK not installed"
)


@pytest.fixture(autouse=True)
def _reset_shared_stderr_handle():
    old = mcp_tool._mcp_stderr_log_fh
    mcp_tool._mcp_stderr_log_fh = None
    yield
    current = mcp_tool._mcp_stderr_log_fh
    if current is not None and current is not old:
        try:
            current.close()
        except Exception:
            pass
    mcp_tool._mcp_stderr_log_fh = old


def test_ephemeral_direct_mcp_header_and_stderr_create_no_profile_log(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        mcp_tool._write_stderr_log_header("private-mcp-server")
        handle = mcp_tool._get_mcp_stderr_log()
        handle.write("private-mcp-stderr\n")
        handle.flush()
        handle.close()

    assert not (home / "logs" / "mcp-stderr.log").exists()
    assert not (home / "logs").exists()


def test_ephemeral_configured_stdio_closes_invocation_local_stderr(
    tmp_path, monkeypatch
):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    captured_handles = []

    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=SimpleNamespace(tools=[]))

    mock_stdio_cm = MagicMock()
    mock_stdio_cm.__aenter__ = AsyncMock(return_value=(object(), object()))
    mock_stdio_cm.__aexit__ = AsyncMock(return_value=False)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)

    def fake_stdio_client(_params, *, errlog):
        captured_handles.append(errlog)
        errlog.write("private-configured-server-stderr\n")
        errlog.flush()
        return mock_stdio_cm

    async def drive():
        with (
            patch("tools.mcp_tool.StdioServerParameters"),
            patch("tools.mcp_tool.stdio_client", side_effect=fake_stdio_client),
            patch("tools.mcp_tool.ClientSession", return_value=mock_session_cm),
            patch("tools.mcp_tool._snapshot_child_pids", return_value=set()),
        ):
            server = mcp_tool.MCPServerTask("private-configured-server")
            await server.start({"command": "echo", "args": ["hello"]})
            await server.shutdown()

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        asyncio.run(drive())

    assert len(captured_handles) == 1
    assert captured_handles[0].closed is True
    assert not (home / "logs" / "mcp-stderr.log").exists()
    assert not (home / "logs").exists()


def test_durable_mcp_header_remains_in_profile_log(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    with bind_persistence_policy(PersistencePolicy.DURABLE):
        mcp_tool._write_stderr_log_header("durable-mcp-server")
        handle = mcp_tool._get_mcp_stderr_log()
        handle.write("durable-mcp-stderr\n")
        handle.flush()

    log_path = home / "logs" / "mcp-stderr.log"
    assert log_path.exists()
    body = log_path.read_text(encoding="utf-8")
    assert "durable-mcp-server" in body
    assert "durable-mcp-stderr" in body
