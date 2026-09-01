"""Extra-cycle regressions for Browser Use ownership and TTS path authority."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy


VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00private-audio"


def _sink_fd(sink) -> int:
    return int(Path(sink.path).name)


def _invoke(execution: str, call):
    if execution == "direct":
        return call()
    if execution == "copied":
        import contextvars

        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, call).result()

    async def run():
        return await asyncio.to_thread(call)

    return asyncio.run(run())


def _browser_inventory(root: Path) -> dict[str, bytes | None]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes()
        for path in sorted(root.rglob("*"))
    }


@pytest.mark.parametrize("execution", ["direct", "copied", "async"])
@pytest.mark.parametrize(
    "outcome", ["success", "nonzero", "timeout", "cancel", "render_failure"]
)
def test_ephemeral_browser_exec_rejects_before_unowned_runtime_or_screenshot(
    tmp_path, monkeypatch, caplog, execution, outcome
):
    from tools import browser_use_cli

    home = tmp_path / ".hermes"
    runtime = tmp_path / "browser-runtime"
    runtime.mkdir()
    unowned = runtime / "preexisting-unowned.png"
    unowned.write_bytes(b"unowned-private-pixels")
    before = _browser_inventory(runtime)
    calls = []

    monkeypatch.setenv("HERMES_HOME", str(home))
    def unexpected_startup(*_args, **_kwargs):
        pytest.fail("ephemeral browser reached startup/discovery")

    for name in (
        "_blocked_url_in_code",
        "_find_cli",
        "_base_subprocess_env",
        "_resolve_real_profile_cdp",
        "_resolve_backend_cdp",
        "_workspace_dir",
        "_read_browser_cfg",
        "is_legacy_browser_use_cloud_config",
    ):
        monkeypatch.setattr(browser_use_cli, name, unexpected_startup)

    def run(*_args, **kwargs):
        calls.append(kwargs)
        session = kwargs["env"].get("BU_NAME", "missing")
        (runtime / f"hermes-bu-owntab-501-{session}-999").write_text("marker")
        (runtime / f"{session}.sock").write_text("socket")
        (runtime / f"{session}.log").write_text("private-daemon-log")
        (runtime / f"{session}.pid").write_text("999")
        external = runtime / f"{session}-external.png"
        external.write_bytes(b"private-external-pixels")
        if outcome == "timeout":
            raise subprocess.TimeoutExpired("browser-use", 5)
        if outcome == "cancel":
            raise asyncio.CancelledError()
        return SimpleNamespace(
            returncode=1 if outcome == "nonzero" else 0,
            stdout=f"screenshot={external} runtime={runtime}",
            stderr=f"daemon path {runtime}/{session}.sock",
        )

    monkeypatch.setattr(browser_use_cli.subprocess, "run", run)
    if outcome == "render_failure":
        monkeypatch.setattr(
            "tools.registry.tool_result",
            lambda _value: (_ for _ in ()).throw(RuntimeError(f"private {runtime}")),
        )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = _invoke(
            execution,
            lambda: browser_use_cli.browser_exec(
                "print(capture_screenshot('/outside/private.png'))",
                session="private-named-session",
                task_id="private-task",
            ),
        )

    parsed = json.loads(rendered)
    assert parsed == {"error": "browser-use is unavailable in ephemeral mode", "success": False}
    assert calls == []
    assert _browser_inventory(runtime) == before
    assert unowned.read_bytes() == b"unowned-private-pixels"
    assert str(runtime) not in rendered
    assert "private-named-session" not in rendered
    assert str(runtime) not in caplog.text
    assert "private-named-session" not in caplog.text
    assert not home.exists()


def _patch_edge_writer(monkeypatch):
    from tools import tts_tool

    async def generate(_text, sink_path, _config, **_kwargs):
        os.write(int(Path(sink_path).name), VALID_MP3)
        return sink_path

    monkeypatch.setattr(tts_tool, "_generate_edge_tts_to_sink", generate)
    return tts_tool


def test_ephemeral_tts_internal_transport_argument_cannot_bypass_policy(
    tmp_path, monkeypatch
):
    tts_tool = _patch_edge_writer(monkeypatch)
    durable = tmp_path / ".hermes" / "audio" / "private.mp3"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(TypeError):
            tts_tool._text_to_speech_single(
                "private speech",
                str(durable),
                provider="edge",
                tts_config_override={"provider": "edge"},
                _ephemeral_transport=True,
            )

    assert not durable.exists()


def test_ephemeral_tts_public_root_argument_cannot_bypass_policy(tmp_path, monkeypatch):
    from tools import tts_tool

    durable_root = tmp_path / ".hermes" / "audio"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(TypeError):
            tts_tool.text_to_speech_tool(
                "private speech",
                provider="edge",
                _ephemeral_dir=durable_root,
            )

    assert not durable_root.exists()


def _invoke_plugin_candidate(monkeypatch, tmp_path: Path, mode: str):
    from tools import tts_tool

    external = tmp_path / "unowned-audio.mp3"
    external.write_bytes(b"unowned-audio")
    observed = {}

    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)
    monkeypatch.setattr(tts_tool, "_plugin_provider_is_voice_compatible", lambda _p: False)

    class SinkAdapter:
        def generate(self, _request, sink):
            observed["requested"] = Path(sink.path)
            observed["fd"] = _sink_fd(sink)
            if mode == "in_root":
                os.write(observed["fd"], VALID_MP3)
                return sink.path
            if mode == "out_of_root":
                return str(external)
            if mode == "symlink":
                unowned = tmp_path / "unowned-symlink.mp3"
                unowned.symlink_to(external)
                observed["unowned"] = unowned
                return str(unowned)
            if mode == "hardlink":
                unowned = tmp_path / "unowned-hardlink.mp3"
                os.link(external, unowned)
                observed["unowned"] = unowned
                return str(unowned)
            if mode == "replacement":
                replacement = tmp_path / "unowned-replacement.mp3"
                replacement.write_bytes(b"unowned-replacement")
                observed["replacement"] = replacement
                return str(replacement)
            raise AssertionError(mode)

        def finish_owned_work(self):
            return None

        def stop_owned_work(self):
            return None

    monkeypatch.setattr(
        tts_tool,
        "_anonymous_adapter_for_provider",
        lambda *_args, **_kwargs: SinkAdapter(),
    )
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        result = json.loads(
            tts_tool._text_to_speech_single(
                "private plugin speech",
                str(tmp_path / ".hermes" / "audio" / "requested.mp3"),
                provider="private-plugin",
                tts_config_override={"provider": "private-plugin"},
            )
        )
    return result, external, observed


@pytest.mark.parametrize("execution", ["direct", "copied", "async"])
def test_ephemeral_tts_accepts_only_proved_in_root_provider_file(
    tmp_path, monkeypatch, caplog, execution
):
    with ThreadPoolExecutor(max_workers=1) as pool:
        # Keep the parametrized execution behavior explicit while allowing the
        # helper to own one monkeypatch context in this test process.
        call = lambda: _invoke_plugin_candidate(monkeypatch, tmp_path, "in_root")
        if execution == "direct":
            result, external, observed = call()
        elif execution == "copied":
            import contextvars

            result, external, observed = pool.submit(contextvars.copy_context().run, call).result()
        else:
            async def run():
                return await asyncio.to_thread(call)

            result, external, observed = asyncio.run(run())

    assert result["success"] is True
    assert result["audio"].startswith("data:audio/mpeg;base64,")
    assert "file_path" not in result
    with pytest.raises(OSError):
        os.fstat(observed["fd"])
    assert str(observed["requested"]) not in caplog.text
    assert external.read_bytes() == b"unowned-audio"


@pytest.mark.parametrize("mode", ["out_of_root", "symlink", "hardlink", "replacement"])
def test_ephemeral_tts_rejects_unproved_provider_path_without_deleting_target(
    tmp_path, monkeypatch, caplog, mode
):
    result, external, observed = _invoke_plugin_candidate(monkeypatch, tmp_path, mode)

    assert result == {"error": "TTS generation failed", "success": False}
    assert external.read_bytes() == b"unowned-audio"
    assert str(tmp_path) not in json.dumps(result)
    assert str(tmp_path) not in caplog.text
    if mode == "replacement":
        assert observed["replacement"].read_bytes() == b"unowned-replacement"
    if mode in {"symlink", "hardlink"}:
        assert observed["unowned"].exists()


def test_durable_browser_and_tts_internal_contracts_remain_available(tmp_path, monkeypatch):
    from tools import browser_use_cli

    tts_tool = _patch_edge_writer(monkeypatch)
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(browser_use_cli, "_find_cli", lambda: ["browser-use"])
    monkeypatch.setattr(browser_use_cli, "_base_subprocess_env", lambda: {})
    monkeypatch.setattr(browser_use_cli, "_resolve_real_profile_cdp", lambda *_a, **_k: None)
    monkeypatch.setattr(browser_use_cli, "_resolve_backend_cdp", lambda *_a, **_k: None)
    monkeypatch.setattr(browser_use_cli, "_read_browser_cfg", lambda: {})
    monkeypatch.setattr(browser_use_cli, "is_legacy_browser_use_cloud_config", lambda _c: False)
    monkeypatch.setattr(
        browser_use_cli.subprocess,
        "run",
        lambda *_a, **_k: SimpleNamespace(returncode=0, stdout="durable", stderr=""),
    )

    browser = json.loads(browser_use_cli.browser_exec("print('durable')", task_id="durable"))
    audio_path = home / "audio" / "durable.mp3"
    audio = json.loads(
        tts_tool._text_to_speech_single(
            "durable speech",
            str(audio_path),
            provider="edge",
            tts_config_override={"provider": "edge"},
        )
    )

    assert browser["workspace"] == str(home / "cache" / "browser-use" / "workspace" / "durable")
    assert audio["file_path"] == str(audio_path)
    assert audio_path.read_bytes() == VALID_MP3


@pytest.mark.parametrize("outcome", ["error", "timeout", "cancel"])
def test_ephemeral_tts_reaps_trusted_root_on_failure_or_cancel(
    tmp_path, monkeypatch, outcome
):
    from tools import tts_tool

    observed = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    class SinkAdapter:
        def generate(self, _request, sink):
            observed["fd"] = _sink_fd(sink)
            os.write(observed["fd"], VALID_MP3)
            if outcome == "cancel":
                raise asyncio.CancelledError("private cancel path")
            if outcome == "timeout":
                raise TimeoutError("private timeout path")
            raise RuntimeError("private provider path")

        def finish_owned_work(self):
            return None

        def stop_owned_work(self):
            return None

    monkeypatch.setattr(
        tts_tool,
        "_anonymous_adapter_for_provider",
        lambda *_args, **_kwargs: SinkAdapter(),
    )
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if outcome == "cancel":
            with pytest.raises(asyncio.CancelledError):
                tts_tool.text_to_speech_tool(
                    "private speech", provider="private-plugin"
                )
            rendered = ""
        else:
            rendered = tts_tool.text_to_speech_tool(
                "private speech", provider="private-plugin"
            )

    with pytest.raises(OSError):
        os.fstat(observed["fd"])
    if rendered:
        assert json.loads(rendered) == {
            "error": "TTS generation failed",
            "success": False,
        }
        assert str(tmp_path) not in rendered


def test_ephemeral_tts_never_calls_path_based_final_publication(
    tmp_path, monkeypatch
):
    from tools import tts_tool

    external = tmp_path / "unowned-final.mp3"
    external.write_bytes(b"unowned-final-audio")
    observed = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    class SinkAdapter:
        def generate(self, _request, sink):
            observed["fd"] = _sink_fd(sink)
            os.write(observed["fd"], VALID_MP3)
            return sink.path

        def finish_owned_work(self):
            return None

        def stop_owned_work(self):
            return None

    monkeypatch.setattr(
        tts_tool,
        "_anonymous_adapter_for_provider",
        lambda *_args, **_kwargs: SinkAdapter(),
    )
    publication_calls = []
    monkeypatch.setattr(
        tts_tool,
        "_build_audio_delivery_files",
        lambda *_a, **_k: publication_calls.append(True) or ([str(external)], False),
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = tts_tool.text_to_speech_tool(
            "private speech", provider="private-plugin"
        )

    parsed = json.loads(rendered)
    assert parsed["success"] is True
    assert parsed["audio"].startswith("data:audio/mpeg;base64,")
    assert publication_calls == []
    assert external.read_bytes() == b"unowned-final-audio"
    with pytest.raises(OSError):
        os.fstat(observed["fd"])
    assert str(tmp_path) not in rendered


def test_ephemeral_tts_delivery_cap_and_cleanup_failure_are_path_free(
    tmp_path, monkeypatch, caplog
):
    from tools import tts_tool

    observed = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    class SinkAdapter:
        def generate(self, _request, sink):
            observed["fd"] = _sink_fd(sink)
            os.write(observed["fd"], VALID_MP3)
            return sink.path

        def finish_owned_work(self):
            return None

        def stop_owned_work(self):
            return None

    monkeypatch.setattr(
        tts_tool,
        "_anonymous_adapter_for_provider",
        lambda *_args, **_kwargs: SinkAdapter(),
    )
    monkeypatch.setattr(
        tts_tool,
        "_resolve_audio_delivery_profile",
        lambda *_a: tts_tool.AudioDeliveryProfile("test", 4, 1.0),
    )
    from tools.tts_staging import AnonymousAudioStage, AnonymousAudioScrubError

    original_scrub = AnonymousAudioStage.scrub_and_close

    def fail_after_scrub(stage):
        original_scrub(stage)
        raise AnonymousAudioScrubError("private cleanup failure")

    monkeypatch.setattr(AnonymousAudioStage, "scrub_and_close", fail_after_scrub)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = tts_tool.text_to_speech_tool(
            "private speech", provider="private-plugin"
        )

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    with pytest.raises(OSError):
        os.fstat(observed["fd"])
    assert str(tmp_path) not in rendered
    assert str(tmp_path) not in caplog.text


def test_ephemeral_tts_private_state_stays_restrictive_after_inner_durable_rebind(
    tmp_path, monkeypatch
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    observed = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    class SinkAdapter:
        def generate(self, _request, sink):
            activate_invocation_persistence_policy(PersistencePolicy.DURABLE)
            observed["fd"] = _sink_fd(sink)
            os.write(observed["fd"], VALID_MP3)
            return sink.path

        def finish_owned_work(self):
            return None

        def stop_owned_work(self):
            return None

    monkeypatch.setattr(
        tts_tool,
        "_anonymous_adapter_for_provider",
        lambda *_args, **_kwargs: SinkAdapter(),
    )
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = tts_tool.text_to_speech_tool(
            "private speech", provider="private-plugin"
        )

    parsed = json.loads(rendered)
    assert parsed["success"] is True
    assert parsed["audio"].startswith("data:audio/mpeg;base64,")
    assert "file_path" not in parsed
    with pytest.raises(OSError):
        os.fstat(observed["fd"])
    assert str(tmp_path) not in rendered
