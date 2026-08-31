"""Unforgeable-authority regressions for invocation-scoped ephemeral TTS."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy


def _tree(root: Path) -> dict[str, tuple[str, bytes | None, int]]:
    if not root.exists():
        return {}
    result: dict[str, tuple[str, bytes | None, int]] = {}
    for path in sorted(root.rglob("*")):
        info = path.lstat()
        result[path.relative_to(root).as_posix()] = (
            "dir" if path.is_dir() else "file",
            None if path.is_dir() else path.read_bytes(),
            info.st_ino,
        )
    return result


def _run_in(execution: str, call):
    if execution == "direct":
        return call()
    if execution == "copied":
        context = contextvars.copy_context()
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(context.run, call).result()

    async def run():
        return await asyncio.to_thread(call)

    return asyncio.run(run())


def _install_forged_global(monkeypatch, tts_tool, root: Path):
    root.mkdir(parents=True, mode=0o700)
    parent_fd = os.open(root.parent, os.O_RDONLY)
    root_fd = os.open(root, os.O_RDONLY)
    info = root.lstat()
    forged = SimpleNamespace(
        parent=root.parent,
        root=root,
        parent_fd=parent_fd,
        root_fd=root_fd,
        root_identity=(info.st_dev, info.st_ino),
        outputs={},
    )
    slot = SimpleNamespace(get=lambda: forged, set=lambda _value: object(), reset=lambda _t: None)
    monkeypatch.setattr(tts_tool, "_EphemeralTTSState", SimpleNamespace, raising=False)
    monkeypatch.setattr(tts_tool, "_EPHEMERAL_TTS_STATE", slot, raising=False)
    return forged


def _patch_exact_plugin(monkeypatch, tts_tool, observed: dict, mode: str = "exact"):
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)
    monkeypatch.setattr(tts_tool, "_plugin_provider_is_voice_compatible", lambda _p: False)

    def provider(_text, output_path, _provider, _config):
        requested = Path(output_path)
        observed["requested"] = requested
        if mode == "exact":
            requested.write_bytes(b"private-owned-audio")
            return str(requested)
        if mode == "file_replacement":
            requested.unlink(missing_ok=True)
            requested.write_bytes(b"unowned-replacement-audio")
            observed["unowned"] = requested
            return str(requested)
        if mode == "wrong_mode":
            requested.write_bytes(b"private-wrong-mode")
            requested.chmod(0o644)
            observed["unowned"] = requested
            return str(requested)
        if mode == "wrong_group":
            requested.write_bytes(b"private-foreign-group")
            foreign_gid = next(group for group in os.getgroups() if group != os.getgid())
            os.chown(requested, -1, foreign_gid)
            observed["unowned"] = requested
            return str(requested)
        if mode == "extra_entry":
            requested.write_bytes(b"private-owned-audio")
            extra = requested.parent / "unowned-extra.txt"
            extra.write_bytes(b"unowned-extra")
            observed["unowned"] = extra
            return str(requested)
        raise AssertionError(mode)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("execution", ["direct", "copied", "async"])
def test_forged_module_global_state_cannot_select_tts_root(
    tmp_path, monkeypatch, entry, execution
):
    from tools import tts_tool

    profile = tmp_path / ".hermes"
    forged_root = tmp_path / "caller-selected-root"
    forged = _install_forged_global(monkeypatch, tts_tool, forged_root)
    before = _tree(tmp_path)
    observed = {}
    _patch_exact_plugin(monkeypatch, tts_tool, observed)
    monkeypatch.setenv("HERMES_HOME", str(profile))

    def call():
        if entry == "internal":
            return tts_tool._text_to_speech_single(
                "private speech",
                str(profile / "audio" / "private.mp3"),
                provider="private-plugin",
                tts_config_override={"provider": "private-plugin"},
            )
        return tts_tool.text_to_speech_tool(
            "private speech", provider="private-plugin"
        )

    try:
        with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
            rendered = _run_in(execution, call)
    finally:
        for fd in (forged.root_fd, forged.parent_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    parsed = json.loads(rendered)
    assert parsed["success"] is True
    assert parsed["audio"].startswith("data:audio/mpeg;base64,")
    assert "file_path" not in parsed
    assert _tree(tmp_path) == before
    assert not profile.exists()
    assert observed["requested"].parent != forged_root
    assert not observed["requested"].parent.exists()


def test_forged_cleanup_state_cannot_delete_caller_tree(tmp_path):
    from tools import tts_tool

    parent = tmp_path / "caller-parent"
    root = parent / "caller-root"
    root.mkdir(parents=True, mode=0o700)
    sentinel = root / "do-not-delete.txt"
    sentinel.write_bytes(b"caller-owned-sentinel")
    before = _tree(parent)
    parent_info = parent.lstat()
    root_info = root.lstat()
    forged = SimpleNamespace(
        parent=parent,
        root=root,
        parent_fd=os.open(parent, os.O_RDONLY),
        root_fd=os.open(root, os.O_RDONLY),
        root_identity=(root_info.st_dev, root_info.st_ino),
        parent_identity=(parent_info.st_dev, parent_info.st_ino),
        outputs={},
    )

    cleanup = getattr(tts_tool, "_cleanup_ephemeral_tts_state", None)
    try:
        if callable(cleanup):
            cleanup(forged)
    finally:
        for fd in (forged.root_fd, forged.parent_fd):
            try:
                os.close(fd)
            except OSError:
                pass

    assert _tree(parent) == before
    assert sentinel.read_bytes() == b"caller-owned-sentinel"


def test_no_importable_tts_authority_or_cleanup_helpers_remain():
    from tools import tts_tool

    forbidden = (
        "_EphemeralTTSState",
        "_EPHEMERAL_TTS_STATE",
        "_directory_fd",
        "_same_identity",
        "_clean_owned_dir_fd",
        "_clean_owned_dir_path",
        "_cleanup_ephemeral_tts_state",
        "_trusted_ephemeral_tts_scope",
        "_read_proved_ephemeral_audio",
        "_ephemeral_result_path",
    )

    assert {name for name in forbidden if hasattr(tts_tool, name)} == set()


@pytest.mark.parametrize(
    "forged",
    [
        {"root": "/tmp/caller"},
        ("/tmp/caller", 7, (1, 2)),
        SimpleNamespace(root=Path("/tmp/caller"), descriptor=7),
    ],
)
@pytest.mark.parametrize("entry", ["internal", "public"])
def test_caller_cannot_pass_tts_authority(tmp_path, forged, entry):
    from tools import tts_tool

    sentinel = tmp_path / "sentinel"
    sentinel.write_bytes(b"unchanged")
    before = _tree(tmp_path)
    target = (
        tts_tool._text_to_speech_single
        if entry == "internal"
        else tts_tool.text_to_speech_tool
    )
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(TypeError):
            target("private speech", _ephemeral_authority=forged)
    assert _tree(tmp_path) == before


@pytest.mark.parametrize("entry", ["internal", "public"])
def test_ephemeral_tts_success_cleans_whole_temp_and_profile(
    tmp_path, monkeypatch, entry
):
    from tools import tts_tool

    runtime = tmp_path / "runtime"
    profile = tmp_path / ".hermes"
    runtime.mkdir()
    monkeypatch.setattr(tts_tool.tempfile, "tempdir", str(runtime))
    monkeypatch.setenv("HERMES_HOME", str(profile))
    observed = {}
    _patch_exact_plugin(monkeypatch, tts_tool, observed)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        if entry == "internal":
            rendered = tts_tool._text_to_speech_single(
                "private speech",
                provider="private-plugin",
                tts_config_override={"provider": "private-plugin"},
            )
        else:
            rendered = tts_tool.text_to_speech_tool(
                "private speech", provider="private-plugin"
            )
    parsed = json.loads(rendered)
    assert parsed["success"] is True
    assert parsed["audio"].startswith("data:audio/mpeg;base64,")
    assert list(runtime.iterdir()) == []
    assert not profile.exists()


@pytest.mark.parametrize(
    "mode", ["file_replacement", "wrong_mode", "wrong_group", "extra_entry"]
)
def test_ephemeral_tts_rejects_unowned_exact_name_or_namespace_without_deleting_it(
    tmp_path, monkeypatch, mode
):
    from tools import tts_tool

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(tts_tool.tempfile, "tempdir", str(runtime))
    observed = {}
    _patch_exact_plugin(monkeypatch, tts_tool, observed, mode)

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = tts_tool._text_to_speech_single(
            "private speech",
            provider="private-plugin",
            tts_config_override={"provider": "private-plugin"},
        )

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    if mode == "wrong_mode":
        # The inode is still the invocation-owned artifact, so cleanup may
        # safely scrub and unlink it after rejecting its altered mode.
        assert not observed["unowned"].exists()
        assert list(runtime.iterdir()) == []
        return
    assert observed["unowned"].exists()
    expected = (
        b"unowned-extra" if mode == "extra_entry" else
        b"private-foreign-group" if mode == "wrong_group" else
        b"unowned-replacement-audio"
    )
    assert observed["unowned"].read_bytes() == expected
    assert str(tmp_path) not in rendered


def test_ephemeral_tts_scrubs_hostile_hardlink_to_owned_artifact(
    tmp_path, monkeypatch
):
    from tools import tts_tool

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    leak = tmp_path / "hostile-link.mp3"
    monkeypatch.setattr(tts_tool.tempfile, "tempdir", str(runtime))
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        requested = Path(output_path)
        requested.write_bytes(b"private-hardlinked-audio")
        os.link(requested, leak)
        return str(requested)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = tts_tool._text_to_speech_single(
            "private speech",
            provider="private-plugin",
            tts_config_override={"provider": "private-plugin"},
        )

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert leak.exists()
    assert leak.read_bytes() == b""
    assert list(runtime.iterdir()) == []


def test_ephemeral_tts_result_construction_failure_still_cleans(
    tmp_path, monkeypatch
):
    from tools import tts_tool

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    monkeypatch.setattr(tts_tool.tempfile, "tempdir", str(runtime))
    observed = {}
    _patch_exact_plugin(monkeypatch, tts_tool, observed)
    monkeypatch.setattr(
        tts_tool.base64,
        "b64encode",
        lambda _raw: (_ for _ in ()).throw(RuntimeError("private encoding failure")),
    )
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        rendered = tts_tool._text_to_speech_single(
            "private speech",
            provider="private-plugin",
            tts_config_override={"provider": "private-plugin"},
        )
    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert list(runtime.iterdir()) == []


def test_durable_tts_fails_closed_if_policy_becomes_ephemeral(
    tmp_path, monkeypatch
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    output = tmp_path / "late-policy.mp3"
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        Path(output_path).write_bytes(b"must-not-persist")
        return output_path

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = tts_tool._text_to_speech_single(
            "private speech",
            str(output),
            provider="private-plugin",
            tts_config_override={"provider": "private-plugin"},
        )
    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert not output.exists()
