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


def _signature(path: Path):
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if path.is_symlink():
        payload = ("symlink", os.readlink(path))
    elif path.is_file():
        payload = ("file", path.read_bytes())
    elif path.is_dir():
        payload = ("dir", tuple(sorted(child.name for child in path.iterdir())))
    else:
        payload = ("other", None)
    return (
        payload,
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_gid,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


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


def test_late_ephemeral_rebind_cleans_requested_not_different_returned_path(
    tmp_path, monkeypatch
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    returned = tmp_path / "pre-existing-returned.mp3"
    returned.write_bytes(b"returned-sentinel")
    returned_before = returned.lstat()
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        Path(output_path).write_bytes(b"new-requested-audio")
        Path(output_path).chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return str(returned)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = tts_tool._text_to_speech_single(
            "private speech",
            str(requested),
            provider="private-plugin",
            tts_config_override={"provider": "private-plugin"},
        )

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert not requested.exists()
    returned_after = returned.lstat()
    assert returned.read_bytes() == b"returned-sentinel"
    assert (
        returned_after.st_dev,
        returned_after.st_ino,
        returned_after.st_uid,
        returned_after.st_gid,
        returned_after.st_mode,
        returned_after.st_nlink,
        returned_after.st_size,
        returned_after.st_mtime_ns,
        returned_after.st_ctime_ns,
    ) == (
        returned_before.st_dev,
        returned_before.st_ino,
        returned_before.st_uid,
        returned_before.st_gid,
        returned_before.st_mode,
        returned_before.st_nlink,
        returned_before.st_size,
        returned_before.st_mtime_ns,
        returned_before.st_ctime_ns,
    )


def _invoke_late_rebind(entry, tts_tool, requested: Path):
    if entry == "internal":
        return tts_tool._text_to_speech_single(
            "private speech",
            str(requested),
            provider="private-plugin",
            tts_config_override={"provider": "private-plugin"},
        )
    return tts_tool.text_to_speech_tool(
        "private speech",
        output_path=str(requested),
        provider="private-plugin",
    )


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("write_requested", [False, True])
@pytest.mark.parametrize(
    "returned_kind",
    ["same", "preexisting", "out-of-root", "absent", "empty", "malformed"],
)
def test_late_rebind_return_value_never_selects_cleanup(
    tmp_path, monkeypatch, entry, write_requested, returned_kind
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    returned = tmp_path / "returned-sentinel.mp3"
    returned.write_bytes(b"returned-sentinel")
    returned_before = _signature(returned)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-return.mp3"
    outside.write_bytes(b"outside-returned-sentinel")
    outside_before = _signature(outside)
    absent = tmp_path / "never-created-return.mp3"
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        if write_requested:
            Path(output_path).write_bytes(b"requested-audio")
            Path(output_path).chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return {
            "same": output_path,
            "preexisting": str(returned),
            "out-of-root": str(outside),
            "absent": str(absent),
            "empty": "",
            "malformed": object(),
        }[returned_kind]

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert not requested.exists()
    assert _signature(returned) == returned_before
    assert _signature(outside) == outside_before
    assert not absent.exists()
    assert str(tmp_path) not in rendered


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("write_requested", [False, True])
@pytest.mark.parametrize("returned_kind", ["same", "different"])
def test_late_rebind_never_deletes_preexisting_requested_or_returned(
    tmp_path, monkeypatch, entry, write_requested, returned_kind
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "preexisting-requested.mp3"
    returned = tmp_path / "preexisting-returned.mp3"
    requested.write_bytes(b"requested-before")
    returned.write_bytes(b"returned-before")
    returned_before = _signature(returned)
    requested_after_provider = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        if write_requested:
            Path(output_path).write_bytes(b"requested-provider-write")
        requested_after_provider["signature"] = _signature(requested)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return output_path if returned_kind == "same" else str(returned)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert _signature(requested) == requested_after_provider["signature"]
    assert _signature(returned) == returned_before


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("outcome", ["error", "timeout", "cancel"])
def test_late_rebind_cleans_exact_requested_on_every_provider_exit(
    tmp_path, monkeypatch, entry, outcome
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    returned = tmp_path / "returned.mp3"
    returned.write_bytes(b"returned-sentinel")
    returned_before = _signature(returned)
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        Path(output_path).write_bytes(b"partial-requested-audio")
        Path(output_path).chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        if outcome == "cancel":
            raise asyncio.CancelledError("private cancel path")
        if outcome == "timeout":
            raise TimeoutError("private timeout path")
        raise RuntimeError("private provider path")

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert not requested.exists()
    assert _signature(returned) == returned_before
    assert str(tmp_path) not in rendered


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize(
    "drift", ["replacement", "symlink", "hardlink", "mode", "group", "relink", "type"]
)
def test_late_rebind_preserves_drifted_requested_object(
    tmp_path, monkeypatch, entry, drift
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    external = tmp_path / "external-sentinel.mp3"
    external.write_bytes(b"external-sentinel")
    external_before = _signature(external)
    observed = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        path = Path(output_path)
        path.write_bytes(b"requested-audio")
        path.chmod(0o600)
        if drift == "replacement":
            path.unlink()
            path.write_bytes(b"replacement-sentinel")
            path.chmod(0o600)
        elif drift == "symlink":
            path.unlink()
            path.symlink_to(external)
        elif drift == "hardlink":
            path.unlink()
            os.link(external, path)
        elif drift == "mode":
            path.chmod(0o640)
        elif drift == "group":
            foreign_group = next(group for group in os.getgroups() if group != os.getgid())
            os.chown(path, -1, foreign_group)
        elif drift == "relink":
            backup = tmp_path / "relink-backup.mp3"
            os.link(path, backup)
            path.unlink()
            os.link(backup, path)
            backup.unlink()
        elif drift == "type":
            path.unlink()
            path.mkdir()
        observed["requested"] = _signature(path)
        observed["external"] = _signature(external)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return str(external)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert _signature(requested) == observed["requested"]
    assert _signature(external) == observed.get("external", external_before)


@pytest.mark.parametrize("entry", ["internal", "public"])
def test_late_rebind_parent_replacement_never_broadens_cleanup(
    tmp_path, monkeypatch, entry
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    parent = tmp_path / "requested-parent"
    parent.mkdir(mode=0o700)
    requested = parent / "requested.mp3"
    moved_parent = tmp_path / "moved-owned-parent"
    observed = {}
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        path = Path(output_path)
        path.write_bytes(b"requested-audio")
        path.chmod(0o600)
        parent.rename(moved_parent)
        parent.mkdir(mode=0o700)
        replacement = parent / path.name
        replacement.write_bytes(b"replacement-sentinel")
        observed["moved"] = _signature(moved_parent / path.name)
        observed["replacement"] = _signature(replacement)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return str(replacement)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert _signature(moved_parent / requested.name) == observed["moved"]
    assert _signature(requested) == observed["replacement"]


@pytest.mark.parametrize("entry", ["internal", "public"])
def test_late_rebind_unlink_failure_has_no_fallback(
    tmp_path, monkeypatch, entry
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    unrelated = tmp_path / "unrelated.mp3"
    unrelated.write_bytes(b"unrelated-sentinel")
    unrelated_before = _signature(unrelated)
    original_unlink = tts_tool.os.unlink
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        Path(output_path).write_bytes(b"requested-audio")
        Path(output_path).chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return str(unrelated)

    def fail_requested_unlink(path, *args, **kwargs):
        if kwargs.get("dir_fd") is not None and Path(path).name == requested.name:
            raise OSError("injected unlink failure")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    monkeypatch.setattr(tts_tool.os, "unlink", fail_requested_unlink)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert requested.exists()
    assert _signature(unrelated) == unrelated_before


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("failure", ["parent-open", "artifact-open"])
def test_requested_attestation_open_failure_is_categorical(
    tmp_path, monkeypatch, entry, failure
):
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    original_open = tts_tool.os.open
    called = False
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(*_args):
        nonlocal called
        called = True
        raise AssertionError("provider must not run after attestation failure")

    def fail_open(path, flags, *args, **kwargs):
        if failure == "parent-open" and Path(path) == requested.parent:
            raise OSError("injected parent open failure")
        if (
            failure == "artifact-open"
            and kwargs.get("dir_fd") is not None
            and Path(path).name == requested.name
        ):
            raise OSError("injected artifact open failure")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    monkeypatch.setattr(tts_tool.os, "open", fail_open)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert not called
    assert not requested.exists()
    assert str(tmp_path) not in rendered


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize(
    "failure", ["artifact-fstat", "parent-fstat", "pre-unlink-stat", "post-stat"]
)
def test_late_rebind_identity_failure_preserves_artifact_without_fallback(
    tmp_path, monkeypatch, entry, failure
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    unrelated = tmp_path / "unrelated.mp3"
    unrelated.write_bytes(b"unrelated-sentinel")
    unrelated_before = _signature(unrelated)
    original_fstat = tts_tool.os.fstat
    original_stat = tts_tool.os.stat
    original_open = tts_tool.os.open
    original_unlink = tts_tool.os.unlink
    unlinks = []
    opens = []
    artifact_descriptor = None
    parent_descriptor = None
    provider_completed = False
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        nonlocal provider_completed
        assert parent_descriptor is not None, opens
        assert artifact_descriptor is not None, opens
        path = Path(output_path)
        path.write_bytes(b"requested-audio")
        path.chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        provider_completed = True
        return str(unrelated)

    def capture_open(path, flags, *args, **kwargs):
        nonlocal artifact_descriptor, parent_descriptor
        descriptor = original_open(path, flags, *args, **kwargs)
        opens.append((os.fspath(path), kwargs.get("dir_fd"), descriptor))
        if Path(path) == requested.parent and kwargs.get("dir_fd") is None:
            parent_descriptor = descriptor
        elif (
            Path(path).name == requested.name
            and kwargs.get("dir_fd") == parent_descriptor
        ):
            artifact_descriptor = descriptor
        return descriptor

    def fail_fstat(descriptor):
        if (
            provider_completed
            and failure == "artifact-fstat"
            and descriptor == artifact_descriptor
        ):
            raise OSError("injected artifact identity failure")
        if (
            provider_completed
            and failure == "parent-fstat"
            and descriptor == parent_descriptor
        ):
            raise OSError("injected parent identity failure")
        return original_fstat(descriptor)

    def capture_unlink(path, *args, **kwargs):
        unlinks.append((path, kwargs.get("dir_fd")))
        return original_unlink(path, *args, **kwargs)

    stat_calls = 0

    def fail_post_stat(path, *args, **kwargs):
        nonlocal stat_calls
        if (
            provider_completed
            and failure in {"pre-unlink-stat", "post-stat"}
            and kwargs.get("dir_fd") == parent_descriptor
            and Path(path).name == requested.name
        ):
            stat_calls += 1
            fail_at = 2 if failure == "pre-unlink-stat" else 3
            if stat_calls == fail_at:
                raise OSError("injected cleanup revalidation failure")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    monkeypatch.setattr(tts_tool.os, "open", capture_open)
    monkeypatch.setattr(tts_tool.os, "fstat", fail_fstat)
    monkeypatch.setattr(tts_tool.os, "stat", fail_post_stat)
    monkeypatch.setattr(tts_tool.os, "unlink", capture_unlink)
    monkeypatch.setattr(
        tts_tool.os,
        "supports_dir_fd",
        set(tts_tool.os.supports_dir_fd) | {capture_open, fail_post_stat},
    )
    monkeypatch.setattr(
        tts_tool.os,
        "supports_follow_symlinks",
        set(tts_tool.os.supports_follow_symlinks) | {fail_post_stat},
    )
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    if failure == "post-stat":
        assert _signature(requested) is None
        assert unlinks == [(requested.name, parent_descriptor)]
    else:
        assert _signature(requested) is not None, (
            unlinks,
            opens,
            parent_descriptor,
            artifact_descriptor,
        )
        assert unlinks == []
    assert _signature(unrelated) == unrelated_before
    assert str(tmp_path) not in rendered


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("preexisting", [False, True])
@pytest.mark.parametrize("write_requested", [False, True])
def test_requested_attestation_preserves_durable_behavior(
    tmp_path, monkeypatch, entry, preexisting, write_requested
):
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    if preexisting:
        requested.write_bytes(b"requested-before")
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        if write_requested:
            Path(output_path).write_bytes(b"durable-audio")
            Path(output_path).chmod(0o600)
        return output_path

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    parsed = json.loads(rendered)
    if write_requested or preexisting:
        assert parsed["success"] is True
        assert requested.exists()
        assert requested.read_bytes() == (
            b"durable-audio" if write_requested else b"requested-before"
        )
    else:
        assert parsed["success"] is False
        assert not requested.exists()


@pytest.mark.parametrize("entry", ["internal", "public"])
def test_late_rebind_blocks_returned_path_before_post_provider_processing(
    tmp_path, monkeypatch, entry
):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    returned = tmp_path / "preexisting-returned.ogg"
    returned.write_bytes(b"returned-sentinel")
    returned_before = _signature(returned)
    processed = []
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        Path(output_path).write_bytes(b"requested-audio")
        Path(output_path).chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        return str(returned)

    def observe_repair(path):
        processed.append(path)
        return path

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    monkeypatch.setattr(tts_tool, "_repair_ogg_container", observe_repair)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert processed == []
    assert not requested.exists()
    assert _signature(returned) == returned_before


@pytest.mark.parametrize("entry", ["internal", "public"])
def test_late_rebind_provider_error_is_not_logged(tmp_path, monkeypatch, caplog, entry):
    from hermes_cli.persistence import activate_invocation_persistence_policy
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    private_marker = "PRIVATE-LATE-ERROR-/untrusted/provider/path"
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(_text, output_path, _provider, _config):
        Path(output_path).write_bytes(b"requested-audio")
        Path(output_path).chmod(0o600)
        activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
        raise RuntimeError(private_marker)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert private_marker not in caplog.text
    assert not requested.exists()


@pytest.mark.parametrize("entry", ["internal", "public"])
@pytest.mark.parametrize("failure", ["fchmod", "fchown"])
def test_requested_attestation_setup_failure_removes_exact_placeholder(
    tmp_path, monkeypatch, entry, failure
):
    from tools import tts_tool

    requested = tmp_path / "requested.mp3"
    called = False
    unlinks = []
    original_unlink = tts_tool.os.unlink
    monkeypatch.setattr(tts_tool, "_resolve_command_provider_config", lambda *_a: None)

    def provider(*_args):
        nonlocal called
        called = True
        raise AssertionError("provider must not run after attestation failure")

    def fail_setup(*_args, **_kwargs):
        raise OSError(f"injected {failure} failure")

    def capture_unlink(path, *args, **kwargs):
        unlinks.append((path, kwargs.get("dir_fd")))
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(tts_tool, "_dispatch_to_plugin_provider", provider)
    monkeypatch.setattr(tts_tool.os, failure, fail_setup)
    monkeypatch.setattr(tts_tool.os, "unlink", capture_unlink)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        rendered = _invoke_late_rebind(entry, tts_tool, requested)

    assert json.loads(rendered) == {
        "error": "TTS generation failed",
        "success": False,
    }
    assert not called
    assert not requested.exists(), unlinks
    assert str(tmp_path) not in rendered
