from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy
from tools.tts_publish import (
    PublishedAudio,
    TTSPublishError,
    TTSPublishUncertain,
    _require_absent_primitive_for_platform,
    publish_durable,
)
from tools.tts_staging import _create_anonymous_audio_stage_for_test
from tools.tts_transaction import (
    DurablePublicationPermit,
    TTSTransaction,
    TTSTransactionError,
    TTSTransactionStop,
)


VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00synthetic-audio"


def _sealed_stage(parent: Path, payload: bytes = VALID_MP3):
    parent.mkdir(parents=True, exist_ok=True)
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, parent)
    os.write(int(Path(stage.sink.path).name), payload)
    return stage, stage.seal(stage.sink.path)


def _publish_one(stage_parent: Path, destination: Path) -> PublishedAudio:
    stage, sealed = _sealed_stage(stage_parent)
    published = None
    error = None
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            try:
                published = publish_durable(transaction.decide(), destination)
            except TTSPublishError as exc:
                error = exc
    if error is not None:
        raise error
    assert published is not None
    return published


def test_absent_destination_is_no_replace_and_parent_fsynced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "output" / "voice.mp3"
    destination.parent.mkdir()
    events: list[str] = []
    real_copy = tts_publish._copy_sealed_to_publication
    real_file_fsync = tts_publish._fsync_publication_file
    real_parent_fsync = tts_publish._fsync_parent

    def record_copy(*args, **kwargs):
        events.append("copy")
        return real_copy(*args, **kwargs)

    def record_file_fsync(fd):
        events.append("file-fsync")
        return real_file_fsync(fd)

    def record_parent_fsync(fd):
        events.append("parent-fsync")
        return real_parent_fsync(fd)

    monkeypatch.setattr(tts_publish, "_copy_sealed_to_publication", record_copy)
    monkeypatch.setattr(tts_publish, "_fsync_publication_file", record_file_fsync)
    monkeypatch.setattr(tts_publish, "_fsync_parent", record_parent_fsync)

    published = _publish_one(tmp_path / "stage", destination)

    assert published.path == destination
    assert destination.read_bytes() == VALID_MP3
    assert events == ["copy", "file-fsync", "parent-fsync"]


def test_existing_destination_uses_authorized_atomic_replace(tmp_path: Path):
    destination = tmp_path / "voice.mp3"
    destination.write_bytes(b"old")
    published = _publish_one(tmp_path / "stage", destination)

    assert published.path == destination
    assert destination.read_bytes() == VALID_MP3


def test_permit_rejects_clone_before_destination_access(tmp_path: Path):
    stage, sealed = _sealed_stage(tmp_path / "stage")
    destination = tmp_path / "missing-parent" / "voice.mp3"
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            permit = transaction.decide()
            clone = object.__new__(DurablePublicationPermit)
            with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
                publish_durable(clone, destination)
            assert not destination.parent.exists()
            permit._consume_for_publication(
                lambda stages, _observation: [item[0].scrub_and_close() for item in stages]
            )


def test_permit_replay_and_cross_transaction_reject_before_destination_access(
    tmp_path: Path,
):
    first, first_sealed = _sealed_stage(tmp_path / "first")
    second, second_sealed = _sealed_stage(tmp_path / "second")
    destination = tmp_path / "voice.mp3"
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as outer:
            outer.add_sealed(first, first_sealed)
            permit = outer.decide()
            with TTSTransaction.begin(4096) as inner:
                inner.add_sealed(second, second_sealed)
                with pytest.raises(TTSTransactionError):
                    publish_durable(permit, destination)
                assert not destination.exists()
                inner.decide()._consume_for_publication(
                    lambda stages, _observation: [
                        item[0].scrub_and_close() for item in stages
                    ]
                )
            publish_durable(permit, destination)
            with pytest.raises(TTSTransactionError):
                publish_durable(permit, tmp_path / "replay.mp3")
            assert not (tmp_path / "replay.mp3").exists()


def test_inactive_permit_rejects_without_destination_access(tmp_path: Path):
    stage, sealed = _sealed_stage(tmp_path / "stage")
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            permit = transaction.decide()
    destination = tmp_path / "voice.mp3"
    with pytest.raises(TTSTransactionError):
        publish_durable(permit, destination)
    assert not destination.exists()


def test_final_latch_flip_before_publish_scrubs_and_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    real_verify = tts_publish._verify_publication_digest

    def verify_then_flip(*args, **kwargs):
        result = real_verify(*args, **kwargs)
        with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
            pass
        return result

    monkeypatch.setattr(tts_publish, "_verify_publication_digest", verify_then_flip)
    with pytest.raises(TTSPublishError, match="^tts_durable_publication_failed$"):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    residues = list(tmp_path.glob(".hermes-tts-publish-*"))
    assert all(path.stat().st_size == 0 for path in residues)


def test_publication_temp_is_same_filesystem_0600_regular_and_single_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    seen: list[os.stat_result] = []
    real_copy = tts_publish._copy_sealed_to_publication

    def inspect_temp(stage, sealed, temp):
        seen.append(os.fstat(temp.fd))
        return real_copy(stage, sealed, temp)

    monkeypatch.setattr(tts_publish, "_copy_sealed_to_publication", inspect_temp)
    _publish_one(tmp_path / "stage", destination)

    assert len(seen) == 1
    assert seen[0].st_dev == destination.parent.stat().st_dev
    assert seen[0].st_mode & 0o777 == 0o600
    assert seen[0].st_nlink == 1


@pytest.mark.parametrize("fault", ["copy", "file-fsync", "rename"])
def test_copy_file_fsync_and_rename_failures_scrub_without_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    if fault == "copy":
        monkeypatch.setattr(
            tts_publish,
            "_copy_sealed_to_publication",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private")),
        )
    elif fault == "file-fsync":
        monkeypatch.setattr(
            tts_publish,
            "_fsync_publication_file",
            lambda _fd: (_ for _ in ()).throw(OSError("private")),
        )
    else:
        monkeypatch.setattr(
            tts_publish,
            "_rename_absent_for_host",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private")),
        )

    with pytest.raises(TTSPublishError) as exc_info:
        _publish_one(tmp_path / "stage", destination)
    assert str(exc_info.value) == "tts_durable_publication_failed"
    assert not destination.exists()
    assert all(path.stat().st_size == 0 for path in tmp_path.glob(".hermes-tts-publish-*"))


def test_parent_fsync_failure_reports_uncertain_without_unsafe_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    monkeypatch.setattr(
        tts_publish,
        "_fsync_parent",
        lambda _fd: (_ for _ in ()).throw(OSError("private")),
    )
    with pytest.raises(TTSPublishUncertain) as exc_info:
        _publish_one(tmp_path / "stage", destination)
    assert str(exc_info.value) == "tts_durable_publication_uncertain"
    assert destination.read_bytes() == VALID_MP3


def test_zero_length_recovery_residue_is_not_pathname_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    monkeypatch.setattr(
        tts_publish,
        "_rename_absent_for_host",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private")),
    )
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    residues = list(tmp_path.glob(".hermes-tts-publish-*"))
    assert len(residues) == 1
    assert residues[0].stat().st_size == 0


def test_temp_name_substitution_scrubs_held_original_without_unlinking_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    replacement = b"untrusted-replacement"
    real_revalidate = tts_publish._revalidate_publication_temp

    def substitute_before_revalidation(parent, temp, size):
        names = list(tmp_path.glob(".hermes-tts-publish-*"))
        assert len(names) == 1
        original = names[0]
        moved = tmp_path / "held-original"
        original.rename(moved)
        original.write_bytes(replacement)
        return real_revalidate(parent, temp, size)

    monkeypatch.setattr(
        tts_publish, "_revalidate_publication_temp", substitute_before_revalidation
    )
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    current = list(tmp_path.glob(".hermes-tts-publish-*"))
    assert len(current) == 1
    assert current[0].read_bytes() == replacement
    assert (tmp_path / "held-original").stat().st_size == 0


@pytest.mark.parametrize("drift", ["mode", "hardlink", "inode", "symlink"])
def test_publication_temp_drift_fails_closed_and_preserves_unproved_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    replacement = b"unproved-object"
    real_revalidate = tts_publish._revalidate_publication_temp

    def drift_then_revalidate(parent, temp, size):
        named = tmp_path / temp.name
        if drift == "mode":
            named.chmod(0o640)
        elif drift == "hardlink":
            os.link(named, tmp_path / "alias")
        else:
            held_original = tmp_path / "held-original"
            named.rename(held_original)
            if drift == "symlink":
                target = tmp_path / "replacement-target"
                target.write_bytes(replacement)
                named.symlink_to(target)
            else:
                named.write_bytes(replacement)
        return real_revalidate(parent, temp, size)

    monkeypatch.setattr(
        tts_publish, "_revalidate_publication_temp", drift_then_revalidate
    )
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    if drift in ("inode", "symlink"):
        named = next(tmp_path.glob(".hermes-tts-publish-*"))
        if drift == "symlink":
            assert named.is_symlink()
            assert named.resolve().read_bytes() == replacement
        else:
            assert named.read_bytes() == replacement
        assert (tmp_path / "held-original").stat().st_size == 0
    elif drift == "hardlink":
        assert (tmp_path / "alias").stat().st_size == 0
    else:
        residue = next(tmp_path.glob(".hermes-tts-publish-*"))
        assert residue.stat().st_size == 0
        assert residue.stat().st_mode & 0o777 == 0o640


@pytest.mark.parametrize(("field", "index"), [("uid", 4), ("gid", 5)])
def test_publication_temp_uid_gid_drift_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    index: int,
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    real_revalidate = tts_publish._revalidate_publication_temp
    real_stat = os.stat

    def drift_then_revalidate(parent, temp, size):
        def altered_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path == temp.name and kwargs.get("dir_fd") == parent.fd:
                values = list(result)
                values[index] += 1
                return os.stat_result(values)
            return result

        monkeypatch.setattr(tts_publish.os, "stat", altered_stat)
        return real_revalidate(parent, temp, size)

    monkeypatch.setattr(
        tts_publish, "_revalidate_publication_temp", drift_then_revalidate
    )
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists(), field
    residue = next(tmp_path.glob(".hermes-tts-publish-*"))
    assert residue.stat().st_size == 0


def test_unmodified_final_interval_publishes_exactly_once(tmp_path: Path):
    destination = tmp_path / "voice.mp3"
    _publish_one(tmp_path / "stage", destination)
    assert destination.read_bytes() == VALID_MP3


def test_rebound_preparer_cannot_inject_callback_after_final_policy_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"

    def fake_prepare(*, parent_fd, source, destination, **_kwargs):
        def evil_publish(src, dst, *, src_dir_fd, dst_dir_fd):
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                os.replace(
                    src,
                    dst,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )

        return tts_publish._PreparedPublicationCall(
            kind="replace",
            callable_ref=evil_publish,
            source=source,
            destination=destination,
            source_dir_fd=parent_fd,
            destination_dir_fd=parent_fd,
            flags=0,
        )

    monkeypatch.setattr(tts_publish, "_prepare_publication_call", fake_prepare)
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()


def test_rebound_prepared_call_constructor_cannot_inject_final_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"

    def fake_constructor(**values):
        def evil_publish(src, dst, *, src_dir_fd, dst_dir_fd):
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                os.replace(
                    src,
                    dst,
                    src_dir_fd=src_dir_fd,
                    dst_dir_fd=dst_dir_fd,
                )

        values["kind"] = "replace"
        values["callable_ref"] = evil_publish
        return type("InjectedCall", (), values)()

    monkeypatch.setattr(tts_publish, "_PreparedPublicationCall", fake_constructor)
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()


def test_rebound_digest_verifier_cannot_mutate_after_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    real_verify = tts_publish._verify_publication_digest

    def verify_then_mutate(temp, sealed):
        real_verify(temp, sealed)
        os.pwrite(temp.fd, b"X", len(VALID_MP3) - 1)

    monkeypatch.setattr(tts_publish, "_verify_publication_digest", verify_then_mutate)
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "slot",
    [
        "os.fsencode",
        "_rename_absent_for_host",
        "_rename_absent_darwin",
        "_darwin_renameatx_np",
        "_DARWIN_RENAMEATX_NP",
    ],
)
def test_darwin_mutable_post_check_slot_cannot_hide_durable_ephemeral_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    slot: str,
):
    from tools import tts_publish

    if sys.platform != "darwin":
        pytest.skip("Darwin runtime required")
    destination = tmp_path / "voice.mp3"
    if slot == "os.fsencode":
        original = tts_publish.os.fsencode

        def tripwire(*args, **kwargs):
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                pass
            return original(*args, **kwargs)

        monkeypatch.setattr(tts_publish.os, "fsencode", tripwire)
    else:
        original = getattr(tts_publish, slot)

        def tripwire(*args, **kwargs):
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                pass
            return original(*args, **kwargs)

        monkeypatch.setattr(tts_publish, slot, tripwire)

    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".hermes-tts-publish-*")
    )


def test_existing_replace_mutable_slot_cannot_hide_policy_flip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    destination.write_bytes(b"old")
    original = tts_publish._replace_existing

    def tripwire(*args, **kwargs):
        with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
            pass
        return original(*args, **kwargs)

    monkeypatch.setattr(tts_publish, "_replace_existing", tripwire)
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert destination.read_bytes() == b"old"


@pytest.mark.parametrize("relative", [".ssh/voice.mp3", ".aws/voice.mp3"])
def test_canonical_file_safety_blocks_protected_home_targets_before_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    destination = home / relative
    with pytest.raises((TTSPublishError, TTSTransactionError)):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    assert not list(destination.parent.glob(".hermes-tts-publish-*"))


def test_canonical_file_safety_checks_approval_gate_before_format_or_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    destination = home / ".ssh" / "config"
    called = False
    real_gate = tts_publish.is_write_approval_required

    def record_gate(path):
        nonlocal called
        called = True
        return real_gate(path)

    monkeypatch.setattr(tts_publish, "is_write_approval_required", record_gate)
    with pytest.raises((TTSPublishError, TTSTransactionError)):
        _publish_one(tmp_path / "stage", destination)
    assert called is True
    assert not destination.exists()
    assert not list(destination.parent.glob(".hermes-tts-publish-*"))


def test_canonical_file_safety_blocks_fixed_system_denial_before_access(
    tmp_path: Path,
):
    destination = Path("/etc/systemd/voice.mp3")
    with pytest.raises((TTSPublishError, TTSTransactionError)):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()


def test_relative_destination_is_frozen_before_cwd_can_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "out").mkdir(parents=True)
    (second / "out").mkdir(parents=True)
    monkeypatch.chdir(first)
    real_gate = tts_publish.is_write_approval_required

    def change_cwd_after_classification(path):
        result = real_gate(path)
        os.chdir(second)
        return result

    monkeypatch.setattr(
        tts_publish, "is_write_approval_required", change_cwd_after_classification
    )
    _publish_one(tmp_path / "stage", Path("out/voice.mp3"))
    assert (first / "out/voice.mp3").read_bytes() == VALID_MP3
    assert not (second / "out/voice.mp3").exists()


def test_same_length_held_temp_mutation_after_fsync_fails_digest_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    real_fsync = tts_publish._fsync_publication_file

    def fsync_then_mutate(fd):
        real_fsync(fd)
        os.pwrite(fd, b"X", len(VALID_MP3) - 1)

    monkeypatch.setattr(tts_publish, "_fsync_publication_file", fsync_then_mutate)
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".hermes-tts-publish-*")
    )


def test_same_length_temp_mutation_in_last_revalidation_fails_digest_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    real_revalidate = tts_publish._revalidate_publication_temp

    def revalidate_then_mutate(parent, temp, size):
        real_revalidate(parent, temp, size)
        os.pwrite(temp.fd, b"X", len(VALID_MP3) - 1)

    monkeypatch.setattr(
        tts_publish, "_revalidate_publication_temp", revalidate_then_mutate
    )
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".hermes-tts-publish-*")
    )


def test_temp_scrub_cancellation_still_closes_parent_and_scrubs_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    class Cancelled(BaseException):
        pass

    destination = tmp_path / "voice.mp3"
    real_temp_scrub = tts_publish._scrub_and_close_fd
    real_stage_scrub = tts_publish.AnonymousAudioStage.scrub_and_close
    source_scrubbed = False
    cancelled = False

    def scrub_then_cancel(fd):
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            raise Cancelled()
        return real_temp_scrub(fd)

    def record_source(self):
        nonlocal source_scrubbed
        real_stage_scrub(self)
        source_scrubbed = True

    monkeypatch.setattr(tts_publish, "_scrub_and_close_fd", scrub_then_cancel)
    monkeypatch.setattr(tts_publish.AnonymousAudioStage, "scrub_and_close", record_source)
    monkeypatch.setattr(
        tts_publish,
        "_rename_absent_for_host",
        lambda *_args: (_ for _ in ()).throw(OSError("private")),
    )
    with pytest.raises((Cancelled, TTSTransactionStop)):
        _publish_one(tmp_path / "stage", destination)
    assert source_scrubbed is True
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".hermes-tts-publish-*")
    )


def test_parent_close_cancellation_still_scrubs_source_and_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    class Cancelled(BaseException):
        pass

    destination = tmp_path / "voice.mp3"
    parent_identity = destination.parent.stat()
    real_close = os.close
    real_stage_scrub = tts_publish.AnonymousAudioStage.scrub_and_close
    source_scrubbed = False
    cancelled = False

    def close_then_cancel(fd):
        nonlocal cancelled
        held = os.fstat(fd)
        if (
            not cancelled
            and held.st_dev == parent_identity.st_dev
            and held.st_ino == parent_identity.st_ino
        ):
            cancelled = True
            raise Cancelled()
        real_close(fd)

    def record_source(self):
        nonlocal source_scrubbed
        real_stage_scrub(self)
        source_scrubbed = True

    monkeypatch.setattr(tts_publish.os, "close", close_then_cancel)
    monkeypatch.setattr(tts_publish.AnonymousAudioStage, "scrub_and_close", record_source)
    monkeypatch.setattr(
        tts_publish,
        "_rename_absent_for_host",
        lambda *_args: (_ for _ in ()).throw(OSError("private")),
    )
    with pytest.raises((Cancelled, TTSTransactionStop)):
        _publish_one(tmp_path / "stage", destination)
    assert source_scrubbed is True
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".hermes-tts-publish-*")
    )


def test_source_cleanup_cancellation_is_rethrown_after_other_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    class Cancelled(BaseException):
        pass

    destination = tmp_path / "voice.mp3"
    real_stage_scrub = tts_publish.AnonymousAudioStage.scrub_and_close
    cancelled = False

    def scrub_then_cancel(self):
        nonlocal cancelled
        if not cancelled:
            cancelled = True
            raise Cancelled()
        return real_stage_scrub(self)

    monkeypatch.setattr(
        tts_publish.AnonymousAudioStage, "scrub_and_close", scrub_then_cancel
    )
    monkeypatch.setattr(
        tts_publish,
        "_rename_absent_for_host",
        lambda *_args: (_ for _ in ()).throw(OSError("private")),
    )
    with pytest.raises((Cancelled, TTSTransactionStop)):
        _publish_one(tmp_path / "stage", destination)
    assert all(
        path.stat().st_size == 0
        for path in tmp_path.glob(".hermes-tts-publish-*")
    )


def test_open_held_parent_closes_fd_when_post_open_fstat_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    class Cancelled(BaseException):
        pass

    descriptor_root = Path("/dev/fd" if sys.platform == "darwin" else "/proc/self/fd")
    before = set(os.listdir(descriptor_root))
    monkeypatch.setattr(
        tts_publish.os,
        "fstat",
        lambda _fd: (_ for _ in ()).throw(Cancelled()),
    )
    with pytest.raises(Cancelled):
        tts_publish._open_held_parent(tmp_path / "voice.mp3")
    after = set(os.listdir(descriptor_root))
    assert len(after - before) <= 1


def test_absent_destination_race_preserves_concurrent_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    real_fsync = tts_publish._fsync_publication_file

    def race(fd):
        real_fsync(fd)
        destination.write_bytes(b"concurrent")

    monkeypatch.setattr(tts_publish, "_fsync_publication_file", race)
    with pytest.raises(TTSPublishError):
        _publish_one(tmp_path / "stage", destination)
    assert destination.read_bytes() == b"concurrent"


def test_existing_destination_race_retains_authorized_replace_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    destination = tmp_path / "voice.mp3"
    destination.write_bytes(b"authorized-old")
    real_fsync = tts_publish._fsync_publication_file

    def race(fd):
        real_fsync(fd)
        destination.write_bytes(b"concurrent")

    monkeypatch.setattr(tts_publish, "_fsync_publication_file", race)
    _publish_one(tmp_path / "stage", destination)
    assert destination.read_bytes() == VALID_MP3


def test_cancellation_scrubs_source_and_temp_then_propagates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish

    class Cancelled(BaseException):
        pass

    destination = tmp_path / "voice.mp3"
    monkeypatch.setattr(
        tts_publish,
        "_copy_sealed_to_publication",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(Cancelled()),
    )
    with pytest.raises(Cancelled):
        _publish_one(tmp_path / "stage", destination)
    assert not destination.exists()
    assert all(path.stat().st_size == 0 for path in tmp_path.glob(".hermes-tts-publish-*"))


def test_temp_scrub_failure_has_sticky_stop_precedence_and_source_cleanup_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_publish
    from tools.tts_staging import AnonymousAudioScrubError

    destination = tmp_path / "voice.mp3"
    source_closed: list[bool] = []
    real_stage_scrub = tts_publish.AnonymousAudioStage.scrub_and_close
    real_temp_scrub = tts_publish._scrub_and_close_fd

    def record_source_scrub(self):
        real_stage_scrub(self)
        source_closed.append(self._closed)

    monkeypatch.setattr(tts_publish.AnonymousAudioStage, "scrub_and_close", record_source_scrub)
    monkeypatch.setattr(
        tts_publish,
        "_rename_absent_for_host",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("private")),
    )
    def close_then_fail(fd):
        real_temp_scrub(fd)
        raise AnonymousAudioScrubError("private")

    monkeypatch.setattr(tts_publish, "_scrub_and_close_fd", close_then_fail)
    with pytest.raises(TTSTransactionStop, match="^tts_anonymous_scrub_failed$"):
        _publish_one(tmp_path / "stage", destination)
    assert source_closed and all(source_closed)


def test_unsupported_platform_fails_before_materialization(tmp_path: Path):
    before = set(tmp_path.iterdir())
    with pytest.raises(TTSPublishError, match="^tts_durable_publication_failed$"):
        _require_absent_primitive_for_platform("win32")
    assert set(tmp_path.iterdir()) == before


def test_published_audio_is_immutable_and_path_only(tmp_path: Path):
    published = _publish_one(tmp_path / "stage", tmp_path / "voice.mp3")
    assert published.path == tmp_path / "voice.mp3"
    with pytest.raises((AttributeError, TypeError)):
        published.path = tmp_path / "other.mp3"
    assert not any(hasattr(published, name) for name in ("fd", "permit", "cleanup"))


def test_twenty_publications_leave_no_descriptor_or_audio_residue(tmp_path: Path):
    descriptor_root = Path("/dev/fd" if sys.platform == "darwin" else "/proc/self/fd")
    before = set(os.listdir(descriptor_root))
    for index in range(20):
        destination = tmp_path / f"voice-{index}.mp3"
        _publish_one(tmp_path / f"stage-{index}", destination)
        destination.unlink()
    after = set(os.listdir(descriptor_root))
    assert len(after - before) <= 1
    assert not list(tmp_path.glob(".hermes-tts-publish-*"))
