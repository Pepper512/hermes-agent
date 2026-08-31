from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.persistence import PersistencePolicy, bind_persistence_policy
from tools.tts_staging import _create_anonymous_audio_stage_for_test
from tools.tts_transaction import (
    DurablePublicationPermit,
    EphemeralDelivery,
    TTSTransaction,
    TTSTransactionError,
    TTSTransactionStop,
)


VALID_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00private-audio"


def _sealed_stage(parent: Path, payload: bytes = VALID_MP3):
    parent.mkdir(parents=True, exist_ok=True)
    stage = _create_anonymous_audio_stage_for_test("mp3", 4096, parent)
    os.write(int(Path(stage.sink.path).name), payload)
    return stage, stage.seal(stage.sink.path)


def _assert_scrubbed(stage) -> None:
    assert stage._closed is True
    assert stage._fd == -1


@pytest.mark.parametrize(
    ("entry", "transitions", "expected"),
    [
        (PersistencePolicy.EPHEMERAL, (), EphemeralDelivery),
        (
            PersistencePolicy.EPHEMERAL,
            (PersistencePolicy.DURABLE,),
            EphemeralDelivery,
        ),
        (PersistencePolicy.DURABLE, (), DurablePublicationPermit),
        (PersistencePolicy.DURABLE, (PersistencePolicy.EPHEMERAL,), TTSTransactionError),
        (
            PersistencePolicy.DURABLE,
            (PersistencePolicy.EPHEMERAL, PersistencePolicy.DURABLE),
            TTSTransactionError,
        ),
    ],
)
def test_decision_is_monotonic(tmp_path: Path, entry, transitions, expected):
    stage, sealed = _sealed_stage(tmp_path)
    with bind_persistence_policy(entry):
        with TTSTransaction.begin(1024) as transaction:
            transaction.add_sealed(stage, sealed)
            for policy in transitions:
                with bind_persistence_policy(policy):
                    pass
            if expected is TTSTransactionError:
                with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
                    transaction.decide()
            else:
                assert type(transaction.decide()) is expected
    if expected is not DurablePublicationPermit:
        _assert_scrubbed(stage)
    else:
        stage.scrub_and_close()


def test_entry_ephemeral_returns_bounded_path_free_memory_and_scrubs(tmp_path: Path):
    stage, sealed = _sealed_stage(tmp_path)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(len(VALID_MP3)) as transaction:
            transaction.add_sealed(stage, sealed)
            delivery = transaction.decide()

    assert type(delivery) is EphemeralDelivery
    assert delivery.chunks == (VALID_MP3,)
    assert delivery.total_bytes == len(VALID_MP3)
    assert not any(
        hasattr(delivery, name)
        for name in ("path", "destination", "cleanup", "token", "descriptor", "fd")
    )
    with pytest.raises(TypeError):
        vars(delivery)
    _assert_scrubbed(stage)


def test_ephemeral_delivery_reads_only_the_sealed_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage, sealed = _sealed_stage(tmp_path)
    sentinel = tmp_path / "caller-destination.mp3"
    sentinel.write_bytes(b"caller-owned")

    def forbidden_path_read(*args, **kwargs):
        raise AssertionError("decision must not read a pathname")

    monkeypatch.setattr(Path, "read_bytes", forbidden_path_read)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            delivery = transaction.decide()
    assert delivery.chunks == (VALID_MP3,)
    assert sentinel.stat().st_size == len(b"caller-owned")


def test_forged_old_global_cannot_change_stage_or_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from tools import tts_transaction

    stage, sealed = _sealed_stage(tmp_path / "stage")
    sentinel = tmp_path / "caller-owned"
    sentinel.write_bytes(b"unchanged")
    forged = SimpleNamespace(
        stage=sentinel,
        cleanup=lambda: sentinel.unlink(),
        destination=sentinel,
    )
    monkeypatch.setattr(
        tts_transaction, "_EPHEMERAL_TTS_STATE", forged, raising=False
    )
    monkeypatch.setattr(
        tts_transaction,
        "_cleanup_ephemeral_tts_state",
        forged.cleanup,
        raising=False,
    )

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            delivery = transaction.decide()

    assert delivery.chunks == (VALID_MP3,)
    assert sentinel.read_bytes() == b"unchanged"
    assert not any(tmp_path.joinpath("stage").glob("hermes-tts-*"))


def test_ephemeral_multiple_chunks_enforces_aggregate_cap_before_second_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first, first_sealed = _sealed_stage(tmp_path / "first")
    second, second_sealed = _sealed_stage(tmp_path / "second")
    called = False
    real_read = type(second).read_bounded

    def record_read(self, sealed):
        nonlocal called
        if self is second:
            called = True
        return real_read(self, sealed)

    monkeypatch.setattr(type(second), "read_bounded", record_read)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(len(VALID_MP3) * 2 - 1) as transaction:
            transaction.add_sealed(first, first_sealed)
            transaction.add_sealed(second, second_sealed)
            with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
                transaction.decide()

    assert called is False
    _assert_scrubbed(first)
    _assert_scrubbed(second)


def test_late_rebind_failure_is_fixed_and_scrubs_all_stages(tmp_path: Path):
    stages = [_sealed_stage(tmp_path / str(index)) for index in range(2)]
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as transaction:
            for stage, sealed in stages:
                transaction.add_sealed(stage, sealed)
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                pass
            with pytest.raises(TTSTransactionError) as exc_info:
                transaction.decide()
    assert str(exc_info.value) == "tts_generation_failed"
    assert "private" not in str(exc_info.value)
    for stage, _ in stages:
        _assert_scrubbed(stage)


@pytest.mark.parametrize("failure", [RuntimeError("provider secret"), TimeoutError("private timeout")])
def test_context_error_scrubs_all_chunks_and_returns_fixed_failure(
    tmp_path: Path, failure: BaseException
):
    stage, sealed = _sealed_stage(tmp_path)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(TTSTransactionError) as exc_info:
            with TTSTransaction.begin(4096) as transaction:
                transaction.add_sealed(stage, sealed)
                raise failure
    assert str(exc_info.value) == "tts_generation_failed"
    assert "private" not in str(exc_info.value)
    assert "secret" not in str(exc_info.value)
    _assert_scrubbed(stage)


def test_context_cancel_scrubs_all_chunks(tmp_path: Path):
    class Cancelled(BaseException):
        pass

    stage, sealed = _sealed_stage(tmp_path)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(Cancelled):
            with TTSTransaction.begin(4096) as transaction:
                transaction.add_sealed(stage, sealed)
                raise Cancelled
    _assert_scrubbed(stage)


def test_scrub_failure_has_high_severity_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage, sealed = _sealed_stage(tmp_path)

    real_scrub = type(stage).scrub_and_close

    def fail_scrub(self) -> None:
        if self is stage:
            raise RuntimeError("private cleanup detail")
        real_scrub(self)

    monkeypatch.setattr(type(stage), "scrub_and_close", fail_scrub)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(TTSTransactionStop) as exc_info:
            with TTSTransaction.begin(4096) as transaction:
                transaction.add_sealed(stage, sealed)
                raise RuntimeError("provider secret")
    assert str(exc_info.value) == "tts_anonymous_scrub_failed"
    assert "private" not in str(exc_info.value)


def test_decide_scrub_failure_is_not_downgraded_by_context_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage, sealed = _sealed_stage(tmp_path)

    def fail_scrub(self) -> None:
        if self is stage:
            raise RuntimeError("private cleanup detail")
        raise AssertionError("unexpected stage")

    monkeypatch.setattr(type(stage), "scrub_and_close", fail_scrub)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with pytest.raises(TTSTransactionStop) as exc_info:
            with TTSTransaction.begin(4096) as transaction:
                transaction.add_sealed(stage, sealed)
                transaction.decide()
    assert str(exc_info.value) == "tts_anonymous_scrub_failed"


def test_durable_permit_is_unforgeable_path_free_and_owns_stages(tmp_path: Path):
    stage, sealed = _sealed_stage(tmp_path)
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            permit = transaction.decide()

    assert type(permit) is DurablePublicationPermit
    assert not any(
        hasattr(permit, name)
        for name in ("path", "destination", "cleanup", "delete", "token", "descriptor", "fd")
    )
    with pytest.raises(TypeError):
        DurablePublicationPermit("/tmp/caller")
    with pytest.raises(TypeError):
        vars(permit)
    assert stage._closed is False
    stage.scrub_and_close()


def test_decide_without_a_sealed_stage_fails_closed():
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(4096) as transaction:
            with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
                transaction.decide()


def test_parallel_decide_transfers_once(tmp_path: Path):
    stage, sealed = _sealed_stage(tmp_path)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            with ThreadPoolExecutor(max_workers=2) as pool:
                outcomes = list(
                    pool.map(
                        lambda _index: _categorize_decide(transaction),
                        range(2),
                    )
                )
    assert sorted(outcomes) == ["delivery", "failure"]
    _assert_scrubbed(stage)


def _categorize_decide(transaction: TTSTransaction) -> str:
    try:
        result = transaction.decide()
    except TTSTransactionError:
        return "failure"
    assert type(result) is EphemeralDelivery
    return "delivery"


def test_reentrant_decide_fails_without_interrupting_outer_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage, sealed = _sealed_stage(tmp_path)
    original_read = type(stage).read_bounded
    nested: list[str] = []

    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)

            def reentrant_read(self, proof):
                with pytest.raises(
                    TTSTransactionError, match="^tts_generation_failed$"
                ):
                    transaction.decide()
                nested.append("rejected")
                return original_read(self, proof)

            monkeypatch.setattr(type(stage), "read_bounded", reentrant_read)
            result = transaction.decide()

    assert type(result) is EphemeralDelivery
    assert nested == ["rejected"]
    _assert_scrubbed(stage)


def test_second_reentrant_and_add_after_decision_fail_closed_without_double_scrub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    stage, sealed = _sealed_stage(tmp_path)
    calls = 0
    real_scrub = type(stage).scrub_and_close

    def record_scrub(self) -> None:
        nonlocal calls
        if self is stage:
            calls += 1
        real_scrub(self)

    monkeypatch.setattr(type(stage), "scrub_and_close", record_scrub)
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with TTSTransaction.begin(4096) as transaction:
            transaction.add_sealed(stage, sealed)
            transaction.decide()
            with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
                transaction.decide()
            with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
                transaction.add_sealed(stage, sealed)
    assert calls == 1


@pytest.mark.parametrize("cap", [None, True, False, 0, -1, 1.5, "1024", 25 * 1024 * 1024 + 1])
def test_invalid_aggregate_cap_fails_before_observation_or_staging(cap):
    with pytest.raises(TTSTransactionError, match="^tts_generation_failed$"):
        with TTSTransaction.begin(cap):
            raise AssertionError("unreachable")
