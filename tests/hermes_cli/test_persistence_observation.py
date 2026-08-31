"""Behavioral tests for transaction-local persistence observation."""

from __future__ import annotations

import asyncio
from contextlib import ExitStack
from contextvars import copy_context
import threading

import pytest

from hermes_cli.persistence import (
    PersistencePolicy,
    activate_invocation_persistence_policy,
    bind_persistence_policy,
    current_persistence_policy,
    observe_persistence_transaction,
)


def test_observation_latches_durable_ephemeral_durable():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as observation:
            with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                assert observation.current_policy is PersistencePolicy.EPHEMERAL
            assert observation.current_policy is PersistencePolicy.DURABLE
            assert observation.ever_ephemeral is True


def test_activate_marks_every_live_observation():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as outer:
            with observe_persistence_transaction() as inner:
                activate_invocation_persistence_policy(PersistencePolicy.EPHEMERAL)
                assert outer.ever_ephemeral is True
                assert inner.ever_ephemeral is True


def test_entry_ephemeral_initializes_latch_true():
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        with observe_persistence_transaction() as observation:
            assert observation.current_policy is PersistencePolicy.EPHEMERAL
            assert observation.ever_ephemeral is True


def test_durable_only_observation_stays_false():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as observation:
            assert observation.current_policy is PersistencePolicy.DURABLE
            assert observation.ever_ephemeral is False


def test_nested_observation_exit_does_not_unregister_outer():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as outer:
            with observe_persistence_transaction():
                with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                    pass
            assert outer.current_policy is PersistencePolicy.DURABLE
            assert outer.ever_ephemeral is True


def test_bind_restore_to_ephemeral_is_observed():
    with ExitStack() as stack:
        stack.enter_context(bind_persistence_policy(PersistencePolicy.EPHEMERAL))
        durable_binding = bind_persistence_policy(PersistencePolicy.DURABLE)
        durable_binding.__enter__()
        with observe_persistence_transaction() as observation:
            assert observation.ever_ephemeral is False
            durable_binding.__exit__(None, None, None)
            assert observation.current_policy is PersistencePolicy.EPHEMERAL
            assert observation.ever_ephemeral is True


def _exercise_durable_ephemeral_durable_transition() -> None:
    assert current_persistence_policy() is PersistencePolicy.DURABLE
    with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
        assert current_persistence_policy() is PersistencePolicy.EPHEMERAL
    assert current_persistence_policy() is PersistencePolicy.DURABLE


def test_context_copy_ephemeral_transition_latches_parent():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as observation:
            copied = copy_context()
            copied.run(_exercise_durable_ephemeral_durable_transition)

            assert observation.current_policy is PersistencePolicy.DURABLE
            assert observation.ever_ephemeral is True


def test_asyncio_task_ephemeral_transition_latches_parent():
    async def run_in_inherited_task() -> None:
        task = asyncio.create_task(_async_durable_ephemeral_durable_transition())
        await task

    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as observation:
            asyncio.run(run_in_inherited_task())

            assert observation.current_policy is PersistencePolicy.DURABLE
            assert observation.ever_ephemeral is True


async def _async_durable_ephemeral_durable_transition() -> None:
    _exercise_durable_ephemeral_durable_transition()


def test_captured_thread_ephemeral_transition_latches_parent():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as observation:
            copied = copy_context()
            finished = threading.Event()

            def transition_and_signal() -> None:
                _exercise_durable_ephemeral_durable_transition()
                finished.set()

            worker = threading.Thread(target=copied.run, args=(transition_and_signal,))
            worker.start()
            worker.join(timeout=5)

            assert worker.is_alive() is False
            assert finished.is_set() is True
            assert observation.current_policy is PersistencePolicy.DURABLE
            assert observation.ever_ephemeral is True


def test_observation_has_no_mutating_public_method():
    with observe_persistence_transaction() as observation:
        public_callables = {
            name
            for name in dir(observation)
            if not name.startswith("_") and callable(getattr(observation, name))
        }
        assert public_callables == set()
        with pytest.raises(AttributeError):
            observation.ever_ephemeral = False
        with pytest.raises(AttributeError):
            observation.current_policy = PersistencePolicy.DURABLE


def test_observation_fails_closed_after_its_registry_entry_exits():
    with observe_persistence_transaction() as observation:
        assert observation.current_policy is PersistencePolicy.DURABLE

    with pytest.raises(RuntimeError, match="inactive persistence observation"):
        _ = observation.current_policy
    with pytest.raises(RuntimeError, match="inactive persistence observation"):
        _ = observation.ever_ephemeral
