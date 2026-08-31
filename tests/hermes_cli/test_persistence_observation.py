"""Behavioral tests for transaction-local persistence observation."""

from __future__ import annotations

from contextlib import ExitStack
from contextvars import copy_context

import pytest

from hermes_cli.persistence import (
    PersistencePolicy,
    activate_invocation_persistence_policy,
    bind_persistence_policy,
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


def test_context_copy_does_not_share_mutable_latch():
    with bind_persistence_policy(PersistencePolicy.DURABLE):
        with observe_persistence_transaction() as observation:
            copied = copy_context()

            def enter_ephemeral_policy() -> None:
                with bind_persistence_policy(PersistencePolicy.EPHEMERAL):
                    assert observation.ever_ephemeral is True

            copied.run(enter_ephemeral_policy)
            assert observation.current_policy is PersistencePolicy.DURABLE
            assert observation.ever_ephemeral is False


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
