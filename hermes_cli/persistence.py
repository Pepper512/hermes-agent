"""Invocation-scoped transcript persistence policy for Hermes CLI runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterator


class PersistencePolicy(str, Enum):
    """The single security boundary controlling invocation persistence."""

    DURABLE = "durable"
    EPHEMERAL = "ephemeral"


_CURRENT_POLICY: ContextVar[PersistencePolicy] = ContextVar(
    "hermes_persistence_policy",
    default=PersistencePolicy.DURABLE,
)


@dataclass(frozen=True)
class _ObservationState:
    observation_id: object
    ever_ephemeral: bool


_OBSERVERS: ContextVar[tuple[_ObservationState, ...]] = ContextVar(
    "hermes_persistence_observers",
    default=(),
)


def _state_for(observation_id: object) -> _ObservationState:
    for state in _OBSERVERS.get():
        if state.observation_id is observation_id:
            return state
    raise RuntimeError("inactive persistence observation")


class PersistenceObservation:
    """Read-only view of one live persistence transaction."""

    __slots__ = ("__observation_id",)

    def __init__(self, observation_id: object) -> None:
        self.__observation_id = observation_id

    @property
    def current_policy(self) -> PersistencePolicy:
        _state_for(self.__observation_id)
        return current_persistence_policy()

    @property
    def ever_ephemeral(self) -> bool:
        return _state_for(self.__observation_id).ever_ephemeral


def _record_transition(policy: PersistencePolicy) -> None:
    if policy is not PersistencePolicy.EPHEMERAL:
        return
    _OBSERVERS.set(
        tuple(
            replace(state, ever_ephemeral=True)
            if not state.ever_ephemeral
            else state
            for state in _OBSERVERS.get()
        )
    )


def coerce_persistence_policy(value: object) -> PersistencePolicy:
    if isinstance(value, PersistencePolicy):
        return value
    try:
        return PersistencePolicy(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid persistence policy") from exc


def current_persistence_policy() -> PersistencePolicy:
    return _CURRENT_POLICY.get()


def activate_invocation_persistence_policy(policy: object) -> object:
    """Activate a process-lifetime invocation policy during CLI bootstrap.

    The CLI uses this before importing its startup graph.  The returned token
    is intentionally opaque; one-shot processes terminate after teardown, so
    callers must not reset it mid-invocation.
    """
    normalized = coerce_persistence_policy(policy)
    _record_transition(normalized)
    return _CURRENT_POLICY.set(normalized)


def persistence_disabled(owner: object | None = None) -> bool:
    """Return whether persistence is forbidden, failing closed.

    ``persistence_policy`` is authoritative.  The legacy boolean may only
    further restrict durable helper agents; it can never re-enable an
    explicitly ephemeral agent.
    """
    ambient_policy = current_persistence_policy()
    if ambient_policy is PersistencePolicy.EPHEMERAL:
        return True

    state: dict[str, object] = {}
    if owner is not None:
        try:
            state = vars(owner)
        except TypeError:
            state = {}
    value = state.get("persistence_policy", ambient_policy)
    try:
        policy = coerce_persistence_policy(value)
    except ValueError:
        return True
    return policy is PersistencePolicy.EPHEMERAL or bool(
        state.get("_persist_disabled", False) if owner is not None else False
    )


@contextmanager
def bind_persistence_policy(policy: object) -> Iterator[PersistencePolicy]:
    """Bind one policy for construction, execution, and teardown."""
    normalized = coerce_persistence_policy(policy)
    prior = current_persistence_policy()
    _record_transition(normalized)
    token = _CURRENT_POLICY.set(normalized)
    try:
        yield normalized
    finally:
        _record_transition(prior)
        _CURRENT_POLICY.reset(token)


@contextmanager
def observe_persistence_transaction() -> Iterator[PersistenceObservation]:
    """Observe policy transitions for one lexical transaction."""
    observation_id = object()
    state = _ObservationState(
        observation_id=observation_id,
        ever_ephemeral=current_persistence_policy() is PersistencePolicy.EPHEMERAL,
    )
    _OBSERVERS.set((*_OBSERVERS.get(), state))
    observation = PersistenceObservation(observation_id)
    try:
        yield observation
    finally:
        _OBSERVERS.set(
            tuple(
                current
                for current in _OBSERVERS.get()
                if current.observation_id is not observation_id
            )
        )


def validate_invocation_policy(args: object) -> PersistencePolicy:
    """Validate the closed public ephemeral surface without side effects."""
    policy = coerce_persistence_policy(
        getattr(args, "persistence_policy", PersistencePolicy.DURABLE)
    )
    if policy is PersistencePolicy.DURABLE:
        return policy

    command = getattr(args, "command", None)
    resumed = bool(
        getattr(args, "resume", None)
        or getattr(args, "continue_last", None)
        or getattr(args, "create_if_missing", False)
    )
    if resumed:
        raise ValueError("--ephemeral-session cannot resume or select a session")
    if getattr(args, "usage_file", None):
        raise ValueError("--ephemeral-session cannot be combined with --usage-file")

    if command == "chat":
        query_sources = sum(
            bool(getattr(args, name, None)) for name in ("query", "query_file")
        )
        if not getattr(args, "oneshot_exit", False):
            raise ValueError("--ephemeral-session requires chat --oneshot")
        if query_sources != 1 or getattr(args, "image", None):
            raise ValueError("--ephemeral-session requires exactly one text query source")
        if getattr(args, "tui", False):
            raise ValueError("--ephemeral-session cannot launch an interactive interface")
        return policy

    if command is not None or not getattr(args, "oneshot", None):
        raise ValueError("--ephemeral-session is valid only with one-shot chat")
    return policy


__all__ = [
    "PersistencePolicy",
    "PersistenceObservation",
    "activate_invocation_persistence_policy",
    "bind_persistence_policy",
    "coerce_persistence_policy",
    "current_persistence_policy",
    "observe_persistence_transaction",
    "persistence_disabled",
    "validate_invocation_policy",
]
