"""Invocation-scoped transcript persistence policy for Hermes CLI runs."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
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


def coerce_persistence_policy(value: object) -> PersistencePolicy:
    if isinstance(value, PersistencePolicy):
        return value
    try:
        return PersistencePolicy(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid persistence policy") from exc


def current_persistence_policy() -> PersistencePolicy:
    return _CURRENT_POLICY.get()


@contextmanager
def bind_persistence_policy(policy: object) -> Iterator[PersistencePolicy]:
    """Bind one policy for construction, execution, and teardown."""
    normalized = coerce_persistence_policy(policy)
    token = _CURRENT_POLICY.set(normalized)
    try:
        yield normalized
    finally:
        _CURRENT_POLICY.reset(token)


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
    "bind_persistence_policy",
    "coerce_persistence_policy",
    "current_persistence_policy",
    "validate_invocation_policy",
]
