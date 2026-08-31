"""Monotonic decision boundary for anonymously staged TTS audio.

The transaction owns persistence observation and every accepted sealed stage.
It either delivers bounded audio in memory, transfers sealed stages into an
opaque durable-publication permit, or destroys every held descriptor.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

from hermes_cli.persistence import (
    PersistenceObservation,
    PersistencePolicy,
    observe_persistence_transaction,
)
from tools.tts_staging import (
    MAX_ANONYMOUS_AUDIO_BYTES,
    AnonymousAudioStage,
    SealedAudio,
)


_TRANSACTION_ERROR = "tts_generation_failed"
_SCRUB_ERROR = "tts_anonymous_scrub_failed"


class TTSTransactionError(RuntimeError):
    """Fixed, path-free TTS decision failure."""


class TTSTransactionStop(TTSTransactionError):
    """High-severity stop because descriptor destruction was not proved."""


def _create_transaction_boundary():
    construction_identity = object()

    class EphemeralDelivery:
        """Bounded in-memory audio returned only by an ephemeral transaction."""

        __slots__ = ("__chunks", "__total_bytes")

        def __new__(cls, *args: object, **kwargs: object):
            raise TypeError("EphemeralDelivery is issued only by TTSTransaction")

        def __init_subclass__(cls, **kwargs: object) -> None:
            raise TypeError("EphemeralDelivery cannot be subclassed")

        def __setattr__(self, name: str, value: object) -> None:
            raise TypeError("EphemeralDelivery is immutable")

        @property
        def chunks(self) -> tuple[bytes, ...]:
            return object.__getattribute__(self, "_EphemeralDelivery__chunks")

        @property
        def total_bytes(self) -> int:
            return object.__getattribute__(
                self, "_EphemeralDelivery__total_bytes"
            )

        def __reduce__(self):
            raise TypeError("EphemeralDelivery cannot be reconstructed")

        def __reduce_ex__(self, protocol: int):
            raise TypeError("EphemeralDelivery cannot be reconstructed")

        def __copy__(self):
            raise TypeError("EphemeralDelivery cannot be copied")

        def __deepcopy__(self, memo: object):
            raise TypeError("EphemeralDelivery cannot be copied")

    class DurablePublicationPermit:
        """Opaque ownership transfer for a later trusted durable publisher."""

        __slots__ = ("__identity", "__observation", "__stages")

        def __new__(cls, *args: object, **kwargs: object):
            raise TypeError(
                "DurablePublicationPermit is issued only by TTSTransaction"
            )

        def __init_subclass__(cls, **kwargs: object) -> None:
            raise TypeError("DurablePublicationPermit cannot be subclassed")

        def __setattr__(self, name: str, value: object) -> None:
            raise TypeError("DurablePublicationPermit is immutable")

        def __reduce__(self):
            raise TypeError("DurablePublicationPermit cannot be reconstructed")

        def __reduce_ex__(self, protocol: int):
            raise TypeError("DurablePublicationPermit cannot be reconstructed")

        def __copy__(self):
            raise TypeError("DurablePublicationPermit cannot be copied")

        def __deepcopy__(self, memo: object):
            raise TypeError("DurablePublicationPermit cannot be copied")

    def issue_delivery(chunks: list[bytes]) -> EphemeralDelivery:
        delivery = object.__new__(EphemeralDelivery)
        immutable_chunks = tuple(chunks)
        object.__setattr__(
            delivery, "_EphemeralDelivery__chunks", immutable_chunks
        )
        object.__setattr__(
            delivery,
            "_EphemeralDelivery__total_bytes",
            sum(len(chunk) for chunk in immutable_chunks),
        )
        return delivery

    def issue_permit(
        identity: object,
        observation: PersistenceObservation,
        stages: list[tuple[AnonymousAudioStage, SealedAudio]],
    ) -> DurablePublicationPermit:
        permit = object.__new__(DurablePublicationPermit)
        object.__setattr__(
            permit,
            "_DurablePublicationPermit__identity",
            (construction_identity, identity),
        )
        object.__setattr__(
            permit,
            "_DurablePublicationPermit__observation",
            observation,
        )
        object.__setattr__(
            permit, "_DurablePublicationPermit__stages", tuple(stages)
        )
        return permit

    class TTSTransaction:
        """One lexical GENERATE/SEAL/DECIDE ownership boundary."""

        __slots__ = (
            "__aggregate_cap",
            "__entry_policy",
            "__identity",
            "__lock",
            "__observation",
            "__stages",
            "__state",
        )

        def __new__(cls, *args: object, **kwargs: object):
            raise TypeError("TTSTransaction must be opened with begin()")

        def __init_subclass__(cls, **kwargs: object) -> None:
            raise TypeError("TTSTransaction cannot be subclassed")

        @classmethod
        @contextmanager
        def begin(cls, aggregate_cap: int) -> Iterator["TTSTransaction"]:
            if (
                type(aggregate_cap) is not int
                or aggregate_cap <= 0
                or aggregate_cap > MAX_ANONYMOUS_AUDIO_BYTES
            ):
                raise TTSTransactionError(_TRANSACTION_ERROR)
            with observe_persistence_transaction() as observation:
                transaction = object.__new__(cls)
                object.__setattr__(
                    transaction, "_TTSTransaction__aggregate_cap", aggregate_cap
                )
                object.__setattr__(
                    transaction,
                    "_TTSTransaction__entry_policy",
                    observation.current_policy,
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__identity", object()
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__lock", RLock()
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__observation", observation
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__stages", []
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__state", "open"
                )
                try:
                    yield transaction
                except TTSTransactionStop as exc:
                    transaction.__abort_if_untransferred()
                    if str(exc) == _SCRUB_ERROR:
                        raise
                    raise TTSTransactionError(_TRANSACTION_ERROR) from None
                except Exception:
                    transaction.__abort_if_untransferred()
                    raise TTSTransactionError(_TRANSACTION_ERROR) from None
                finally:
                    transaction.__abort_if_untransferred()

        def add_sealed(
            self,
            stage: AnonymousAudioStage,
            sealed: SealedAudio,
        ) -> None:
            with self.__lock:
                if self.__state != "open" or not self.__valid_pair(stage, sealed):
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                if any(held_stage is stage for held_stage, _ in self.__stages):
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                self.__stages.append((stage, sealed))

        def decide(self) -> EphemeralDelivery | DurablePublicationPermit:
            with self.__lock:
                if self.__state != "open":
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                self.__state = "deciding"
                if not self.__stages:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                if (
                    self.__entry_policy is PersistencePolicy.DURABLE
                    and not self.__observation.ever_ephemeral
                    and self.__observation.current_policy
                    is PersistencePolicy.DURABLE
                ):
                    permit = issue_permit(
                        self.__identity,
                        self.__observation,
                        self.__stages,
                    )
                    self.__stages = []
                    self.__state = "transferred"
                    return permit
                if self.__entry_policy is PersistencePolicy.DURABLE:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR)

                chunks: list[bytes] = []
                allocated = 0
                try:
                    for stage, sealed in self.__stages:
                        size = sealed._size
                        if (
                            type(size) is not int
                            or size <= 0
                            or size > self.__aggregate_cap - allocated
                        ):
                            raise TTSTransactionError(_TRANSACTION_ERROR)
                        chunk = stage.read_bounded(sealed)
                        if len(chunk) != size:
                            raise TTSTransactionError(_TRANSACTION_ERROR)
                        chunks.append(chunk)
                        allocated += size
                    delivery = issue_delivery(chunks)
                except BaseException:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR) from None
                self.__scrub_all()
                return delivery

        @staticmethod
        def __valid_pair(stage: object, sealed: object) -> bool:
            return (
                type(stage) is AnonymousAudioStage
                and type(sealed) is SealedAudio
                and not stage._closed
                and stage._sealed
                and sealed._authority is stage._authority
                and sealed._fd == stage._fd
            )

        def __scrub_all(self) -> None:
            stages = self.__stages
            self.__stages = []
            failed = False
            for stage, _ in stages:
                try:
                    stage.scrub_and_close()
                except BaseException:
                    failed = True
            self.__state = "scrubbed"
            if failed:
                raise TTSTransactionStop(_SCRUB_ERROR) from None

        def __abort_if_untransferred(self) -> None:
            with self.__lock:
                if self.__state in ("transferred", "scrubbed"):
                    return
                self.__scrub_all()

    return TTSTransaction, EphemeralDelivery, DurablePublicationPermit


TTSTransaction, EphemeralDelivery, DurablePublicationPermit = (
    _create_transaction_boundary()
)
del _create_transaction_boundary


__all__ = [
    "DurablePublicationPermit",
    "EphemeralDelivery",
    "TTSTransaction",
    "TTSTransactionError",
    "TTSTransactionStop",
]
