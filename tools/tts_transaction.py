"""Monotonic decision boundary for anonymously staged TTS audio.

The transaction owns persistence observation and every accepted sealed stage.
It either delivers bounded audio in memory, transfers sealed stages into an
opaque durable-publication permit, or destroys every held descriptor.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
from typing import Callable, Iterator

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
    boundary_lock = RLock()
    claimed_stages: dict[object, object] = {}
    issued_permits: dict[object, object] = {}
    active_transactions: ContextVar[tuple[object, ...]] = ContextVar(
        "hermes_tts_active_transactions",
        default=(),
    )

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

        __slots__ = ()

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

        def _consume_for_publication(
            self,
            consumer: Callable[
                [tuple[tuple[AnonymousAudioStage, SealedAudio], ...], PersistenceObservation],
                object,
            ],
        ) -> object:
            return consume_permit(self, consumer)

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

    def issue_permit(transaction: object) -> DurablePublicationPermit:
        permit = object.__new__(DurablePublicationPermit)
        with boundary_lock:
            issued_permits[permit] = transaction
        return permit

    def unregister_permit(permit: object, transaction: object) -> None:
        if permit is None:
            return
        with boundary_lock:
            if issued_permits.get(permit) is transaction:
                del issued_permits[permit]

    def consume_permit(
        permit: object,
        consumer: Callable[
            [tuple[tuple[AnonymousAudioStage, SealedAudio], ...], PersistenceObservation],
            object,
        ],
    ) -> object:
        if type(permit) is not DurablePublicationPermit or not callable(consumer):
            raise TTSTransactionError(_TRANSACTION_ERROR)
        with boundary_lock:
            transaction = issued_permits.get(permit)
        active = active_transactions.get()
        if transaction is None or not active or active[-1] is not transaction:
            raise TTSTransactionError(_TRANSACTION_ERROR)
        return transaction._TTSTransaction__consume_permit(permit, consumer)

    class TTSTransaction:
        """One lexical GENERATE/SEAL/DECIDE ownership boundary."""

        __slots__ = (
            "__aggregate_cap",
            "__entry_policy",
            "__identity",
            "__lock",
            "__observation",
            "__permit",
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
                    transaction, "_TTSTransaction__permit", None
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__stages", []
                )
                object.__setattr__(
                    transaction, "_TTSTransaction__state", "open"
                )
                active_token = active_transactions.set(
                    (*active_transactions.get(), transaction)
                )
                try:
                    try:
                        yield transaction
                    except TTSTransactionStop as exc:
                        transaction.__abort_if_unconsumed()
                        if str(exc) == _SCRUB_ERROR:
                            raise
                        raise TTSTransactionError(_TRANSACTION_ERROR) from None
                    except Exception:
                        transaction.__abort_if_unconsumed()
                        raise TTSTransactionError(_TRANSACTION_ERROR) from None
                    finally:
                        transaction.__abort_if_unconsumed()
                finally:
                    active_transactions.reset(active_token)

        def add_sealed(
            self,
            stage: AnonymousAudioStage,
            sealed: SealedAudio,
        ) -> None:
            with self.__lock:
                if self.__state != "open":
                    self.__fail_add(stage)
                if not self.__valid_pair(stage, sealed):
                    self.__fail_add(stage)
                with boundary_lock:
                    stage_key = stage._authority
                    owner = claimed_stages.get(stage_key)
                    if owner is not None:
                        self.__fail_add(stage)
                    claimed_stages[stage_key] = self.__identity
                try:
                    self.__stages.append((stage, sealed))
                except BaseException:
                    self.__fail_add(stage)

        def decide(self) -> EphemeralDelivery | DurablePublicationPermit:
            with self.__lock:
                if self.__state != "open":
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                self.__state = "deciding"
                if not self.__stages:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                try:
                    declared_total = self.__validate_all_declared()
                except BaseException:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR) from None
                if (
                    self.__entry_policy is PersistencePolicy.DURABLE
                    and (
                        self.__observation.ever_ephemeral
                        or self.__observation.current_policy
                        is not PersistencePolicy.DURABLE
                    )
                ):
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR)

                try:
                    chunks = self.__revalidate_all(
                        retain=self.__entry_policy is PersistencePolicy.EPHEMERAL,
                        declared_total=declared_total,
                    )
                except BaseException:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR) from None

                if self.__entry_policy is PersistencePolicy.DURABLE:
                    permit = issue_permit(self)
                    self.__permit = permit
                    self.__state = "permitted"
                    return permit

                delivery = issue_delivery(chunks)
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
                and sealed._output_format == stage.sink.output_format
            )

        def __validate_all_declared(self) -> int:
            total = 0
            for stage, sealed in self.__stages:
                if not self.__valid_pair(stage, sealed):
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                size = sealed._size
                if (
                    type(size) is not int
                    or size <= 0
                    or size > self.__aggregate_cap - total
                ):
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                total += size
            return total

        def __revalidate_all(
            self,
            *,
            retain: bool,
            declared_total: int,
        ) -> list[bytes]:
            chunks: list[bytes] = []
            actual_total = 0
            for stage, sealed in self.__stages:
                chunk = stage.read_bounded(sealed)
                size = sealed._size
                if len(chunk) != size or len(chunk) > self.__aggregate_cap - actual_total:
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                actual_total += len(chunk)
                if retain:
                    chunks.append(chunk)
            if actual_total != declared_total:
                raise TTSTransactionError(_TRANSACTION_ERROR)
            return chunks

        def __consume_permit(
            self,
            permit: object,
            consumer: Callable[
                [tuple[tuple[AnonymousAudioStage, SealedAudio], ...], PersistenceObservation],
                object,
            ],
        ) -> object:
            with self.__lock:
                with boundary_lock:
                    authentic = issued_permits.get(permit) is self
                if (
                    not authentic
                    or self.__state != "permitted"
                    or self.__permit is not permit
                ):
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                if (
                    self.__observation.ever_ephemeral
                    or self.__observation.current_policy
                    is not PersistencePolicy.DURABLE
                ):
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                self.__state = "consuming"
                try:
                    declared_total = self.__validate_all_declared()
                    self.__revalidate_all(
                        retain=False,
                        declared_total=declared_total,
                    )
                    result = consumer(tuple(self.__stages), self.__observation)
                except TTSTransactionStop:
                    self.__scrub_all()
                    raise
                except Exception:
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR) from None
                except BaseException:
                    self.__scrub_all()
                    raise
                if any(not stage._closed for stage, _ in self.__stages):
                    self.__scrub_all()
                    raise TTSTransactionError(_TRANSACTION_ERROR)
                self.__release_closed_claims()
                unregister_permit(self.__permit, self)
                self.__permit = None
                self.__stages = []
                self.__state = "consumed"
                return result

        def __reject_presented(self, stage: object) -> None:
            if type(stage) is not AnonymousAudioStage or stage._closed:
                return
            with boundary_lock:
                stage_key = stage._authority
                owner = claimed_stages.get(stage_key)
                if owner is not None and owner is not self.__identity:
                    return
                claimed_stages[stage_key] = self.__identity
            try:
                stage.scrub_and_close()
            except BaseException:
                if stage._closed:
                    self.__release_claim(stage)
                raise TTSTransactionStop(_SCRUB_ERROR) from None
            self.__release_claim(stage)

        def __fail_add(self, stage: object) -> None:
            failed_scrub = False
            try:
                self.__reject_presented(stage)
            except TTSTransactionStop:
                failed_scrub = True
            try:
                self.__scrub_all()
            except TTSTransactionStop:
                failed_scrub = True
            if failed_scrub:
                raise TTSTransactionStop(_SCRUB_ERROR) from None
            raise TTSTransactionError(_TRANSACTION_ERROR)

        def __release_claim(self, stage: AnonymousAudioStage) -> None:
            with boundary_lock:
                stage_key = stage._authority
                if claimed_stages.get(stage_key) is self.__identity:
                    del claimed_stages[stage_key]

        def __release_closed_claims(self) -> None:
            for stage, _ in self.__stages:
                if stage._closed:
                    self.__release_claim(stage)

        def __scrub_all(self) -> None:
            unregister_permit(self.__permit, self)
            self.__permit = None
            failed = False
            remaining: list[tuple[AnonymousAudioStage, SealedAudio]] = []
            for stage, sealed in self.__stages:
                try:
                    stage.scrub_and_close()
                except BaseException:
                    failed = True
                if stage._closed:
                    self.__release_claim(stage)
                else:
                    remaining.append((stage, sealed))
            self.__stages = remaining
            self.__state = "scrub_failed" if remaining else "scrubbed"
            if failed or remaining:
                raise TTSTransactionStop(_SCRUB_ERROR) from None

        def __abort_if_unconsumed(self) -> None:
            with self.__lock:
                if self.__state in ("consumed", "scrubbed"):
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
