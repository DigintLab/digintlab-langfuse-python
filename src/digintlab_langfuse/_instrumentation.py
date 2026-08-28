"""Process-wide OpenTelemetry instrumentation ownership."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import TracerProvider

from digintlab_langfuse.spans import scrub_auth_header, scrub_auth_header_async

logger = logging.getLogger(__name__)


class _ProcessInstrumentor(Protocol):
    @property
    def is_instrumented_by_opentelemetry(self) -> bool: ...

    def instrument(self, **kwargs: object) -> object: ...

    def uninstrument(self) -> object: ...


class _ProcessInstrumentationError(RuntimeError):
    """Raised when process instrumentation cannot be completely released."""


@dataclass(slots=True)
class _ProcessInstrumentationLease:
    instrumentor: _ProcessInstrumentor
    signature: tuple[str, ...]
    users: int
    owned_by_library: bool


_process_instrumentation_lock = RLock()
_process_instrumentation_leases: dict[str, _ProcessInstrumentationLease] = {}


def _acquire_process_instrumentation(
    *,
    name: str,
    instrumentor: _ProcessInstrumentor,
    signature: tuple[str, ...],
    install: Callable[[], None],
) -> None:
    with _process_instrumentation_lock:
        existing = _process_instrumentation_leases.get(name)
        if existing is not None:
            if existing.signature != signature:
                message = f"Process-wide {name} instrumentation is already active with different configuration."
                raise RuntimeError(message)
            if existing.users == 0 and not instrumentor.is_instrumented_by_opentelemetry:
                install()
                existing.instrumentor = instrumentor
                existing.owned_by_library = True
            existing.users += 1
            return

        already_instrumented = instrumentor.is_instrumented_by_opentelemetry
        if not already_instrumented:
            install()
        _process_instrumentation_leases[name] = _ProcessInstrumentationLease(
            instrumentor=instrumentor,
            signature=signature,
            users=1,
            owned_by_library=not already_instrumented,
        )


def _release_process_instrumentation(name: str) -> None:
    with _process_instrumentation_lock:
        lease = _process_instrumentation_leases[name]
        if lease.users > 1:
            lease.users -= 1
            return
        if lease.owned_by_library:
            lease.instrumentor.uninstrument()
        del _process_instrumentation_leases[name]


def _abandon_failed_release(name: str) -> None:
    with _process_instrumentation_lock:
        _process_instrumentation_leases[name].users -= 1


class ProcessInstrumentation:
    """Own optional HTTPX and botocore instrumentation leases."""

    def __init__(
        self,
        *,
        instrument_httpx: bool,
        instrument_botocore: bool,
        tracer_provider: TracerProvider | None,
        redact_http_auth_headers: bool,
    ) -> None:
        self._httpx_instrumentor = HTTPXClientInstrumentor() if instrument_httpx else None
        self._botocore_instrumentor = BotocoreInstrumentor() if instrument_botocore else None  # type: ignore[no-untyped-call]
        self._tracer_provider = tracer_provider
        self._redact_http_auth_headers = redact_http_auth_headers
        self._acquired_instrumentors: list[str] = []
        self._instrumented = False
        self._lock = RLock()

    def instrument(self) -> None:
        """Acquire and install the configured process instrumentors."""
        with self._lock:
            if self._instrumented:
                return
            instrumentor_kwargs: dict[str, TracerProvider] = {}
            if self._tracer_provider is not None:
                instrumentor_kwargs["tracer_provider"] = self._tracer_provider
            provider_signature = f"tracer-provider:{id(self._tracer_provider)}"
            try:
                self._instrument_botocore(instrumentor_kwargs, provider_signature)
                self._instrument_httpx(instrumentor_kwargs, provider_signature)
            except BaseException:
                self._rollback_acquired_instrumentors()
                raise
            self._instrumented = True

    def _instrument_botocore(self, kwargs: dict[str, TracerProvider], provider_signature: str) -> None:
        instrumentor = self._botocore_instrumentor
        if instrumentor is None:
            return
        _acquire_process_instrumentation(
            name="botocore",
            instrumentor=instrumentor,
            signature=(provider_signature,),
            install=lambda: instrumentor.instrument(**kwargs),
        )
        self._acquired_instrumentors.append("botocore")

    def _instrument_httpx(self, kwargs: dict[str, TracerProvider], provider_signature: str) -> None:
        instrumentor = self._httpx_instrumentor
        if instrumentor is None:
            return

        def install() -> None:
            if self._redact_http_auth_headers:
                instrumentor.instrument(
                    request_hook=scrub_auth_header,
                    async_request_hook=scrub_auth_header_async,
                    **kwargs,
                )
            else:
                instrumentor.instrument(**kwargs)

        _acquire_process_instrumentation(
            name="httpx",
            instrumentor=instrumentor,
            signature=(provider_signature, f"redact-auth:{self._redact_http_auth_headers}"),
            install=install,
        )
        self._acquired_instrumentors.append("httpx")

    def _rollback_acquired_instrumentors(self) -> None:
        for name in reversed(self._acquired_instrumentors):
            try:
                _release_process_instrumentation(name)
            except Exception:
                _abandon_failed_release(name)
                logger.exception("Failed to roll back %s process instrumentation", name)
        self._acquired_instrumentors.clear()

    def uninstrument(self) -> None:
        """Release the configured process instrumentors."""
        with self._lock:
            if not self._instrumented:
                return

            failures: list[Exception] = []
            failed_names: list[str] = []
            for name in reversed(self._acquired_instrumentors):
                try:
                    _release_process_instrumentation(name)
                except Exception as exc:
                    failures.append(exc)
                    failed_names.append(name)
                    logger.exception("Failed to release %s process instrumentation", name)
            self._acquired_instrumentors = list(reversed(failed_names))
            self._instrumented = bool(self._acquired_instrumentors)
            if failures:
                message = "Failed to release process instrumentation."
                raise _ProcessInstrumentationError(message) from failures[0]
