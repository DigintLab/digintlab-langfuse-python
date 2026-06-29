"""Langfuse SDK client construction and instrumentation helpers.

The module stores explicit Langfuse settings in SDK environment variables, creates `Langfuse` clients, wraps the LangChain callback handler, and
starts or stops optional `httpx` and `botocore` OpenTelemetry instrumentors.
"""

import asyncio
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TypedDict

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langfuse.types import MaskFunction, MaskOtelSpansFunction, TraceContext
from opentelemetry.instrumentation.botocore import BotocoreInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from digintlab_langfuse.spans import ObservationKind, scrub_auth_header
from digintlab_langfuse.types import JsonValue, to_json_value

LANGFUSE_DEFAULT_BASE_URL = "https://cloud.langfuse.com"


class LangfuseInitKwargs(TypedDict):
    """Keyword arguments forwarded to the Langfuse SDK client."""

    public_key: str
    secret_key: str
    base_url: str
    timeout: int | None
    debug: bool
    environment: str | None
    release: str | None
    flush_at: int | None
    flush_interval: float | None
    sample_rate: float | None
    mask: MaskFunction | None
    mask_otel_spans: MaskOtelSpansFunction | None
    httpx_client: httpx.Client | None
    should_export_span: Callable[[ReadableSpan], bool] | None
    tracer_provider: TracerProvider | None
    span_exporter: SpanExporter | None


@dataclass(frozen=True, slots=True)
class LangfuseClientConfig:
    """Configuration required to initialize the Langfuse SDK client."""

    public_key: str
    secret_key: str
    base_url: str = LANGFUSE_DEFAULT_BASE_URL
    timeout: int | None = None
    environment: str | None = None
    release: str | None = None
    sample_rate: float | None = None
    flush_at: int | None = None
    flush_interval: float | None = None
    debug: bool = False
    mask: MaskFunction | None = None
    mask_otel_spans: MaskOtelSpansFunction | None = None
    httpx_client: httpx.Client | None = None
    should_export_span: Callable[[ReadableSpan], bool] | None = None
    tracer_provider: TracerProvider | None = None
    span_exporter: SpanExporter | None = None
    redact_http_auth_headers: bool = True


def configure_langfuse_environment(config: LangfuseClientConfig) -> None:
    """Expose explicit Langfuse config under SDK-compatible env vars."""
    os.environ["LANGFUSE_PUBLIC_KEY"] = config.public_key
    os.environ["LANGFUSE_SECRET_KEY"] = config.secret_key
    os.environ["LANGFUSE_BASE_URL"] = config.base_url


def init_langfuse_client(config: LangfuseClientConfig) -> Langfuse:
    """Initialize and return the authenticated Langfuse SDK singleton."""
    configure_langfuse_environment(config)
    kwargs: LangfuseInitKwargs = {
        "public_key": config.public_key,
        "secret_key": config.secret_key,
        "base_url": config.base_url,
        "timeout": config.timeout,
        "debug": config.debug,
        "environment": config.environment,
        "release": config.release,
        "flush_at": config.flush_at,
        "flush_interval": config.flush_interval,
        "sample_rate": config.sample_rate,
        "mask": config.mask,
        "mask_otel_spans": config.mask_otel_spans,
        "httpx_client": config.httpx_client,
        "should_export_span": config.should_export_span,
        "tracer_provider": config.tracer_provider,
        "span_exporter": config.span_exporter,
    }
    return Langfuse(**kwargs)


async def shutdown_langfuse_client(lf: Langfuse) -> None:
    """Shutdown Langfuse telemetry without blocking the event loop indefinitely."""
    await asyncio.wait_for(asyncio.to_thread(lf.shutdown), timeout=30.0)


class LangfuseTelemetry:
    """Own one Langfuse client plus optional process-wide instrumentation.

    Instances keep the SDK client, a LangChain callback handler, optional instrumentors, and the `_instrumented` flag used by `instrument()` and
    `uninstrument()`.
    """

    def __init__(
        self,
        client: Langfuse,
        *,
        public_key: str,
        instrument_httpx: bool,
        instrument_botocore: bool,
        tracer_provider: TracerProvider | None = None,
        redact_http_auth_headers: bool = True,
    ) -> None:
        self.client = client
        self._langchain_handler = LangfuseCallbackHandler(public_key=public_key)
        self._httpx_instrumentor = HTTPXClientInstrumentor() if instrument_httpx else None
        self._botocore_instrumentor = BotocoreInstrumentor() if instrument_botocore else None  # type: ignore[no-untyped-call]
        self._tracer_provider = tracer_provider
        self._redact_http_auth_headers = redact_http_auth_headers
        self._instrumented = False

    def instrument(self) -> None:
        """Install configured process-level OpenTelemetry instrumentors."""
        if self._instrumented:
            return
        instrumentor_kwargs: dict[str, TracerProvider] = {}
        if self._tracer_provider is not None:
            instrumentor_kwargs["tracer_provider"] = self._tracer_provider
        if self._botocore_instrumentor is not None:
            self._botocore_instrumentor.instrument(**instrumentor_kwargs)
        if self._httpx_instrumentor is not None:
            if self._redact_http_auth_headers:
                self._httpx_instrumentor.instrument(request_hook=scrub_auth_header, **instrumentor_kwargs)
            else:
                self._httpx_instrumentor.instrument(**instrumentor_kwargs)
        self._instrumented = True

    def uninstrument(self) -> None:
        """Remove configured process-level OpenTelemetry instrumentors."""
        if not self._instrumented:
            return

        try:
            if self._botocore_instrumentor is not None:
                self._botocore_instrumentor.uninstrument()
            if self._httpx_instrumentor is not None:
                self._httpx_instrumentor.uninstrument()
        finally:
            self._instrumented = False

    def langchain_callbacks(
        self,
        *,
        proxy_mode: bool,
        callbacks: Sequence[BaseCallbackHandler] = (),
        proxy_callbacks: Sequence[BaseCallbackHandler] = (),
    ) -> list[BaseCallbackHandler]:
        """Return callbacks for one LangChain/LangGraph call.

        When `proxy_mode` is true, the returned list contains caller callbacks plus proxy callbacks. Otherwise it prepends this object's Langfuse
        LangChain callback handler.
        """
        if proxy_mode:
            return [*callbacks, *proxy_callbacks]
        return [self._langchain_handler, *callbacks]

    @contextmanager
    def trace_call(
        self,
        *,
        name: str,
        session_id: str,
        user_id: str,
        root_observation: bool,
        root_kind: ObservationKind = ObservationKind.AGENT,
        tags: Sequence[str] = (),
        metadata: dict[str, JsonValue] | None = None,
        version: str | None = None,
        trace_name: str | None = None,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        root_input: object = None,
        root_metadata: dict[str, JsonValue] | None = None,
    ) -> Iterator[None]:
        """Attach Langfuse trace attributes and optionally open the root observation.

        The context manager always calls `propagate_attributes`; when `root_observation` is true it also opens `client.start_as_current_observation`.
        """
        if trace_id and parent_span_id:
            trace_context = TraceContext(trace_id=trace_id, parent_span_id=parent_span_id)
        elif trace_id:
            trace_context = TraceContext(trace_id=trace_id)
        else:
            trace_context = None
        if not tags and metadata is None and version is None and trace_name is None:
            propagation = propagate_attributes(session_id=session_id, user_id=user_id)
        else:
            propagation = propagate_attributes(
                session_id=session_id,
                user_id=user_id,
                trace_name=trace_name or name,
                tags=list(tags) or None,
                metadata=metadata,
                version=version,
            )

        with propagation:
            if not root_observation:
                yield
                return
            if trace_context is None and root_input is None and root_metadata is None and version is None:
                with self.client.start_as_current_observation(name=name, as_type=root_kind.value):
                    yield
                return
            with self.client.start_as_current_observation(
                trace_context=trace_context,
                name=name,
                as_type=root_kind.value,
                input=to_json_value(root_input),
                metadata=root_metadata,
                version=version,
            ):
                yield

    def trace_url(self, *, trace_id: str | None = None) -> str | None:
        """Return a Langfuse trace URL for the active or provided trace id."""
        return self.client.get_trace_url(trace_id=trace_id)

    @staticmethod
    def deterministic_trace_id(seed: str) -> str:
        """Return a deterministic Langfuse trace id for an external id."""
        return Langfuse.create_trace_id(seed=seed)

    async def shutdown(self) -> None:
        """Stop instrumentation and shutdown Langfuse telemetry with the library timeout."""
        try:
            self.uninstrument()
        finally:
            await shutdown_langfuse_client(self.client)


def init_langfuse_telemetry(
    config: LangfuseClientConfig,
    *,
    instrument_httpx: bool = True,
    instrument_botocore: bool = False,
) -> LangfuseTelemetry:
    """Initialize Langfuse and install project-level instrumentation hooks."""
    telemetry = LangfuseTelemetry(
        init_langfuse_client(config),
        public_key=config.public_key,
        instrument_httpx=instrument_httpx,
        instrument_botocore=instrument_botocore,
        tracer_provider=config.tracer_provider,
        redact_http_auth_headers=config.redact_http_auth_headers,
    )
    telemetry.instrument()
    return telemetry
