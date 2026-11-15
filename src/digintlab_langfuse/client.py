"""Langfuse SDK client construction, lifecycle, and instrumentation helpers."""

import asyncio
import logging
import math
import os
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field
from typing import Self, TypedDict

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, propagate_attributes
from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler
from langfuse.types import MaskFunction, MaskOtelSpansFunction, TraceContext
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter

from digintlab_langfuse._instrumentation import ProcessInstrumentation
from digintlab_langfuse._lifecycle import (
    DEFAULT_OPERATION_TIMEOUT_SECONDS,
    flush_client,
    initialize_client,
    shutdown_client,
    shutdown_failed_client,
    shutdown_started,
    validate_operation_timeout,
)
from digintlab_langfuse._reliability import fail_open_context
from digintlab_langfuse.callbacks import ToolObservationResult, ToolSpanCallback
from digintlab_langfuse.json_values import JsonValue, to_json_value
from digintlab_langfuse.litellm import with_litellm_proxy_trace_metadata
from digintlab_langfuse.redaction import SensitiveDataRedactionConfig, create_sensitive_data_mask, create_sensitive_data_otel_mask
from digintlab_langfuse.scores import (
    DEFAULT_SCORE_TARGET,
    Score,
    ScoreTarget,
    ScoreValue,
    emit_score,
    score_current_span,
    score_current_trace,
)
from digintlab_langfuse.spans import (
    ObservationKind,
    TraceRuntimeContext,
    observe_node_span,
)
from digintlab_langfuse.spans import (
    emit_guardrail_observation as emit_langfuse_guardrail_observation,
)
from digintlab_langfuse.spans import (
    trace_runtime_context as capture_trace_runtime_context,
)

logger = logging.getLogger(__name__)

__all__ = ["LangfuseClientConfig", "LangfuseTelemetry", "init_langfuse_telemetry"]


class _LangfuseInitKwargs(TypedDict):
    """Keyword arguments forwarded to the Langfuse SDK client."""

    public_key: str
    secret_key: str
    base_url: str | None
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


def _validate_endpoint_url(base_url: str | None, *, allow_insecure_http: bool) -> None:
    """Enforce endpoint security even when using Langfuse's environment defaults."""
    effective_base_url = base_url
    if effective_base_url is None:
        effective_base_url = os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
    if effective_base_url is None:
        return

    url = httpx.URL(effective_base_url)
    if url.host is None or url.scheme not in {"http", "https"}:
        message = "Langfuse base_url must be an absolute HTTP(S) URL."
        raise ValueError(message)
    if url.scheme != "https" and not allow_insecure_http:
        message = "Plain HTTP Langfuse endpoints require allow_insecure_http=True."
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class LangfuseClientConfig:
    """Configuration required to initialize the Langfuse SDK client."""

    public_key: str
    secret_key: str = field(repr=False)
    base_url: str | None = None
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
    redact_http_auth_headers: bool = False
    sensitive_data_redaction: SensitiveDataRedactionConfig | None = None
    allow_insecure_http: bool = False
    flush_timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS
    shutdown_timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        """Validate process-level client configuration before allocating SDK resources."""
        if not self.public_key.strip() or not self.secret_key.strip():
            message = "Langfuse public_key and secret_key must not be empty."
            raise ValueError(message)
        _validate_endpoint_url(self.base_url, allow_insecure_http=self.allow_insecure_http)
        if self.timeout is not None and self.timeout <= 0:
            message = "Langfuse timeout must be greater than zero."
            raise ValueError(message)
        if self.flush_at is not None and self.flush_at < 1:
            message = "Langfuse flush_at must be at least one."
            raise ValueError(message)
        if self.flush_interval is not None and (not math.isfinite(self.flush_interval) or self.flush_interval <= 0):
            message = "Langfuse flush_interval must be finite and greater than zero."
            raise ValueError(message)
        if self.sample_rate is not None and not 0 <= self.sample_rate <= 1:
            message = "Langfuse sample_rate must be between zero and one."
            raise ValueError(message)
        validate_operation_timeout(self.flush_timeout_seconds)
        validate_operation_timeout(self.shutdown_timeout_seconds)
        if self.sensitive_data_redaction is not None and (self.mask is not None or self.mask_otel_spans is not None):
            message = "Pass sensitive_data_redaction or custom mask hooks, not both."
            raise ValueError(message)


def _init_langfuse_client(config: LangfuseClientConfig) -> Langfuse:
    """Initialize the sole Langfuse SDK resource owner for one public key."""
    mask = create_sensitive_data_mask(config.sensitive_data_redaction) if config.sensitive_data_redaction is not None else config.mask
    mask_otel_spans = (
        create_sensitive_data_otel_mask(config.sensitive_data_redaction) if config.sensitive_data_redaction is not None else config.mask_otel_spans
    )
    kwargs: _LangfuseInitKwargs = {
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
        "mask": mask,
        "mask_otel_spans": mask_otel_spans,
        "httpx_client": config.httpx_client,
        "should_export_span": config.should_export_span,
        "tracer_provider": config.tracer_provider,
        "span_exporter": config.span_exporter,
    }
    return initialize_client(config.public_key, lambda: Langfuse(**kwargs))


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
        redact_http_auth_headers: bool = False,
        flush_timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS,
        shutdown_timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS,
    ) -> None:
        validate_operation_timeout(flush_timeout_seconds)
        validate_operation_timeout(shutdown_timeout_seconds)
        self.client = client
        self._langchain_handler = LangfuseCallbackHandler(public_key=public_key)
        self._instrumentation = ProcessInstrumentation(
            instrument_httpx=instrument_httpx,
            instrument_botocore=instrument_botocore,
            tracer_provider=tracer_provider,
            redact_http_auth_headers=redact_http_auth_headers,
        )
        self._flush_timeout_seconds = flush_timeout_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds
        self._client_shutdown = False
        self._shutdown_lock = asyncio.Lock()

    def instrument(self) -> None:
        """Install configured process-level OpenTelemetry instrumentors."""
        self._instrumentation.instrument()

    def uninstrument(self) -> None:
        """Remove configured process-level OpenTelemetry instrumentors."""
        self._instrumentation.uninstrument()

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

    def tool_span_callback(
        self,
        *,
        summarize_output: Callable[[object], ToolObservationResult] | None = None,
        json_value_converter: Callable[[object], JsonValue] = to_json_value,
    ) -> ToolSpanCallback:
        """Return a LangChain callback that writes tool observations with this client's trace context."""
        return ToolSpanCallback(
            summarize_output=summarize_output,
            client_getter=lambda: self.client,
            json_value_converter=json_value_converter,
        )

    def observe_node_span(
        self,
        name: str,
        *,
        enabled: bool,
        kind: ObservationKind = ObservationKind.SPAN,
    ) -> AbstractContextManager[None]:
        """Open a node observation with this telemetry client's active trace context."""
        return observe_node_span(name, enabled=enabled, kind=kind, client_getter=lambda: self.client)

    def emit_guardrail_observation(
        self,
        name: str,
        *,
        output: JsonValue,
        observation_input: JsonValue = None,
        status_message: str = "",
    ) -> None:
        """Emit a guardrail observation with this telemetry client's active trace context."""
        emit_langfuse_guardrail_observation(
            name,
            observation_input=observation_input,
            output=output,
            status_message=status_message,
            client_getter=lambda: self.client,
        )

    def with_litellm_proxy_trace_metadata(
        self,
        kwargs: dict[str, JsonValue],
        run_manager: object | None = None,
        *,
        trace_context: TraceRuntimeContext | None = None,
    ) -> dict[str, JsonValue]:
        """Attach this telemetry client's active trace metadata to LiteLLM proxy kwargs."""
        return with_litellm_proxy_trace_metadata(
            kwargs,
            run_manager,
            client_getter=lambda: self.client,
            trace_context=trace_context,
        )

    def trace_runtime_context(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> TraceRuntimeContext | None:
        """Capture portable trace linkage for detached or queued work."""
        return capture_trace_runtime_context(
            self.client,
            session_id=session_id,
            user_id=user_id,
            run_id=run_id,
            metadata=metadata,
        )

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

        def start_propagation() -> AbstractContextManager[object]:
            if not tags and metadata is None and version is None and trace_name is None:
                return propagate_attributes(session_id=session_id, user_id=user_id)
            return propagate_attributes(
                session_id=session_id,
                user_id=user_id,
                trace_name=trace_name or name,
                tags=list(tags) or None,
                metadata=metadata,
                version=version,
            )

        with fail_open_context(
            start_propagation,
            logger=logger,
            operation="Langfuse trace attribute propagation",
        ):
            if not root_observation:
                yield
                return

            def start_root_observation() -> AbstractContextManager[object]:
                if trace_context is None and root_input is None and root_metadata is None and version is None:
                    return self.client.start_as_current_observation(name=name, as_type=root_kind.value)
                return self.client.start_as_current_observation(
                    trace_context=trace_context,
                    name=name,
                    as_type=root_kind.value,
                    input=to_json_value(root_input),
                    metadata=root_metadata,
                    version=version,
                )

            with fail_open_context(
                start_root_observation,
                logger=logger,
                operation="Langfuse root observation lifecycle",
            ):
                yield

    def emit_score[TValue: ScoreValue](
        self,
        *,
        score: Score[TValue],
        target: ScoreTarget = DEFAULT_SCORE_TARGET,
    ) -> None:
        """Emit a typed Langfuse score with this telemetry client's SDK client."""
        emit_score(self.client, score=score, target=target)

    def score_current_trace[TValue: ScoreValue](self, *, score: Score[TValue]) -> None:
        """Attach a typed score to the trace active in this telemetry client."""
        score_current_trace(self.client, score=score)

    def score_current_span[TValue: ScoreValue](self, *, score: Score[TValue]) -> None:
        """Attach a typed score to the active span or generation in this telemetry client."""
        score_current_span(self.client, score=score)

    async def shutdown(self) -> None:
        """Stop instrumentation and shutdown Langfuse telemetry with the library timeout."""
        async with self._shutdown_lock:
            uninstrument_error: Exception | None = None
            try:
                self.uninstrument()
            except RuntimeError as exc:
                uninstrument_error = exc
            if not self._client_shutdown:
                await shutdown_client(self.client, timeout_seconds=self._shutdown_timeout_seconds)
                self._client_shutdown = True
            if uninstrument_error is not None:
                raise uninstrument_error

    async def flush(self) -> None:
        """Flush pending Langfuse telemetry without shutting down the client."""
        async with self._shutdown_lock:
            if self._client_shutdown:
                message = "Cannot flush Langfuse telemetry after shutdown."
                raise RuntimeError(message)
            await flush_client(self.client, timeout_seconds=self._flush_timeout_seconds)

    async def __aenter__(self) -> Self:
        """Return this initialized telemetry owner."""
        if self._client_shutdown or shutdown_started(self.client):
            message = "Cannot enter a Langfuse telemetry owner after shutdown."
            raise RuntimeError(message)
        return self

    async def __aexit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        """Shutdown telemetry when leaving an async ownership scope."""
        await self.shutdown()


def init_langfuse_telemetry(
    config: LangfuseClientConfig,
    *,
    instrument_httpx: bool = False,
    instrument_botocore: bool = False,
) -> LangfuseTelemetry:
    """Initialize Langfuse and install project-level instrumentation hooks."""
    client = _init_langfuse_client(config)
    try:
        telemetry = LangfuseTelemetry(
            client,
            public_key=config.public_key,
            instrument_httpx=instrument_httpx,
            instrument_botocore=instrument_botocore,
            tracer_provider=config.tracer_provider,
            redact_http_auth_headers=config.redact_http_auth_headers,
            flush_timeout_seconds=config.flush_timeout_seconds,
            shutdown_timeout_seconds=config.shutdown_timeout_seconds,
        )
        telemetry.instrument()
    except BaseException:
        try:
            shutdown_failed_client(
                client,
                public_key=config.public_key,
                timeout_seconds=config.shutdown_timeout_seconds,
            )
        except Exception:
            logger.exception("Failed to shutdown Langfuse client after instrumentation setup failure")
        raise
    return telemetry
