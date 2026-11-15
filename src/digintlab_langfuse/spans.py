"""Langfuse observation helpers backed by context-local active-parent state.

`observe_node_span()` opens observations and keeps their handles in context-local active-parent state. Other helpers read that state, emit guardrail
observations, return the current trace id, and redact credential fields from instrumented HTTP spans.
"""

import logging
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

from langfuse import (
    Langfuse,
    LangfuseAgent,
    LangfuseChain,
    LangfuseEmbedding,
    LangfuseEvaluator,
    LangfuseGeneration,
    LangfuseGuardrail,
    LangfuseRetriever,
    LangfuseSpan,
    LangfuseTool,
    get_client,
)
from langfuse.api.commons.types import ObservationLevel
from langfuse.model import ChatPromptClient, TextPromptClient
from langfuse.types import TraceContext
from opentelemetry.instrumentation.httpx import RequestInfo
from opentelemetry.trace import Span

from digintlab_langfuse._reliability import fail_open_context
from digintlab_langfuse.json_values import JsonValue, to_json_value

__all__ = [
    "ObservationKind",
    "TraceRuntimeContext",
    "active_parent_observation_id",
    "current_trace_id",
    "emit_guardrail_observation",
    "observe_generation",
    "observe_node_span",
    "scrub_auth_header",
    "scrub_auth_header_async",
    "trace_runtime_context",
]

OBSERVATION_STATUS_MESSAGE_MAX_CHARS = 200
logger = logging.getLogger(__name__)
type _NodeObservation = (
    LangfuseSpan
    | LangfuseGeneration
    | LangfuseEmbedding
    | LangfuseAgent
    | LangfuseTool
    | LangfuseChain
    | LangfuseRetriever
    | LangfuseEvaluator
    | LangfuseGuardrail
)


class ObservationKind(StrEnum):
    """Observation `as_type` values used by this adapter when opening Langfuse observations."""

    GENERATION = "generation"
    EMBEDDING = "embedding"
    SPAN = "span"
    AGENT = "agent"
    TOOL = "tool"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    EVALUATOR = "evaluator"
    GUARDRAIL = "guardrail"


@dataclass(slots=True)
class _ActiveParentObservation:
    client: Langfuse
    observation: _NodeObservation
    active: bool = True


_active_parent_observation_context: ContextVar[_ActiveParentObservation | None] = ContextVar(
    "active_langfuse_parent_observation",
    default=None,
)


@dataclass(frozen=True, slots=True)
class TraceRuntimeContext:
    """Portable Langfuse trace context for work that cannot rely on the active SDK context."""

    trace_id: str
    parent_observation_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None
    run_id: str | None = None
    metadata: Mapping[str, JsonValue] | None = None

    def as_trace_context(self) -> TraceContext:
        """Return the SDK trace context representation."""
        if self.parent_observation_id:
            return TraceContext(trace_id=self.trace_id, parent_span_id=self.parent_observation_id)
        return TraceContext(trace_id=self.trace_id)

    def as_litellm_metadata(self) -> dict[str, str]:
        """Return string metadata suitable for LiteLLM/OpenAI-compatible forwarding."""
        metadata = {str(key): str(value) for key, value in (self.metadata or {}).items() if value is not None}
        metadata["existing_trace_id"] = self.trace_id
        metadata["trace_id"] = self.trace_id
        if self.run_id:
            metadata["run_id"] = self.run_id
            metadata.setdefault("session_id", self.run_id)
        if self.session_id:
            metadata["session_id"] = self.session_id
        if self.user_id:
            metadata["trace_user_id"] = self.user_id
        if self.parent_observation_id:
            metadata["parent_observation_id"] = self.parent_observation_id
        return metadata


def _shares_langfuse_owner(client: Langfuse, candidate: object) -> bool:
    if client is candidate:
        return True
    if not isinstance(candidate, Langfuse):
        return False
    resources = client._resources  # noqa: SLF001 - exact-pinned SDK facades share this owner.
    return resources is not None and resources is candidate._resources  # noqa: SLF001


def active_parent_observation_id(client: object | None = None) -> str | None:
    """Return the active node observation id when it belongs to the requested client."""
    parent = _active_parent_observation()
    if parent is None or (client is not None and not _shares_langfuse_owner(parent.client, client)):
        return None
    observation_id = parent.observation.id
    return observation_id if isinstance(observation_id, str) and observation_id else None


def current_trace_id(client: Langfuse) -> str | None:
    """Return the active Langfuse trace id, or None when no trace is active."""
    trace_id = client.get_current_trace_id()
    return trace_id if isinstance(trace_id, str) and trace_id else None


def trace_runtime_context(
    client: Langfuse,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    run_id: str | None = None,
    metadata: Mapping[str, JsonValue] | None = None,
) -> TraceRuntimeContext | None:
    """Capture the active Langfuse trace and parent observation as a portable runtime context."""
    trace_id = current_trace_id(client)
    if trace_id is None:
        return None

    raw_current_observation_id = client.get_current_observation_id()
    current_observation_id = raw_current_observation_id if isinstance(raw_current_observation_id, str) and raw_current_observation_id else None
    return TraceRuntimeContext(
        trace_id=trace_id,
        parent_observation_id=active_parent_observation_id(client) or current_observation_id,
        session_id=session_id,
        user_id=user_id,
        run_id=run_id,
        metadata=metadata,
    )


def _active_parent_observation() -> _ActiveParentObservation | None:
    parent = _active_parent_observation_context.get()
    return parent if parent is not None and parent.active else None


def _start_current_node_observation(
    client: Langfuse,
    *,
    name: str,
    kind: ObservationKind,
) -> AbstractContextManager[_NodeObservation]:
    return client.start_as_current_observation(name=name, as_type=kind.value)


@contextmanager
def observe_node_span(
    name: str,
    *,
    enabled: bool,
    kind: ObservationKind = ObservationKind.SPAN,
    client_getter: Callable[[], Langfuse] | None = None,
) -> Iterator[None]:
    """Open a Langfuse observation for an application graph node.

    The context manager yields without opening an observation when disabled or when the client has no current trace id. Exceptions update the
    observation level/status before being re-raised. Telemetry lifecycle failures are logged and fail open.
    """
    if not enabled:
        yield
        return

    try:
        lf = (client_getter or get_client)()
        trace_id = current_trace_id(lf)
    except Exception:
        logger.exception("Failed to prepare Langfuse node observation")
        yield
        return
    if not trace_id:
        yield
        return

    with fail_open_context(
        lambda: _start_current_node_observation(lf, name=name, kind=kind),
        logger=logger,
        operation="Langfuse node observation lifecycle",
    ) as span:
        if span is None:
            yield
            return
        parent = _ActiveParentObservation(client=lf, observation=span)
        token = _active_parent_observation_context.set(parent)
        try:
            yield
        except BaseException as exc:
            try:
                span.update(level=ObservationLevel.ERROR.value, status_message=str(exc)[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS])
            except Exception:
                logger.exception("Failed to update Langfuse node observation after application error")
            raise
        finally:
            parent.active = False
            _active_parent_observation_context.reset(token)


@contextmanager
def observe_generation(
    name: str,
    *,
    enabled: bool = True,
    model: str | None = None,
    observation_input: object = None,
    metadata: Mapping[str, JsonValue] | None = None,
    version: str | None = None,
    level: ObservationLevel = ObservationLevel.DEFAULT,
    status_message: str = "",
    completion_start_time: datetime | None = None,
    usage_details: Mapping[str, int] | None = None,
    cost_details: Mapping[str, float] | None = None,
    prompt: TextPromptClient | ChatPromptClient | None = None,
    client_getter: Callable[[], Langfuse] | None = None,
) -> Iterator[LangfuseGeneration | None]:
    """Open a generation observation and yield the SDK handle for optional updates."""
    if not enabled:
        yield None
        return

    try:
        lf = (client_getter or get_client)()
        trace_id = current_trace_id(lf)
    except Exception:
        logger.exception("Failed to prepare Langfuse generation observation")
        yield None
        return
    if not trace_id:
        yield None
        return

    status = status_message[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS] if status_message else None
    with fail_open_context(
        lambda: lf.start_as_current_observation(
            name=name,
            as_type=ObservationKind.GENERATION.value,
            input=to_json_value(observation_input),
            metadata=dict(metadata) if metadata is not None else None,
            version=version,
            level=level.value,
            status_message=status,
            completion_start_time=completion_start_time,
            model=model,
            usage_details=dict(usage_details) if usage_details is not None else None,
            cost_details=dict(cost_details) if cost_details is not None else None,
            prompt=prompt,
        ),
        logger=logger,
        operation="Langfuse generation observation lifecycle",
    ) as generation:
        if generation is None:
            yield None
            return
        try:
            yield generation
        except BaseException as exc:
            try:
                generation.update(level=ObservationLevel.ERROR.value, status_message=str(exc)[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS])
            except Exception:
                logger.exception("Failed to update Langfuse generation observation after application error")
            raise


def emit_guardrail_observation(
    name: str,
    *,
    output: JsonValue,
    observation_input: JsonValue = None,
    level: ObservationLevel = ObservationLevel.WARNING,
    status_message: str = "",
    client_getter: Callable[[], Langfuse] | None = None,
) -> None:
    """Emit a short-lived guardrail observation under the active node.

    The function returns without emitting when there is no active parent observation or no current trace id. Telemetry lifecycle failures are logged
    and fail open.
    """
    try:
        parent = _active_parent_observation()
        if parent is None:
            return

        lf = (client_getter or get_client)()
        if not _shares_langfuse_owner(parent.client, lf) or current_trace_id(lf) is None:
            return

        status = status_message[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS] if status_message else None
        span = parent.observation.start_observation(
            name=name,
            as_type=ObservationKind.GUARDRAIL.value,
            input=observation_input,
            output=output,
            level=level.value,
            status_message=status,
        )
        span.end()
    except Exception:
        logger.exception("Failed to emit Langfuse guardrail observation")


def scrub_auth_header(span: Span, request: RequestInfo) -> None:
    """Redact credential material from OpenTelemetry HTTP spans before export."""
    if not span.is_recording():
        return
    if request.headers is not None and "authorization" in request.headers:
        span.set_attribute("http.request.header.authorization", "[REDACTED]")


async def scrub_auth_header_async(span: Span, request: RequestInfo) -> None:
    """Adapt authorization redaction to the HTTPX async request-hook signature."""
    scrub_auth_header(span, request)
