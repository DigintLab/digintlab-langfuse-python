"""Langfuse observation helpers backed by a context-local parent stack.

`observe_node_span()` opens observations and pushes their ids into `_active_parent_observation_ids`. Other helpers read that id, emit guardrail
observations, return the current trace id, and redact credential fields from instrumented HTTP spans.
"""

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime
from enum import StrEnum

import httpx
from langfuse import Langfuse, LangfuseGeneration, get_client
from langfuse.api.commons.types import ObservationLevel
from langfuse.model import ChatPromptClient, TextPromptClient
from langfuse.types import TraceContext
from opentelemetry.trace import Span

from digintlab_langfuse.types import JsonValue, to_json_value

OBSERVATION_STATUS_MESSAGE_MAX_CHARS = 200
_active_parent_observation_ids: ContextVar[tuple[str, ...]] = ContextVar("active_langfuse_parent_observation_ids", default=())


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


def active_parent_observation_id() -> str | None:
    """Return the most recently pushed Langfuse observation id."""
    stack = _active_parent_observation_ids.get()
    return stack[-1] if stack else None


def current_trace_id(client: Langfuse) -> str | None:
    """Return the active Langfuse trace id, or None when no trace is active."""
    trace_id = client.get_current_trace_id()
    return trace_id if isinstance(trace_id, str) and trace_id else None


def _push_active_parent(span_id: str) -> None:
    _active_parent_observation_ids.set((*_active_parent_observation_ids.get(), span_id))


def _pop_active_parent(span_id: str) -> None:
    stack = _active_parent_observation_ids.get()
    if not stack:
        return
    if stack[-1] == span_id:
        _active_parent_observation_ids.set(stack[:-1])
        return
    _active_parent_observation_ids.set(tuple(item for item in stack if item != span_id))


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
    observation level/status before being re-raised.
    """
    if not enabled:
        yield
        return

    lf = (client_getter or get_client)()
    if not lf.get_current_trace_id():
        yield
        return

    span = lf.start_observation(name=name, as_type=kind.value)
    _push_active_parent(span.id)
    try:
        yield
    except BaseException as exc:
        span.update(level=ObservationLevel.ERROR.value, status_message=str(exc)[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS])
        raise
    finally:
        _pop_active_parent(span.id)
        span.end()


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

    lf = (client_getter or get_client)()
    if not lf.get_current_trace_id():
        yield None
        return

    status = status_message[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS] if status_message else None
    with lf.start_as_current_observation(
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
    ) as generation:
        try:
            yield generation
        except BaseException as exc:
            generation.update(level=ObservationLevel.ERROR.value, status_message=str(exc)[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS])
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

    The function returns without emitting when there is no active parent observation or no current trace id.
    """
    parent_id = active_parent_observation_id()
    if not parent_id:
        return

    lf = (client_getter or get_client)()
    trace_id = current_trace_id(lf)
    if not trace_id:
        return

    if status_message:
        span = lf.start_observation(
            trace_context=TraceContext(trace_id=trace_id, parent_span_id=parent_id),
            name=name,
            as_type=ObservationKind.GUARDRAIL.value,
            input=observation_input,
            output=output,
            level=level.value,
            status_message=status_message[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS],
        )
    else:
        span = lf.start_observation(
            trace_context=TraceContext(trace_id=trace_id, parent_span_id=parent_id),
            name=name,
            as_type=ObservationKind.GUARDRAIL.value,
            input=observation_input,
            output=output,
            level=level.value,
        )
    span.end()


def scrub_auth_header(span: Span, request: httpx.Request) -> None:
    """Redact credential material from OpenTelemetry HTTP spans before export."""
    if not span.is_recording():
        return
    if "authorization" in request.headers:
        span.set_attribute("http.request.header.authorization", "[REDACTED]")
    if "/api-token-auth" in request.url.path:
        span.set_attribute("http.request.body", "[REDACTED: credential exchange]")
