"""LangChain tool callbacks that create and close Langfuse observations.

`ToolSpanCallback` handles `on_tool_start`, `on_tool_end`, and `on_tool_error`. It stores observations by LangChain run id, uses the active
Langfuse trace/parent ids when present, and converts callback payloads to `JsonValue` before updating Langfuse.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langfuse import Langfuse, get_client
from langfuse.types import TraceContext

from digintlab_langfuse.spans import (
    OBSERVATION_STATUS_MESSAGE_MAX_CHARS,
    ObservationKind,
    ObservationLevel,
    active_parent_observation_id,
    current_trace_id,
)
from digintlab_langfuse.types import JsonValue, to_json_value

if TYPE_CHECKING:
    from langfuse._client.span import LangfuseObservationWrapper

logger = logging.getLogger(__name__)


class ToolObservationResult(TypedDict, total=False):
    """Optional fields used to finish a Langfuse tool observation."""

    output: JsonValue
    level: ObservationLevel
    status_message: str


def _tool_name(serialized: dict[str, object], kwargs: dict[str, object]) -> str:
    """Return `serialized["name"]`, `kwargs["name"]`, the last serialized id item, or `tool`."""
    raw_name = serialized.get("name") or kwargs.get("name")
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    raw_id = serialized.get("id")
    if isinstance(raw_id, list) and raw_id and isinstance(raw_id[-1], str) and raw_id[-1].strip():
        return raw_id[-1].strip()
    return "tool"


def _default_tool_output_summary(output: object) -> ToolObservationResult:
    """Wrap arbitrary tool output in the default successful observation shape."""
    return {"output": {"status": "ok", "output": to_json_value(output)}}


def _tool_status_message(summary: ToolObservationResult) -> str:
    """Build the short Langfuse status message shown for failed tool observations."""
    output = summary.get("output")
    if not isinstance(output, dict):
        return "tool_error"
    error = output.get("error", "tool_error")
    message = output.get("message", "")
    error_text = error if isinstance(error, str) else "tool_error"
    message_text = message if isinstance(message, str) else ""
    return (
        f"{error_text}: {message_text}"[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS] if message_text else error_text[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS]
    )


class ToolSpanCallback(BaseCallbackHandler):
    """Emit Langfuse tool observations under the active graph-node span.

    `_spans` maps LangChain run ids to Langfuse observation wrappers. Start callbacks return without storing a span when there is no active trace.
    """

    def __init__(
        self,
        *,
        summarize_output: Callable[[object], ToolObservationResult] | None = None,
        client_getter: Callable[[], Langfuse] | None = None,
        json_value_converter: Callable[[object], JsonValue] = to_json_value,
    ) -> None:
        super().__init__()
        self._spans: dict[str, LangfuseObservationWrapper] = {}
        self._summarize_output = summarize_output or _default_tool_output_summary
        self._client_getter = client_getter or get_client
        self._json_value_converter = json_value_converter

    def on_tool_start(
        self,
        serialized: dict[str, object],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        inputs: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        try:
            lf = self._client_getter()
            trace_id = current_trace_id(lf)
        except Exception:
            logger.exception("Failed to prepare Langfuse tool observation")
            return
        if not trace_id:
            return

        try:
            current_observation_id = lf.get_current_observation_id()
        except Exception:
            logger.exception("Failed to resolve the current Langfuse observation")
            return
        parent_id = active_parent_observation_id() or (current_observation_id if isinstance(current_observation_id, str) else None)
        trace_context = TraceContext(trace_id=trace_id, parent_span_id=parent_id) if parent_id else TraceContext(trace_id=trace_id)
        tool_input: dict[str, JsonValue] = {"input": input_str}
        if inputs is not None:
            tool_input["inputs"] = self._json_value_converter(inputs)
        tool_metadata: dict[str, JsonValue] = {
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else "",
            "tags": self._json_value_converter(tags or []),
        }
        if metadata:
            tool_metadata["metadata"] = self._json_value_converter(metadata)

        try:
            span = lf.start_observation(
                trace_context=trace_context,
                name=_tool_name(serialized, kwargs),
                as_type=ObservationKind.TOOL.value,
                input=tool_input,
                metadata=tool_metadata,
            )
        except Exception:
            logger.exception("Failed to start Langfuse tool observation")
            return
        self._spans[str(run_id)] = span

    def on_tool_end(
        self,
        output: object,
        *,
        run_id: UUID,
        **_kwargs: object,
    ) -> None:
        span = self._spans.pop(str(run_id), None)
        if span is None:
            return
        try:
            summary = self._summarize_output(output)
        except BaseException:
            self._end_tool_observation(span)
            raise
        level = summary.get("level")
        status_message = summary.get("status_message")
        try:
            if level is None and status_message is None:
                span.update(output=summary.get("output"))
            elif level is None:
                span.update(output=summary.get("output"), status_message=status_message)
            elif status_message is None:
                span.update(output=summary.get("output"), level=level.value)
            else:
                span.update(output=summary.get("output"), level=level.value, status_message=status_message)
        except Exception:
            logger.exception("Failed to update Langfuse tool observation")
        finally:
            self._end_tool_observation(span)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **_kwargs: object,
    ) -> None:
        span = self._spans.pop(str(run_id), None)
        if span is None:
            return
        summary: ToolObservationResult = {
            "output": {
                "status": "error",
                "error": type(error).__name__,
                "message": str(error)[:OBSERVATION_STATUS_MESSAGE_MAX_CHARS],
            },
            "level": ObservationLevel.ERROR,
        }
        summary["status_message"] = _tool_status_message(summary)
        try:
            span.update(output=summary["output"], level=summary["level"].value, status_message=summary["status_message"])
        except Exception:
            logger.exception("Failed to update Langfuse tool observation after tool error")
        finally:
            self._end_tool_observation(span)

    @staticmethod
    def _end_tool_observation(span: "LangfuseObservationWrapper") -> None:
        try:
            span.end()
        except Exception:
            logger.exception("Failed to end Langfuse tool observation")
