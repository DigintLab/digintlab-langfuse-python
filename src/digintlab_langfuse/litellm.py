"""Helpers that add Langfuse trace fields to LiteLLM request kwargs.

The functions read metadata from a LangChain run manager, merge caller-provided metadata, add current Langfuse trace and parent-observation ids, and
copy the merged values into both `metadata` and `extra_body.metadata` when possible.
"""

from collections.abc import Callable
from typing import Protocol

from langfuse import get_client

from digintlab_langfuse.spans import TraceRuntimeContext, active_parent_observation_id
from digintlab_langfuse.types import JsonValue

type LiteLLMParamValue = JsonValue


class LangfuseTraceClient(Protocol):
    """Minimal Langfuse client surface needed to read the active trace context."""

    def get_current_trace_id(self) -> object: ...

    def get_current_observation_id(self) -> object: ...


def metadata_from_run_manager(run_manager: object | None) -> dict[str, str]:
    """Return LangChain run metadata as strings for LiteLLM proxy forwarding."""
    if run_manager is None:
        return {}
    metadata = getattr(run_manager, "metadata", None)
    if not isinstance(metadata, dict):
        return {}
    return {str(k): str(v) for k, v in metadata.items() if v is not None}


def with_extra_body_metadata(kwargs: dict[str, LiteLLMParamValue], metadata: dict[str, str]) -> dict[str, LiteLLMParamValue]:
    """Mirror trace metadata into both top-level and `extra_body.metadata` request fields."""
    extra_body = kwargs.get("extra_body")
    metadata_value: dict[str, LiteLLMParamValue] = dict(metadata)
    if extra_body is None:
        return {**kwargs, "extra_body": {"metadata": metadata_value}}
    if not isinstance(extra_body, dict):
        return {**kwargs, "extra_body": extra_body}

    extra_metadata = extra_body.get("metadata")
    merged_metadata: dict[str, LiteLLMParamValue] = {str(k): str(v) for k, v in extra_metadata.items()} if isinstance(extra_metadata, dict) else {}
    merged_metadata.update(metadata)
    merged_extra_body = {str(k): v for k, v in extra_body.items()}
    merged_extra_body["metadata"] = merged_metadata
    return {**kwargs, "extra_body": merged_extra_body}


def _trace_context_from_client(
    client: LangfuseTraceClient,
    *,
    active_parent_getter: Callable[[], str | None] | None,
) -> TraceRuntimeContext | None:
    trace_id = client.get_current_trace_id()
    if not trace_id:
        return None

    raw_observation_id = (active_parent_getter or active_parent_observation_id)() or client.get_current_observation_id()
    parent_observation_id = raw_observation_id if isinstance(raw_observation_id, str) and raw_observation_id else None
    return TraceRuntimeContext(trace_id=str(trace_id), parent_observation_id=parent_observation_id)


def with_langfuse_trace_metadata(
    kwargs: dict[str, LiteLLMParamValue],
    run_manager: object | None = None,
    *,
    client_getter: Callable[[], LangfuseTraceClient] | None = None,
    active_parent_getter: Callable[[], str | None] | None = None,
    trace_context: TraceRuntimeContext | None = None,
) -> dict[str, LiteLLMParamValue]:
    """Add active Langfuse trace metadata to LiteLLM/OpenAI-compatible kwargs.

    If no trace is active, the original kwargs are returned unchanged. Existing caller metadata is copied before trace fields are added.
    """
    runtime_context = trace_context
    if runtime_context is None:
        runtime_context = _trace_context_from_client((client_getter or get_client)(), active_parent_getter=active_parent_getter)
    if runtime_context is None:
        return kwargs

    metadata = metadata_from_run_manager(run_manager)
    raw = kwargs.get("metadata")
    if isinstance(raw, dict):
        metadata.update({str(k): str(v) for k, v in raw.items()})

    metadata.update(runtime_context.as_litellm_metadata())
    run_id = metadata.get("run_id", "")
    if run_id:
        metadata.setdefault("session_id", run_id)

    metadata_json: dict[str, LiteLLMParamValue] = dict(metadata)
    return with_extra_body_metadata({**kwargs, "metadata": metadata_json}, metadata)
