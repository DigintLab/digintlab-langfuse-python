"""Trace runtime context tests."""

from digintlab_langfuse.litellm import with_litellm_proxy_trace_metadata
from digintlab_langfuse.spans import TraceRuntimeContext


def test_trace_runtime_context_exports_litellm_metadata() -> None:
    context = TraceRuntimeContext(
        trace_id="trace-1",
        parent_observation_id="obs-1",
        run_id="run-1",
        user_id="analyst",
        metadata={"tenant": "acme"},
    )

    result = with_litellm_proxy_trace_metadata({"metadata": {"request_id": "req-1"}}, trace_context=context)

    raw_extra_body = result["extra_body"]
    assert isinstance(raw_extra_body, dict)
    raw_metadata = raw_extra_body["metadata"]
    assert isinstance(raw_metadata, dict)
    metadata = {str(key): str(value) for key, value in raw_metadata.items()}
    assert metadata == {
        "request_id": "req-1",
        "tenant": "acme",
        "existing_trace_id": "trace-1",
        "trace_id": "trace-1",
        "run_id": "run-1",
        "session_id": "run-1",
        "user_id": "analyst",
        "parent_observation_id": "obs-1",
    }
