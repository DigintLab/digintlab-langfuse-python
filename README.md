# digintlab-langfuse-python

Typed Python helpers for Langfuse tracing, scoring, callbacks, and LiteLLM metadata in DigIntLab agent runtimes.

The package keeps the reusable Langfuse integration logic outside application code. It owns SDK client construction, process instrumentation, LangChain callbacks, trace propagation, observation helpers, score helpers, and JSON-safe payload conversion. Application repositories should keep their own domain-specific defaults and adapters on top of this package.

## Install

```bash
uv add digintlab-langfuse-python
```

The import package is:

```python
import digintlab_langfuse
```

## Development

```bash
uv sync --group dev
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest
```

## Scope

This library is intentionally small:

- `digintlab_langfuse.client` builds Langfuse clients, owns instrumentation, forwards configured tracer providers to HTTP/AWS instrumentors, and provides trace context managers.
- `digintlab_langfuse.spans` opens observations, generation spans, guardrail observations, and HTTP auth redaction hooks.
- `digintlab_langfuse.callbacks` emits Langfuse tool observations from LangChain tool callbacks.
- `digintlab_langfuse.litellm` forwards active Langfuse trace metadata into LiteLLM-compatible request kwargs.
- `digintlab_langfuse.scores` provides typed score helpers for trace, observation, span, and session scores.
- `digintlab_langfuse.timing` provides local elapsed timers and OpenTelemetry span timing aggregation.

Prompt and experiment helpers exist for Langfuse workflows, but applications should expose them only when they have a real product use case.

## Examples

Emit a numeric elapsed-time score for a local operation:

```python
from digintlab_langfuse import ScoreTarget, score_elapsed

with score_elapsed(
    lf,
    name="provider_call_elapsed_seconds",
    target=ScoreTarget(run_id=run_id, trace_id=trace_id),
    digits=2,
):
    call_provider()
```

Emit common run and current-trace scores through the telemetry owner without manually building score objects:

```python
telemetry.emit_numeric_score("provider_call_elapsed_seconds", 1.42, run_id=run_id, trace_id=trace_id)
telemetry.score_current_trace_categorical("routing_decision", "escalated")
```

Use bound runtime helpers when an application owns one telemetry object:

```python
tool_callback = telemetry.tool_span_callback(summarize_output=summarize_tool_output)
with telemetry.observe_node_span("correlate", enabled=True):
    run_node()
kwargs = telemetry.with_litellm_trace_metadata(kwargs)
```

Collect provider/client-only OpenTelemetry span timings:

```python
from digintlab_langfuse import LangfuseClientConfig, create_span_timing_recorder, init_langfuse_telemetry

timing = create_span_timing_recorder()
telemetry = init_langfuse_telemetry(
    LangfuseClientConfig(
        public_key=public_key,
        secret_key=secret_key,
        tracer_provider=timing.tracer_provider,
    )
)

buckets = timing.snapshot(reset=True)
```

## Validation

CI runs formatting, linting, strict typing, and tests through `uv`.
