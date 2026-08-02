# digintlab-langfuse-python

Typed Python helpers for Langfuse tracing, scoring, callbacks, and LiteLLM metadata in DigIntLab agent runtimes.

The package keeps the reusable Langfuse integration logic outside application code. It owns SDK client construction, process instrumentation, LangChain callbacks, trace propagation, observation helpers, score helpers, and JSON-safe payload conversion. Application repositories should keep their own domain-specific defaults and adapters on top of this package.

## Install

Requires Python 3.13+ and Langfuse 4.x (`>=4.14.2,<5`).

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
- `digintlab_langfuse.budgets` validates numeric ceilings and compares observed measurements to budget limits.

Prompt and experiment helpers exist for Langfuse workflows, but applications should expose them only when they have a real product use case.

## Observation lifecycle

`observe_node_span()` is a context manager, not a detached trace-link helper. When tracing is enabled and a trace is active, it opens the node with Langfuse `start_as_current_observation`; this makes the node the SDK-current parent for work executed inside the block and requests closure of the node observation on normal or exceptional exit. When disabled or outside an active trace, it is a no-op.

`emit_guardrail_observation()` emits only while an `observe_node_span()` scope is active. The package-local state retains the live node handle and a shared active flag: the helper starts the guardrail from that handle, making it a direct node child even if a generation is currently nested inside the node, and copied async contexts become inert after the node closes. It does not construct a detached `TraceContext`. The guardrail observation is explicitly ended before the function returns. Application-level root observations opened by `LangfuseTelemetry.trace_call()` are context-managed in the same way. When the SDK accepts the lifecycle calls, their root, node, and guardrail observations therefore all have an end time after those scopes exit.

Runtime observation failures are fail-open: client lookup and observation start, update, or end errors are logged without replacing application results or exceptions. Initialization and instrumentation configuration remain fail-fast. If the SDK itself rejects an end call, the package cannot guarantee a server-side end time.

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
    telemetry.emit_guardrail_observation(
        "output_policy",
        output={"allowed": True},
        status_message="passed",
    )
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

Check numeric runtime or cost budgets:

```python
from digintlab_langfuse import check_numeric_budgets, validate_numeric_budget_ceilings

ceilings = validate_numeric_budget_ceilings({"latency_seconds": 10.0}, allowed_names={"latency_seconds"})
violations = check_numeric_budgets({"latency_seconds": 12.5}, ceilings)
```

## Validation

CI runs formatting, linting, strict typing, and tests through `uv`.
