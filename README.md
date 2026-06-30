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
- `digintlab_langfuse.budgets` validates numeric ceilings and compares observed measurements to budget limits.

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
