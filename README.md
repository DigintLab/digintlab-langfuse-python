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

- `digintlab_langfuse.client` builds Langfuse clients, owns instrumentation, and provides trace context managers.
- `digintlab_langfuse.spans` opens observations, generation spans, guardrail observations, and HTTP auth redaction hooks.
- `digintlab_langfuse.callbacks` emits Langfuse tool observations from LangChain tool callbacks.
- `digintlab_langfuse.litellm` forwards active Langfuse trace metadata into LiteLLM-compatible request kwargs.
- `digintlab_langfuse.scores` provides typed score helpers for trace, observation, span, and session scores.

Prompt and experiment helpers exist for Langfuse workflows, but applications should expose them only when they have a real product use case.

## Validation

CI runs formatting, linting, strict typing, and tests through `uv`.
