# digintlab-langfuse-python

Typed helpers for tracing, scoring, and redacting agent runs with Langfuse.

## Install

Python 3.13 to 3.15. Not on PyPI; install a tagged release from GitHub:

```bash
uv add "digintlab-langfuse-python @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.5.0"
```

Add the `[redaction]` extra for PII detection. Each [release](https://github.com/DigintLab/digintlab-langfuse-python/releases) also ships a wheel and an sdist.

## Usage

```python
from digintlab_langfuse import LangfuseClientConfig, init_langfuse_telemetry
from digintlab_langfuse.scores import NumericScore, ScoreTarget

telemetry = init_langfuse_telemetry(
    LangfuseClientConfig(public_key=public_key, secret_key=secret_key, environment="production"),
)

async with telemetry:
    with telemetry.trace_call(name="agent-run", session_id=session_id, user_id=user_id, root_observation=True):
        await run_agent()

    telemetry.emit_score(score=NumericScore(name="quality", value=0.95), target=ScoreTarget(trace_id=trace_id))
```

Create one telemetry owner per process, after workers fork, and don't construct `Langfuse(...)` yourself. Leaving the `async with` block flushes and shuts down; `await telemetry.flush()` exports without closing.

To continue a trace in another process, capture `telemetry.trace_runtime_context()` and pass it to `with_litellm_proxy_trace_metadata()` on the other side.

## Modules

- `client`: `LangfuseClientConfig`, `init_langfuse_telemetry`, and `LangfuseTelemetry` (`trace_call`, `flush`, `shutdown`, and wrappers for the helpers below).
- `spans`: node, generation, and guardrail observations; the HTTPX auth-header scrubbing hooks.
- `callbacks`: `ToolSpanCallback`, a LangChain handler that records tool calls.
- `litellm`: adds the active trace ids to LiteLLM proxy request metadata.
- `scores`: one typed score API (`NumericScore`, `BooleanScore`, `CategoricalScore`, `TextScore`, `CorrectionScore`).
- `prompts`, `experiments`: thin typed wrappers for Langfuse prompts, datasets, and local experiments.
- `budgets`, `timing`: numeric ceiling checks and span timing aggregation.
- `redaction`, `json_values`: the redaction policy and the JSON value type.

For all configuration fields and their defaults, see `LangfuseClientConfig` in [`client.py`](src/digintlab_langfuse/client.py).

## Sensitive data

Nothing is redacted by default. To mask credential fields, bearer tokens, and PII before export:

```python
from digintlab_langfuse import LangfuseClientConfig, SensitiveDataRedactionConfig

config = LangfuseClientConfig(
    public_key=public_key,
    secret_key=secret_key,
    sensitive_data_redaction=SensitiveDataRedactionConfig(),
)
```

PII detection uses Presidio's local rule-based recognizers. It loads no NLP model and makes no network calls, so it won't catch names, addresses, or non-English text. For a custom policy, pass `mask` and `mask_otel_spans` instead.

To strip `Authorization` headers from HTTPX spans, set `redact_http_auth_headers=True` and initialize with `instrument_httpx=True`.

> [!WARNING]
> Span names, identifiers, links, events, resource attributes, and metadata forwarded to LiteLLM are never rewritten. Keep secrets out of them.

## Limitations

- One Langfuse project per process. A second `init_langfuse_telemetry()` raises, and `shutdown()` does not release the claim.
- The package uses private Langfuse SDK internals, so Langfuse is limited to one minor version (`>=5.2.0,<5.3`). `tests/package_contract/test_langfuse_internals.py` catches breaking changes.
- Observation helpers fail open: errors are logged and never replace your result. Configuration, scoring, flush, and shutdown raise.
- HTTPX and botocore instrumentation is opt-in and process-wide. Instrumentation owned by other code is left alone.
- Plain `http://` endpoints need `allow_insecure_http=True`.

## Development

```bash
task install
task format
task check
task test
```

The E2E suite needs a disposable Langfuse project:

```bash
RUN_LANGFUSE_E2E=1 LANGFUSE_E2E_PUBLIC_KEY=... LANGFUSE_E2E_SECRET_KEY=... \
LANGFUSE_E2E_BASE_URL=https://cloud.langfuse.com task test-e2e
```

## Versioning

The public API is each public module's `__all__` plus the top-level exports; anything starting with an underscore is private. Releases follow [SemVer](https://semver.org/). Before 1.0, minor releases may break things, and those changes are listed in `CHANGELOG.md`. Moving to a new Langfuse minor version is always at least a minor release.

## License

[LGPL-3.0-or-later](COPYING.LESSER). You can use the library in any application, proprietary ones included. If you distribute a modified copy of the library itself, you must share those changes under the LGPL.
