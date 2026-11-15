# digintlab-langfuse-python

Typed Python helpers for Langfuse tracing, scoring, callbacks, and LiteLLM metadata in DigIntLab agent runtimes.

## Install

Requires Python 3.13, 3.14, or 3.15. Langfuse is pinned to the version tested by this package.

The package is not published on PyPI. Install it from a tagged GitHub release (replace `v0.3.0` with the release you want). Git installs use your configured Git credentials when the repository requires authentication:

```bash
uv add "digintlab-langfuse-python @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.3.0"
```

With pip:

```bash
pip install "digintlab-langfuse-python @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.3.0"
```

Alternatively, each [GitHub release](https://github.com/DigintLab/digintlab-langfuse-python/releases) attaches a wheel and a source distribution. Download the wheel and install the local file:

```bash
gh release download v0.3.0 -R DigintLab/digintlab-langfuse-python -p "*.whl"
uv add ./digintlab_langfuse_python-0.3.0-py3-none-any.whl
```

## Usage

```python
from digintlab_langfuse import LangfuseClientConfig, init_langfuse_telemetry

telemetry = init_langfuse_telemetry(
    LangfuseClientConfig(
        public_key=public_key,
        secret_key=secret_key,
        environment="production",
        release=release,
        redact_http_auth_headers=True,
    ),
    instrument_httpx=True,
)

async with telemetry:
    with telemetry.trace_call(
        name="agent-run",
        session_id=session_id,
        user_id=user_id,
        root_observation=True,
    ):
        await run_agent()
```

Scores use one typed API rather than type-specific convenience methods:

```python
from digintlab_langfuse.scores import NumericScore, ScoreTarget

telemetry.emit_score(
    score=NumericScore(name="quality", value=0.95),
    target=ScoreTarget(trace_id=trace_id),
)
```

Important behavior:

- Create one telemetry owner per process, after workers fork. Do not also construct `Langfuse(...)` directly.
- HTTPX and botocore instrumentation are opt-in. Instrumentation owned by other libraries is left untouched; initialization fails when HTTPX is externally instrumented and authorization-header redaction is requested because the redaction hooks cannot be guaranteed.
- Observation helpers fail open so telemetry errors do not replace application results or exceptions. Configuration, scoring, flush, and shutdown fail fast.
- Use `await telemetry.flush()` to export pending data without closing the owner. The async context manager shuts it down.
- Capture `telemetry.trace_runtime_context()` before sending work across a process or queue boundary. Pass it to `with_litellm_proxy_trace_metadata()` in the worker.
- Plain HTTP endpoints require `allow_insecure_http=True`.

## Sensitive data

PII redaction is optional:

```bash
uv add "digintlab-langfuse-python[redaction] @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.3.0"
```

With pip, pass the same requirement string to `pip install`.

```python
from digintlab_langfuse import LangfuseClientConfig, SensitiveDataRedactionConfig

config = LangfuseClientConfig(
    public_key=public_key,
    secret_key=secret_key,
    sensitive_data_redaction=SensitiveDataRedactionConfig(
        pii_entities=frozenset({"EMAIL_ADDRESS", "IP_ADDRESS", "PHONE_NUMBER"}),
        pii_score_threshold=0.4,
        redact_bearer_tokens=True,
    ),
)
```

The policy masks common credential fields and bearer tokens, then uses Presidio's local rule-based recognizers for PII. It uses Presidio's bundled domain-suffix snapshot and does not load an NLP model, call an LLM, or make network requests. Configure `pii_entities` and `pii_score_threshold` for the data you expect; use `mask` and `mask_otel_spans` for a fully custom policy.

> [!WARNING]
> Do not put sensitive data in span names, identifiers, links, events, or resource attributes. OpenTelemetry export hooks cannot rewrite them.
>
> `SensitiveDataRedactionConfig` also does not sanitize metadata forwarded to LiteLLM. Keep runtime-context, run-manager, and request metadata non-sensitive,
> or sanitize it before calling `with_litellm_proxy_trace_metadata()`.

HTTP authorization-header redaction is separate. Enable both `instrument_httpx=True` and `redact_http_auth_headers=True`. Tool callbacks capture inputs and outputs unless you provide `summarize_output` or enable masking.

## Development

```bash
task install
task format
task check
task test
```

Run the authenticated E2E suite only against a disposable Langfuse project:

```bash
RUN_LANGFUSE_E2E=1 \
LANGFUSE_E2E_PUBLIC_KEY=... \
LANGFUSE_E2E_SECRET_KEY=... \
LANGFUSE_E2E_BASE_URL=https://cloud.langfuse.com \
task test-e2e
```

## License

This package is licensed under the GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`). You may use it in any application, including proprietary applications that import the library unmodified. If you distribute a modified version of the library itself, those modifications must be made available under the LGPL. See [`COPYING.LESSER`](COPYING.LESSER) for the LGPL terms and [`COPYING`](COPYING) for the GNU GPL v3 text it builds on.
