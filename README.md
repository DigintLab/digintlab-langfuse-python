# digintlab-langfuse-python

Typed Python helpers for Langfuse tracing, scoring, callbacks, and LiteLLM metadata in DigIntLab agent runtimes.

Contents: [Install](#install) | [Usage](#usage) | [Configuration](#configuration) | [Modules](#modules) | [Sensitive data](#sensitive-data) | [Limitations](#limitations) | [Examples](#examples) | [Development](#development) | [API stability](#api-stability) | [License](#license)

## Install

Requires Python 3.13, 3.14, or 3.15. Langfuse is constrained to the patch releases of the minor version tested by this package.

The package is not published on PyPI. Install it from a tagged GitHub release (replace `v0.4.0` with the release you want):

```bash
uv add "digintlab-langfuse-python @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.4.0"
```

With pip:

```bash
pip install "digintlab-langfuse-python @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.4.0"
```

Each [GitHub release](https://github.com/DigintLab/digintlab-langfuse-python/releases) also attaches a wheel and a source distribution, which can be installed directly:

```bash
uv add "digintlab-langfuse-python @ https://github.com/DigintLab/digintlab-langfuse-python/releases/download/v0.4.0/digintlab_langfuse_python-0.4.0-py3-none-any.whl"
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

## Configuration

`LangfuseClientConfig` is a frozen dataclass. Only `public_key` and `secret_key` are required. Invalid values raise `ValueError` at construction, before any SDK resource is allocated.

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `public_key` | `str` | required | Langfuse project public key; must not be blank. |
| `secret_key` | `str` | required | Langfuse project secret key; must not be blank; hidden from `repr`. |
| `base_url` | `str \| None` | `None` | Langfuse host; must be an absolute HTTP(S) URL. When `None`, the `LANGFUSE_BASE_URL` or `LANGFUSE_HOST` environment variable is validated and used by the SDK. |
| `timeout` | `int \| None` | `None` | SDK request timeout; must be greater than zero. |
| `environment` | `str \| None` | `None` | Langfuse environment label. |
| `release` | `str \| None` | `None` | Release identifier attached to traces. |
| `sample_rate` | `float \| None` | `None` | Trace sampling rate between 0 and 1. |
| `flush_at` | `int \| None` | `None` | Batch size that triggers an export; at least 1. |
| `flush_interval` | `float \| None` | `None` | Seconds between exports; finite and greater than zero. |
| `debug` | `bool` | `False` | Enable Langfuse SDK debug logging. |
| `mask` | `MaskFunction \| None` | `None` | Custom Langfuse data mask. Mutually exclusive with `sensitive_data_redaction`. |
| `mask_otel_spans` | `MaskOtelSpansFunction \| None` | `None` | Custom export-stage span mask. Mutually exclusive with `sensitive_data_redaction`. |
| `httpx_client` | `httpx.Client \| None` | `None` | HTTP client the SDK uses to reach Langfuse. |
| `should_export_span` | `Callable[[ReadableSpan], bool] \| None` | `None` | Filter deciding which spans are exported. |
| `tracer_provider` | `TracerProvider \| None` | `None` | OpenTelemetry provider for the SDK and for opt-in instrumentation. |
| `span_exporter` | `SpanExporter \| None` | `None` | Replacement span exporter, for example an in-memory exporter in tests. |
| `redact_http_auth_headers` | `bool` | `False` | Redact the `Authorization` header on HTTPX spans; needs `instrument_httpx=True`. |
| `sensitive_data_redaction` | `SensitiveDataRedactionConfig \| None` | `None` | Install the built-in redaction policy as both mask hooks. |
| `allow_insecure_http` | `bool` | `False` | Permit a plain `http://` endpoint. |
| `flush_timeout_seconds` | `float` | `30.0` | Time limit for `flush()`; finite and greater than zero. |
| `shutdown_timeout_seconds` | `float` | `30.0` | Time limit for `shutdown()`; finite and greater than zero. |

`init_langfuse_telemetry(config, *, instrument_httpx=False, instrument_botocore=False)` creates the client and returns a `LangfuseTelemetry`.

## Modules

Every name below is in the module's `__all__`. Examples assume `telemetry` from the Usage section.

### `client` and top-level exports

`LangfuseTelemetry` owns the SDK client (`telemetry.client`) plus optional instrumentation. Use it as an async context manager, or call `await telemetry.flush()` and `await telemetry.shutdown()`. `trace_call` propagates session and user ids and, with `root_observation=True`, opens a root observation of kind `root_kind` (default `ObservationKind.AGENT`). It also accepts `tags`, `metadata`, `version`, `trace_name`, `trace_id`, `parent_span_id`, `root_input`, and `root_metadata`. Other methods wrap the module functions below with this owner's client: `observe_node_span`, `emit_guardrail_observation`, `tool_span_callback`, `langchain_callbacks`, `with_litellm_proxy_trace_metadata`, `trace_runtime_context`, `emit_score`, `score_current_trace`, `score_current_span`, `instrument`, and `uninstrument`. There is no separate `trace_scope`; `trace_call` is the trace scope.

```python
from digintlab_langfuse import ObservationKind

with telemetry.trace_call(
    name="ingest",
    session_id="s-1",
    user_id="u-1",
    root_observation=True,
    root_kind=ObservationKind.CHAIN,
    tags=["batch"],
):
    ...
```

### `spans`

Observation helpers over context-local active-parent state: `observe_node_span`, `observe_generation`, `emit_guardrail_observation`, and `current_trace_id`. `trace_runtime_context` returns a `TraceRuntimeContext` for work that crosses a process or queue boundary. `scrub_auth_header` and `scrub_auth_header_async` are the HTTPX hooks behind `redact_http_auth_headers`. All helpers open observations only inside an active trace.

```python
from digintlab_langfuse.spans import ObservationKind, observe_generation, observe_node_span

with observe_node_span("retrieve", enabled=True, kind=ObservationKind.RETRIEVER, client_getter=lambda: telemetry.client):
    with observe_generation("answer", model="my-model", client_getter=lambda: telemetry.client) as generation:
        if generation is not None:
            generation.update(output="done")
```

### `callbacks`

`ToolSpanCallback` is a LangChain callback handler that records each tool call as a Langfuse tool observation under the active node span. `summarize_output` returns a `ToolObservationResult` (`output`, `level`, `status_message`) to control what is stored. The tool input is always captured.

```python
from digintlab_langfuse.callbacks import ToolObservationResult


def summarize(output: object) -> ToolObservationResult:
    return {"output": {"status": "ok", "chars": len(str(output))}}


callback = telemetry.tool_span_callback(summarize_output=summarize)
```

### `litellm`

`with_litellm_proxy_trace_metadata` merges the active trace ids into `extra_body.metadata` of LiteLLM proxy kwargs, and returns the kwargs unchanged when no trace is active. Pass `trace_context=` to use a captured `TraceRuntimeContext` instead of the active trace. It raises `TypeError` if `extra_body` is not a dict.

```python
from digintlab_langfuse.litellm import with_litellm_proxy_trace_metadata

kwargs = with_litellm_proxy_trace_metadata({"model": "my-model"}, trace_context=telemetry.trace_runtime_context())
```

### `scores`

One typed API for all score kinds: `NumericScore`, `BooleanScore`, `CategoricalScore`, `TextScore`, and `CorrectionScore`. `ScoreTarget` selects the trace, observation, or session (`run_id`, `trace_id`, `observation_id`, `session_id`, `dataset_run_id`, `score_id`, `comment`, `config_id`, `metadata`). `score_elapsed` measures a block and emits its seconds as a numeric score.

```python
from digintlab_langfuse.scores import BooleanScore, ScoreTarget, emit_score, score_elapsed

emit_score(telemetry.client, score=BooleanScore(name="passed", value=True), target=ScoreTarget(trace_id=trace_id))
with score_elapsed(telemetry.client, name="latency_seconds", target=ScoreTarget(trace_id=trace_id), digits=3):
    ...
```

### `prompts`

Typed requests for fetching and creating managed prompts. `TextPromptRequest` and `ChatPromptRequest` take `name`, `version`, `label`, `cache_ttl_seconds`, `fallback`, `max_retries`, and `fetch_timeout_seconds`. `TextManagedPrompt` and `ChatManagedPrompt` take `name`, `prompt`, `labels`, `tags`, `config`, and `commit_message`. `create_or_update_prompt` creates a new prompt version.

```python
from digintlab_langfuse.prompts import TextManagedPrompt, TextPromptRequest, create_or_update_prompt, get_prompt

create_or_update_prompt(telemetry.client, TextManagedPrompt(name="greeting", prompt="Hello {{name}}", labels=("production",)))
prompt = get_prompt(telemetry.client, TextPromptRequest(name="greeting", label="production"))
```

### `experiments`

Thin helpers over Langfuse datasets and local experiments. `run_local_experiment` normalizes items to JSON values and takes `evaluators`, `run_evaluators`, `max_concurrency` (default 50), `metadata`, and `experiment_id`.

```python
from digintlab_langfuse.experiments import DatasetConfig, create_dataset, run_local_experiment

create_dataset(telemetry.client, DatasetConfig(name="smoke"))
result = run_local_experiment(
    telemetry.client,
    name="smoke-run",
    data=[{"input": "2+2", "expected_output": "4"}],
    task=lambda *, item, **kwargs: "4",
)
```

`create_dataset_item` and the `ExperimentItem` type are also exported.

### `budgets`

Numeric ceiling checks for CI-style gates. `check_numeric_budgets` returns a `NumericBudgetViolation` for each missing or exceeded measurement. `validate_numeric_budget_ceilings` validates raw ceilings (optionally against `allowed_names`) and rejects non-numeric or unknown entries.

```python
from digintlab_langfuse.budgets import check_numeric_budgets, validate_numeric_budget_ceilings

ceilings = validate_numeric_budget_ceilings({"p95_seconds": 2.0})
for violation in check_numeric_budgets({"p95_seconds": 3.1}, ceilings):
    print(violation.name, violation.message)  # p95_seconds 3.1 > 2.0
```

### `timing`

`elapsed_timer` times a local block. `create_span_timing_recorder` adds a `SpanTimingProcessor` to a tracer provider (a new one by default) and aggregates ended spans into `SpanTimingBucket` values keyed by `default_span_timing_key` or your own `key`. Beyond `max_buckets` (default 1000), new keys fold into `__overflow__`. Pass `recorder.tracer_provider` as `LangfuseClientConfig.tracer_provider` to record Langfuse spans.

```python
from digintlab_langfuse.timing import create_span_timing_recorder, elapsed_timer

with elapsed_timer() as timer:
    ...
print(timer.rounded(3))

recorder = create_span_timing_recorder()
with recorder.tracer_provider.get_tracer("demo").start_as_current_span("work"):
    ...
buckets = recorder.snapshot(reset=True)
```

### `redaction`

Opt-in masking built from a `SensitiveDataRedactionConfig` (`replacement="[REDACTED]"`, `sensitive_field_names`, `pii_entities`, `pii_score_threshold=0.4`, `redact_bearer_tokens=True`). `create_sensitive_data_mask` and `create_sensitive_data_otel_mask` build the two Langfuse hooks; `redact_sensitive_data` applies the policy to any value directly. See [Sensitive data](#sensitive-data).

```python
from digintlab_langfuse import SensitiveDataRedactionConfig
from digintlab_langfuse.redaction import redact_sensitive_data

policy = SensitiveDataRedactionConfig(pii_entities=frozenset({"EMAIL_ADDRESS"}))
redact_sensitive_data({"password": "hunter2", "note": "mail a@example.com"}, policy)
# {"password": "[REDACTED]", "note": "mail [REDACTED]"}
```

### `json_values`

`JsonValue` is the recursive JSON type used across the API. `to_json_value` converts enums, mappings, sequences, and other objects (via `str`) into it, and raises `ValueError` for non-finite floats.

```python
from digintlab_langfuse.json_values import JsonValue, to_json_value

payload: JsonValue = to_json_value({"count": 3, "tags": ("a", "b")})
```

## Sensitive data

PII redaction is optional:

```bash
uv add "digintlab-langfuse-python[redaction] @ git+https://github.com/DigintLab/digintlab-langfuse-python@v0.4.0"
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

HTTP authorization-header redaction is separate. Enable both `instrument_httpx=True` and `redact_http_auth_headers=True`. Tool callbacks always capture the tool input and capture the output unless you provide `summarize_output`; enable masking to cover both.

## Limitations

- **One Langfuse project per process.** A second `init_langfuse_telemetry()` raises `RuntimeError`, whether it reuses the public key or uses another project's. It also raises when a Langfuse SDK resource was created outside this package. `shutdown()` does not release this ownership, so initialize once per process lifetime. Run separate projects in separate processes.
- **Langfuse is limited to one minor version** (`langfuse>=5.2.0,<5.3`). The package reaches into SDK internals, which `tests/package_contract/test_langfuse_internals.py` checks on every dependency update, so a new Langfuse minor release is only allowed after review. Do not override the range.
- **Private SDK internals are used.** Ownership checks and lifecycle handling read `LangfuseResourceManager` state (`_instances`, `_lock`) and the client's `_resources`. `ToolSpanCallback` is typed against `LangfuseObservationWrapper` from `langfuse._client.span`.
- **Observation helpers fail open and no-op quietly.** Errors in `trace_call`, `observe_node_span`, `observe_generation`, `emit_guardrail_observation`, `ToolSpanCallback`, and `with_litellm_proxy_trace_metadata` are logged and never replace your result or exception. Outside an active trace these helpers record nothing. Configuration, scoring, `flush`, and `shutdown` fail fast.
- **Instrumentation is process-wide.** HTTPX and botocore instrumentors are shared by every owner in the process and must use the same tracer provider and redaction setting, or initialization raises. Instrumentation installed by other code is left in place, and HTTPX initialization fails if that code owns it and `redact_http_auth_headers=True`. `shutdown()` uninstruments only what this package installed.
- **Redaction has limits.**
  - It runs through Langfuse's `mask` and `mask_otel_spans` hooks, so it covers masked data and string span attributes only. Span names, identifiers, links, events, and resource attributes are not rewritten.
  - Metadata forwarded to LiteLLM is not sanitized.
  - Matching is by field name (normalized, so `apiKey` matches `api_key`), bearer-token pattern, and Presidio's English rule-based recognizers. There is no NLP model, so names and addresses are not detected, and non-English text is not analyzed.
  - Secrets in free text without a recognized field name or bearer prefix are not masked.
  - `redact_http_auth_headers` only masks `Authorization`; other headers such as cookies or API-key headers are not scrubbed.
  - The default `pii_entities` apply when you enable `sensitive_data_redaction`; pass an empty set to skip PII analysis and the `redaction` extra.

## Examples

Runnable scripts live in [`examples/`](examples). Each reads `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and `LANGFUSE_BASE_URL`, and exits with a message if one is unset:

- [`basic_tracing.py`](examples/basic_tracing.py): trace one call and attach numeric, categorical, and elapsed-time scores.
- [`langchain_tool_callback.py`](examples/langchain_tool_callback.py): record a LangChain tool call as a tool observation.
- [`redaction.py`](examples/redaction.py): enable the built-in redaction policy.

```bash
uv run python examples/basic_tracing.py
```

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

## API stability

The public API is the names listed in each public module's `__all__` plus the top-level `digintlab_langfuse` exports. Modules and names that start with an underscore are private.

Versioning follows [SemVer](https://semver.org/). Before 1.0, a minor version (0.x to 0.(x+1)) may contain breaking changes, which are listed in `CHANGELOG.md` under "Changed" or "Removed". Patch versions are backward compatible.

Langfuse is limited to the patch releases of one tested minor version. Moving to a new Langfuse minor version is released as at least a minor version of this package.

## License

This package is licensed under the GNU Lesser General Public License v3.0 or later (`LGPL-3.0-or-later`). You may use it in any application, including proprietary applications that import the library unmodified. If you distribute a modified version of the library itself, those modifications must be made available under the LGPL. See [`COPYING.LESSER`](COPYING.LESSER) for the LGPL terms and [`COPYING`](COPYING) for the GNU GPL v3 text it builds on.
