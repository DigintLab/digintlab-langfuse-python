"""Production lifecycle, privacy, and bounded-resource contracts."""

import asyncio
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from langfuse.types import MaskOtelSpansParams, OtelSpanData, OtelSpanIdentifier

from digintlab_langfuse import _lifecycle as lifecycle_module
from digintlab_langfuse._lifecycle import flush_client, shutdown_client
from digintlab_langfuse.budgets import check_numeric_budgets, validate_numeric_budget_ceilings
from digintlab_langfuse.callbacks import ToolObservationResult, ToolSpanCallback
from digintlab_langfuse.client import (
    LangfuseClientConfig,
    LangfuseTelemetry,
    _init_langfuse_client,
    init_langfuse_telemetry,
)
from digintlab_langfuse.json_values import JsonValue, to_json_value
from digintlab_langfuse.litellm import LangfuseTraceClient, with_litellm_proxy_trace_metadata
from digintlab_langfuse.redaction import SensitiveDataRedactionConfig, create_sensitive_data_otel_mask
from digintlab_langfuse.timing import SpanTimingProcessor


class FakeInstrumentor:
    """Small process-instrumentor double with explicit global state."""

    def __init__(
        self,
        *,
        instrumented: bool = False,
        instrument_error: Exception | None = None,
        uninstrument_error: Exception | None = None,
    ) -> None:
        self.is_instrumented_by_opentelemetry = instrumented
        self.instrument_error = instrument_error
        self.uninstrument_error = uninstrument_error
        self.instrument_calls = 0
        self.uninstrument_calls = 0

    def instrument(self, **_kwargs: object) -> None:
        self.instrument_calls += 1
        if self.instrument_error is not None:
            raise self.instrument_error
        self.is_instrumented_by_opentelemetry = True

    def uninstrument(self) -> None:
        self.uninstrument_calls += 1
        if self.uninstrument_error is not None:
            raise self.uninstrument_error
        self.is_instrumented_by_opentelemetry = False


def _telemetry(
    *,
    instrument_httpx: bool = True,
    instrument_botocore: bool = False,
    redact_http_auth_headers: bool = False,
) -> LangfuseTelemetry:
    return LangfuseTelemetry(
        MagicMock(),
        public_key=f"pk-{uuid4()}",
        instrument_httpx=instrument_httpx,
        instrument_botocore=instrument_botocore,
        redact_http_auth_headers=redact_http_auth_headers,
    )


def test_process_instrumentation_does_not_uninstall_external_owner() -> None:
    httpx = FakeInstrumentor(instrumented=True)
    with patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx):
        telemetry = _telemetry()
        telemetry.instrument()
        telemetry.uninstrument()

    assert httpx.instrument_calls == 0
    assert httpx.uninstrument_calls == 0


def test_process_instrumentation_rejects_external_httpx_owner_when_redaction_is_required() -> None:
    httpx = FakeInstrumentor(instrumented=True)

    with (
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx),
        pytest.raises(RuntimeError, match="authorization-header redaction"),
    ):
        _telemetry(redact_http_auth_headers=True).instrument()

    assert httpx.instrument_calls == 0
    assert httpx.uninstrument_calls == 0


def test_process_instrumentation_is_reference_counted_between_telemetry_instances() -> None:
    httpx = FakeInstrumentor()
    with patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx):
        first = _telemetry()
        second = _telemetry()
        first.instrument()
        second.instrument()

        first.uninstrument()
        assert httpx.uninstrument_calls == 0

        second.uninstrument()

    assert httpx.instrument_calls == 1
    assert httpx.uninstrument_calls == 1


def test_telemetry_instrumentation_lifecycle_is_thread_safe() -> None:
    teardown_started = Event()
    allow_teardown = Event()

    class SlowInstrumentor(FakeInstrumentor):
        def uninstrument(self) -> None:
            teardown_started.set()
            allow_teardown.wait(timeout=5)
            super().uninstrument()

    httpx = SlowInstrumentor()
    with patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx):
        telemetry = _telemetry()
        telemetry.instrument()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(telemetry.uninstrument)
            assert teardown_started.wait(timeout=1)
            second = executor.submit(telemetry.uninstrument)
            time.sleep(0.02)
            allow_teardown.set()
            first.result(timeout=1)
            second.result(timeout=1)

    assert httpx.uninstrument_calls == 1


def test_process_instrumentation_rolls_back_partial_initialization() -> None:
    botocore = FakeInstrumentor()
    httpx = FakeInstrumentor(instrument_error=RuntimeError("httpx setup failed"))
    with (
        patch("digintlab_langfuse._instrumentation.BotocoreInstrumentor", return_value=botocore),
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx),
        pytest.raises(RuntimeError, match="httpx setup failed"),
    ):
        _telemetry(instrument_botocore=True).instrument()

    assert botocore.instrument_calls == 1
    assert botocore.uninstrument_calls == 1


def test_failed_process_instrumentation_rollback_can_be_adopted_and_released() -> None:
    botocore = FakeInstrumentor(uninstrument_error=RuntimeError("rollback failed"))
    httpx = FakeInstrumentor(instrument_error=RuntimeError("httpx setup failed"))
    with (
        patch("digintlab_langfuse._instrumentation.BotocoreInstrumentor", return_value=botocore),
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx),
        pytest.raises(RuntimeError, match="httpx setup failed"),
    ):
        _telemetry(instrument_botocore=True).instrument()

    botocore.uninstrument_error = None
    with patch("digintlab_langfuse._instrumentation.BotocoreInstrumentor", return_value=botocore):
        successor = _telemetry(instrument_httpx=False, instrument_botocore=True)
        successor.instrument()
        successor.uninstrument()

    assert botocore.instrument_calls == 1
    assert botocore.uninstrument_calls == 2


def test_client_initialization_does_not_mutate_process_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "operator-public-key")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "operator-secret-key")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://operator.example")
    before = dict(os.environ)
    config = LangfuseClientConfig(public_key=f"pk-{uuid4()}", secret_key="secret", base_url="https://langfuse.example")

    with patch("digintlab_langfuse.client.Langfuse"):
        _init_langfuse_client(config)

    assert dict(os.environ) == before


def test_client_initialization_rejects_duplicate_process_owner() -> None:
    public_key = f"pk-{uuid4()}"
    config = LangfuseClientConfig(public_key=public_key, secret_key="secret")

    with patch("digintlab_langfuse.client.Langfuse"):
        _init_langfuse_client(config)
        with pytest.raises(RuntimeError, match="already been initialized"):
            _init_langfuse_client(config)


def test_client_initialization_rejects_a_second_project_in_one_process() -> None:
    with patch("digintlab_langfuse.client.Langfuse"):
        _init_langfuse_client(LangfuseClientConfig(public_key="project-one", secret_key="secret"))
        with pytest.raises(RuntimeError, match="one Langfuse project"):
            _init_langfuse_client(LangfuseClientConfig(public_key="project-two", secret_key="secret"))


def test_telemetry_initialization_shuts_down_client_after_instrumentation_failure() -> None:
    client = MagicMock()
    httpx = FakeInstrumentor(instrument_error=RuntimeError("httpx setup failed"))
    config = LangfuseClientConfig(public_key=f"pk-{uuid4()}", secret_key="secret")

    with (
        patch("digintlab_langfuse.client._init_langfuse_client", return_value=client),
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx),
        pytest.raises(RuntimeError, match="httpx setup failed"),
    ):
        init_langfuse_telemetry(config, instrument_httpx=True)

    client.shutdown.assert_called_once_with()


def test_failed_initialization_shutdown_respects_the_configured_deadline() -> None:
    cleanup_started = Event()
    cleanup_finished = Event()
    owner_released = Event()
    release_cleanup = Event()

    def blocking_shutdown() -> None:
        cleanup_started.set()
        release_cleanup.wait(timeout=5)
        cleanup_finished.set()

    client = MagicMock()
    client.shutdown.side_effect = blocking_shutdown
    httpx = FakeInstrumentor(instrument_error=RuntimeError("httpx setup failed"))
    config = LangfuseClientConfig(
        public_key=f"pk-{uuid4()}",
        secret_key="secret",
        shutdown_timeout_seconds=0.01,
    )

    with (
        patch("digintlab_langfuse.client._init_langfuse_client", return_value=client),
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor", return_value=httpx),
        patch("digintlab_langfuse._lifecycle.release_failed_owner", side_effect=lambda _public_key: owner_released.set()),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        initialization = executor.submit(init_langfuse_telemetry, config, instrument_httpx=True)
        try:
            with pytest.raises(RuntimeError, match="httpx setup failed"):
                initialization.result(timeout=0.5)
            assert cleanup_started.is_set()
        finally:
            release_cleanup.set()

        assert cleanup_finished.wait(timeout=1)
        assert owner_released.wait(timeout=1)


def test_telemetry_initialization_shuts_down_client_after_owner_construction_failure() -> None:
    client = MagicMock()
    config = LangfuseClientConfig(public_key=f"pk-{uuid4()}", secret_key="secret")
    message = "callback setup failed"

    with (
        patch("digintlab_langfuse.client._init_langfuse_client", return_value=client),
        patch("digintlab_langfuse.client.LangfuseTelemetry", side_effect=RuntimeError(message)),
        pytest.raises(RuntimeError, match=message),
    ):
        init_langfuse_telemetry(config)

    client.shutdown.assert_called_once_with()


def test_telemetry_initialization_can_retry_after_owner_construction_failure() -> None:
    failed_client = MagicMock()
    succeeding_client = MagicMock()
    succeeding_owner = MagicMock()
    config = LangfuseClientConfig(public_key=f"pk-{uuid4()}", secret_key="secret")

    with (
        patch("digintlab_langfuse.client.Langfuse", return_value=failed_client),
        patch("digintlab_langfuse.client.LangfuseTelemetry", side_effect=RuntimeError("callback setup failed")),
        pytest.raises(RuntimeError, match="callback setup failed"),
    ):
        init_langfuse_telemetry(config)

    with (
        patch("digintlab_langfuse.client.Langfuse", return_value=succeeding_client),
        patch("digintlab_langfuse.client.LangfuseTelemetry", return_value=succeeding_owner),
    ):
        telemetry = init_langfuse_telemetry(config)

    assert telemetry is succeeding_owner
    failed_client.shutdown.assert_called_once_with()
    succeeding_owner.instrument.assert_called_once_with()


@pytest.mark.asyncio
async def test_telemetry_flush_and_async_context_own_the_client_lifecycle() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(
        client,
        public_key=f"pk-{uuid4()}",
        instrument_httpx=False,
        instrument_botocore=False,
    )

    async with telemetry:
        await telemetry.flush()

    client.flush.assert_called_once_with()
    client.shutdown.assert_called_once_with()

    await telemetry.shutdown()
    client.shutdown.assert_called_once_with()
    with pytest.raises(RuntimeError, match="after shutdown"):
        async with telemetry:
            pass


@pytest.mark.asyncio
async def test_timed_out_shutdown_does_not_start_a_concurrent_retry() -> None:
    started = Event()
    release = Event()
    call_lock = Lock()
    shutdown_calls = 0

    def blocking_shutdown() -> None:
        nonlocal shutdown_calls
        with call_lock:
            shutdown_calls += 1
        started.set()
        release.wait(timeout=5)

    client = MagicMock()
    client.shutdown.side_effect = blocking_shutdown
    telemetry = LangfuseTelemetry(
        client,
        public_key="pk-timeout",
        instrument_httpx=False,
        instrument_botocore=False,
        shutdown_timeout_seconds=0.01,
    )

    with pytest.raises(TimeoutError):
        await telemetry.shutdown()
    assert started.is_set()
    lifecycle_state = lifecycle_module.lifecycle_state(client)
    in_flight_shutdown = lifecycle_state.shutdown_future

    with pytest.raises(RuntimeError, match="after shutdown"):
        async with telemetry:
            pass

    telemetry._shutdown_timeout_seconds = 1
    retry = asyncio.create_task(telemetry.shutdown())
    try:
        await asyncio.sleep(0.02)
        assert shutdown_calls == 1
        assert lifecycle_state.shutdown_future is in_flight_shutdown
    finally:
        release.set()
    await retry
    assert shutdown_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [0.0, math.nan, math.inf])
async def test_invalid_lifecycle_timeout_does_not_start_an_operation(timeout_seconds: float) -> None:
    flush_target = MagicMock()
    shutdown_target = MagicMock()

    with pytest.raises(ValueError, match="finite"):
        await flush_client(flush_target, timeout_seconds=timeout_seconds)
    with pytest.raises(ValueError, match="finite"):
        await shutdown_client(shutdown_target, timeout_seconds=timeout_seconds)

    flush_target.flush.assert_not_called()
    shutdown_target.shutdown.assert_not_called()


@pytest.mark.asyncio
async def test_successful_shutdown_closes_the_lifecycle_executor() -> None:
    client = MagicMock()
    state = lifecycle_module.lifecycle_state(client)

    await shutdown_client(client)

    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        state.executor.submit(lambda: None)


def test_client_configuration_rejects_unsafe_or_invalid_values() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        LangfuseClientConfig(public_key="", secret_key="secret")
    with pytest.raises(ValueError, match="allow_insecure_http"):
        LangfuseClientConfig(public_key="public", secret_key="secret", base_url="http://langfuse.example")
    with pytest.raises(ValueError, match="flush_at"):
        LangfuseClientConfig(public_key="public", secret_key="secret", flush_at=0)
    with pytest.raises(ValueError, match="sample_rate"):
        LangfuseClientConfig(public_key="public", secret_key="secret", sample_rate=1.1)
    with pytest.raises(ValueError, match="timeouts"):
        LangfuseClientConfig(public_key="public", secret_key="secret", shutdown_timeout_seconds=0.0)
    for value in (math.nan, math.inf):
        with pytest.raises(ValueError, match="finite"):
            LangfuseClientConfig(public_key="public", secret_key="secret", flush_interval=value)
        with pytest.raises(ValueError, match="finite"):
            LangfuseClientConfig(public_key="public", secret_key="secret", flush_timeout_seconds=value)


def test_langfuse_endpoint_delegates_upstream_defaults_with_https_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_BASE_URL", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    config = LangfuseClientConfig(public_key="public", secret_key="secret")
    assert config.base_url is None

    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    with pytest.raises(ValueError, match="allow_insecure_http"):
        LangfuseClientConfig(public_key="public", secret_key="secret")
    assert LangfuseClientConfig(public_key="public", secret_key="secret", allow_insecure_http=True).base_url is None

    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://secure.example")
    assert LangfuseClientConfig(public_key="public", secret_key="secret").base_url is None

    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://untrusted.example")
    with pytest.raises(ValueError, match="allow_insecure_http"):
        LangfuseClientConfig(public_key="public", secret_key="secret")
    assert LangfuseClientConfig(public_key="public", secret_key="secret", base_url="https://explicit.example").base_url == "https://explicit.example"

    with pytest.raises(ValueError, match="absolute HTTP"):
        LangfuseClientConfig(public_key="public", secret_key="secret", base_url="")


def test_client_configuration_repr_does_not_expose_secret_key() -> None:
    config = LangfuseClientConfig(public_key="public", secret_key="super-secret")

    assert "super-secret" not in repr(config)


def test_sensitive_data_redaction_is_disabled_by_default_and_explicit_when_enabled() -> None:
    default_config = LangfuseClientConfig(public_key=f"pk-{uuid4()}", secret_key="secret")
    redacted_config = LangfuseClientConfig(
        public_key=f"pk-{uuid4()}",
        secret_key="secret",
        sensitive_data_redaction=SensitiveDataRedactionConfig(),
    )

    with (
        patch("digintlab_langfuse.client.Langfuse") as langfuse_cls,
        patch("digintlab_langfuse._lifecycle._initialized_public_keys", set()),
    ):
        _init_langfuse_client(default_config)
        default_mask = langfuse_cls.call_args.kwargs["mask"]
        default_otel_mask = langfuse_cls.call_args.kwargs["mask_otel_spans"]
    with (
        patch("digintlab_langfuse.client.Langfuse") as langfuse_cls,
        patch("digintlab_langfuse._lifecycle._initialized_public_keys", set()),
    ):
        _init_langfuse_client(redacted_config)
        redaction_mask = langfuse_cls.call_args.kwargs["mask"]
        redaction_otel_mask = langfuse_cls.call_args.kwargs["mask_otel_spans"]

    assert default_mask is None
    assert default_otel_mask is None
    assert callable(redaction_mask)
    assert callable(redaction_otel_mask)
    assert redaction_mask(
        data={
            "password": "correct horse battery staple",
            "contact": "person@example.com",
            "nested": {"api-key": "key-value"},
        }
    ) == {
        "password": "[REDACTED]",
        "contact": "[REDACTED]",
        "nested": {"api-key": "[REDACTED]"},
    }


def test_sensitive_data_replacement_is_treated_as_literal_text() -> None:
    from digintlab_langfuse.redaction import redact_sensitive_data

    config = SensitiveDataRedactionConfig(replacement=r"\g<0>")

    assert redact_sensitive_data("person@example.com", config) == r"\g<0>"


def test_otel_redaction_covers_structured_strings_and_common_credential_keys() -> None:
    identifier = OtelSpanIdentifier(trace_id="trace", span_id="span")
    mask = create_sensitive_data_otel_mask(SensitiveDataRedactionConfig())
    result = mask(
        params=MaskOtelSpansParams(
            spans={
                identifier: OtelSpanData(
                    trace_id="trace",
                    span_id="span",
                    parent_span_id=None,
                    name="provider-call",
                    instrumentation_scope_name="test",
                    instrumentation_scope_version=None,
                    attributes={
                        "payload": '{"password":"secret","email":"person@example.com"}',
                        "non_finite": '{"password":"secret","metric":NaN}',
                        "camel_case": '{"clientSecret":"secret","accessToken":"token"}',
                        "batch": ('{"password":"secret"}',),
                        "unmodified": "public",
                        "numerical": 42,
                    },
                    resource_attributes={},
                )
            }
        )
    )

    assert result is not None
    patch_result = result.span_patches[identifier]
    assert patch_result is not None
    assert patch_result.set_attributes == {
        "payload": '{"password":"[REDACTED]","email":"[REDACTED]"}',
        "non_finite": '{"password":"[REDACTED]","metric":"[REDACTED]"}',
        "camel_case": '{"clientSecret":"[REDACTED]","accessToken":"[REDACTED]"}',
        "batch": ('{"password":"[REDACTED]"}',),
    }


def test_sensitive_data_redaction_rejects_custom_mask_hooks() -> None:
    custom_mask = MagicMock()

    with pytest.raises(ValueError, match="sensitive_data_redaction"):
        LangfuseClientConfig(
            public_key="public",
            secret_key="secret",
            sensitive_data_redaction=SensitiveDataRedactionConfig(),
            mask_otel_spans=custom_mask,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_json_conversion_rejects_non_finite_floats(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        to_json_value(value)


def test_json_values_module_normalizes_plain_enum_values() -> None:
    from enum import Enum

    from digintlab_langfuse.json_values import to_json_value as normalize_json_value

    class Result(Enum):
        SUCCESS = "success"

    assert normalize_json_value({"result": Result.SUCCESS}) == {"result": "success"}


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_numeric_budgets_reject_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_numeric_budget_ceilings({"latency": value})
    with pytest.raises(ValueError, match="finite"):
        check_numeric_budgets({"latency": value}, {"latency": 1.0})


def test_span_timing_processor_bounds_dynamic_bucket_cardinality() -> None:
    processor = SpanTimingProcessor(max_buckets=2)
    spans = []
    for name in ("one", "two", "three", "four"):
        span = MagicMock()
        span.name = name
        span.instrumentation_scope = None
        span.start_time = 1
        span.end_time = 1_000_000_001
        spans.append(span)

    for span in spans:
        processor.on_end(span)

    buckets = processor.snapshot()
    assert [bucket.key for bucket in buckets] == ["__overflow__", "one", "two"]
    assert buckets[0].count == 2


def test_tool_callback_conversion_failure_is_fail_open() -> None:
    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-1"
    client.get_current_observation_id.return_value = "parent-1"

    def fail_conversion(_value: object) -> JsonValue:
        message = "conversion failed"
        raise RuntimeError(message)

    callback = ToolSpanCallback(client_getter=lambda: client, json_value_converter=fail_conversion)
    with patch("digintlab_langfuse.callbacks.logger", create=True) as logger:
        callback.on_tool_start({"name": "tool"}, "input", run_id=uuid4(), inputs={"secret": "value"})

    logger.exception.assert_called_once()
    client.start_observation.assert_not_called()


def test_tool_callback_closes_span_before_propagating_base_exception() -> None:
    client = MagicMock()
    client.get_current_trace_id.return_value = "trace-1"
    client.get_current_observation_id.return_value = "parent-1"
    span = MagicMock()
    client.start_observation.return_value = span

    def interrupt_summary(_value: object) -> ToolObservationResult:
        raise KeyboardInterrupt

    callback = ToolSpanCallback(client_getter=lambda: client, summarize_output=interrupt_summary)
    run_id = uuid4()
    callback.on_tool_start({"name": "tool"}, "input", run_id=run_id)

    with pytest.raises(KeyboardInterrupt):
        callback.on_tool_end("output", run_id=run_id)

    span.end.assert_called_once_with()


def test_litellm_context_lookup_failure_is_fail_open() -> None:
    kwargs: dict[str, JsonValue] = {"model": "provider/model"}

    def fail_client_lookup() -> LangfuseTraceClient:
        message = "client lookup failed"
        raise RuntimeError(message)

    with patch("digintlab_langfuse.litellm.logger", create=True) as logger:
        result = with_litellm_proxy_trace_metadata(kwargs, client_getter=fail_client_lookup)

    assert result is kwargs
    logger.exception.assert_called_once()
