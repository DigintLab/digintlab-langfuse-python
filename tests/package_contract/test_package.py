"""Standalone package contract tests for digintlab_langfuse."""

import ast
import asyncio
from pathlib import Path
from typing import assert_type
from unittest.mock import MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from langfuse import Langfuse
from langfuse.api.commons.types import ObservationLevel
from langfuse.model import ChatPromptClient, TextPromptClient
from opentelemetry.instrumentation.httpx import RequestInfo
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import digintlab_langfuse
from digintlab_langfuse import client as client_module
from digintlab_langfuse.budgets import NumericBudgetViolation, check_numeric_budgets, validate_numeric_budget_ceilings
from digintlab_langfuse.callbacks import ToolObservationResult, ToolSpanCallback
from digintlab_langfuse.client import LangfuseClientConfig, LangfuseTelemetry, _init_langfuse_client, init_langfuse_telemetry
from digintlab_langfuse.experiments import (
    DatasetConfig,
    create_dataset,
    create_dataset_item,
    run_local_experiment,
)
from digintlab_langfuse.json_values import JsonValue
from digintlab_langfuse.litellm import with_litellm_proxy_trace_metadata
from digintlab_langfuse.prompts import (
    ChatPromptRequest,
    PromptKind,
    TextManagedPrompt,
    create_or_update_prompt,
    get_prompt,
)
from digintlab_langfuse.redaction import SensitiveDataRedactionConfig
from digintlab_langfuse.scores import BooleanScore, NumericScore, ScoreTarget, emit_score, score_elapsed
from digintlab_langfuse.spans import (
    ObservationKind,
    TraceRuntimeContext,
    active_parent_observation_id,
    emit_guardrail_observation,
    observe_generation,
    observe_node_span,
    scrub_auth_header,
    scrub_auth_header_async,
    trace_runtime_context,
)
from digintlab_langfuse.timing import SpanTimingProcessor, create_span_timing_recorder, elapsed_timer

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "digintlab_langfuse"
EXPECTED_ROOT_EXPORTS = [
    "LangfuseClientConfig",
    "LangfuseTelemetry",
    "ObservationKind",
    "SensitiveDataRedactionConfig",
    "TraceRuntimeContext",
    "init_langfuse_telemetry",
]


def test_package_does_not_import_other_local_packages() -> None:
    """The reusable library must not depend on sibling application packages."""
    local_package_roots = frozenset(path.name for path in PACKAGE_ROOT.parent.iterdir() if path.is_dir() and path.name != PACKAGE_ROOT.name)
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.partition(".")[0] in local_package_roots:
                offenders.append(f"{path.name}: from {node.module} import ...")
            if isinstance(node, ast.Import):
                offenders.extend(f"{path.name}: import {alias.name}" for alias in node.names if alias.name.partition(".")[0] in local_package_roots)

    assert offenders == []


def test_root_exports_only_public_library_surface() -> None:
    assert set(digintlab_langfuse.__all__) == set(EXPECTED_ROOT_EXPORTS)
    assert set(client_module.__all__) == {"LangfuseClientConfig", "LangfuseTelemetry", "init_langfuse_telemetry"}


def test_root_reexports_primary_telemetry_api() -> None:
    assert digintlab_langfuse.LangfuseClientConfig is LangfuseClientConfig
    assert digintlab_langfuse.LangfuseTelemetry is LangfuseTelemetry
    assert digintlab_langfuse.ObservationKind is ObservationKind
    assert digintlab_langfuse.SensitiveDataRedactionConfig is SensitiveDataRedactionConfig
    assert digintlab_langfuse.TraceRuntimeContext is TraceRuntimeContext
    assert digintlab_langfuse.init_langfuse_telemetry is init_langfuse_telemetry


def test_validate_numeric_budget_ceilings_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="Unknown budget ceiling names"):
        validate_numeric_budget_ceilings({"known": 1, "typo": 2}, allowed_names={"known"})


@pytest.mark.parametrize("value", [True, "1"])
def test_validate_numeric_budget_ceilings_rejects_non_numeric_values(value: bool | str) -> None:
    with pytest.raises(TypeError, match="must be a number"):
        validate_numeric_budget_ceilings({"max_runtime": value})


def test_check_numeric_budgets_reports_exceeded_and_missing_measurements() -> None:
    violations = check_numeric_budgets(
        {"latency_seconds": 12.5},
        {"latency_seconds": 10.0, "cost_usd": 1.0},
    )

    assert violations == (
        NumericBudgetViolation(name="latency_seconds", ceiling=10.0, observed=12.5),
        NumericBudgetViolation(name="cost_usd", ceiling=1.0),
    )
    assert [violation.message for violation in violations] == ["12.5 > 10.0", "cost_usd missing"]


def test_client_config_forwards_langfuse_runtime_options() -> None:
    def should_export_span(_span: ReadableSpan) -> bool:
        return True

    config = LangfuseClientConfig(
        public_key="pk-config",
        secret_key="sk",
        base_url="https://langfuse.example",
        timeout=12,
        environment="staging",
        release="2026.06.21",
        sample_rate=0.25,
        flush_at=20,
        flush_interval=3.5,
        debug=True,
        should_export_span=should_export_span,
    )

    with patch("digintlab_langfuse.client.Langfuse") as langfuse_cls:
        _init_langfuse_client(config)

    langfuse_cls.assert_called_once_with(
        public_key="pk-config",
        secret_key="sk",
        base_url="https://langfuse.example",
        timeout=12,
        debug=True,
        environment="staging",
        release="2026.06.21",
        flush_at=20,
        flush_interval=3.5,
        sample_rate=0.25,
        mask=None,
        mask_otel_spans=None,
        httpx_client=None,
        should_export_span=should_export_span,
        tracer_provider=None,
        span_exporter=None,
    )


def test_httpx_auth_redaction_is_disabled_by_default() -> None:
    client = MagicMock()
    with patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor") as httpx_cls:
        httpx_cls.return_value.is_instrumented_by_opentelemetry = False
        telemetry = LangfuseTelemetry(
            client,
            public_key="pk",
            instrument_httpx=True,
            instrument_botocore=False,
        )
        telemetry.instrument()
        telemetry.uninstrument()

    httpx_cls.return_value.instrument.assert_called_once_with()


def test_httpx_auth_redaction_covers_sync_and_async_clients_when_enabled() -> None:
    client = MagicMock()
    with patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor") as httpx_cls:
        httpx_cls.return_value.is_instrumented_by_opentelemetry = False
        telemetry = LangfuseTelemetry(
            client,
            public_key="pk",
            instrument_httpx=True,
            instrument_botocore=False,
            redact_http_auth_headers=True,
        )
        telemetry.instrument()
        telemetry.uninstrument()

    httpx_cls.return_value.instrument.assert_called_once_with(
        request_hook=scrub_auth_header,
        async_request_hook=scrub_auth_header_async,
    )


@pytest.mark.asyncio
async def test_httpx_auth_redaction_accepts_instrumentor_request_info() -> None:
    span = MagicMock()
    span.is_recording.return_value = True
    request = RequestInfo(
        method=b"GET",
        url=httpx.URL("https://example.test"),
        headers=httpx.Headers({"Authorization": "Bearer secret"}),
        stream=None,
        extensions={},
    )

    scrub_auth_header(span, request)
    await scrub_auth_header_async(span, request)

    assert span.set_attribute.call_count == 2
    span.set_attribute.assert_called_with("http.request.header.authorization", "[REDACTED]")

    request_without_headers = RequestInfo(
        method=b"GET",
        url=httpx.URL("https://example.test"),
        headers=None,
        stream=None,
        extensions={},
    )
    scrub_auth_header(span, request_without_headers)


def test_init_instruments_selected_project_hooks() -> None:
    client = MagicMock()
    with (
        patch("digintlab_langfuse.client._init_langfuse_client", return_value=client),
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor") as httpx_instrumentor,
        patch("digintlab_langfuse._instrumentation.BotocoreInstrumentor") as botocore_instrumentor,
    ):
        httpx_instrumentor.return_value.is_instrumented_by_opentelemetry = False
        botocore_instrumentor.return_value.is_instrumented_by_opentelemetry = False
        telemetry = init_langfuse_telemetry(
            LangfuseClientConfig(public_key="pk-init", secret_key="sk"),
            instrument_httpx=True,
            instrument_botocore=True,
        )
        telemetry.uninstrument()

    assert telemetry.client is client
    httpx_instrumentor.return_value.instrument.assert_called_once()
    botocore_instrumentor.return_value.instrument.assert_called_once()


def test_configured_tracer_provider_reaches_process_instrumentors() -> None:
    client = MagicMock()
    tracer_provider = TracerProvider()
    with (
        patch("digintlab_langfuse.client._init_langfuse_client", return_value=client),
        patch("digintlab_langfuse._instrumentation.HTTPXClientInstrumentor") as httpx_instrumentor,
        patch("digintlab_langfuse._instrumentation.BotocoreInstrumentor") as botocore_instrumentor,
    ):
        httpx_instrumentor.return_value.is_instrumented_by_opentelemetry = False
        botocore_instrumentor.return_value.is_instrumented_by_opentelemetry = False
        telemetry = init_langfuse_telemetry(
            LangfuseClientConfig(
                public_key="pk-tracer",
                secret_key="sk",
                tracer_provider=tracer_provider,
                redact_http_auth_headers=True,
            ),
            instrument_httpx=True,
            instrument_botocore=True,
        )
        telemetry.uninstrument()

    httpx_instrumentor.return_value.instrument.assert_called_once_with(
        request_hook=scrub_auth_header,
        async_request_hook=scrub_auth_header_async,
        tracer_provider=tracer_provider,
    )
    botocore_instrumentor.return_value.instrument.assert_called_once_with(tracer_provider=tracer_provider)


def test_elapsed_timer_measures_local_block() -> None:
    with patch("digintlab_langfuse.timing.time.monotonic", side_effect=[10.0, 10.1234, 10.5]):
        with elapsed_timer() as timer:
            assert timer.rounded(3) == 0.123

        assert timer.elapsed_seconds == 0.5
        assert timer.stop() == 0.5


def test_score_elapsed_emits_numeric_score_after_success() -> None:
    lf = MagicMock()

    with (
        patch("digintlab_langfuse.timing.time.monotonic", side_effect=[10.0, 10.129]),
        score_elapsed(
            lf,
            name="operation_elapsed_seconds",
            target=ScoreTarget(run_id="run-1", trace_id="trace-1"),
            digits=2,
        ),
    ):
        pass

    lf.create_score.assert_called_once_with(
        name="operation_elapsed_seconds",
        value=0.13,
        data_type="NUMERIC",
        trace_id="trace-1",
        metadata={"run_id": "run-1", "session_id": "run-1"},
    )


def test_score_elapsed_does_not_emit_score_when_body_raises() -> None:
    lf = MagicMock()
    message = "boom"

    with (
        patch("digintlab_langfuse.timing.time.monotonic", side_effect=[10.0, 10.5]),
        pytest.raises(RuntimeError, match=message),
        score_elapsed(lf, name="failed_elapsed_seconds"),
    ):
        raise RuntimeError(message)

    lf.create_score.assert_not_called()


def test_span_timing_processor_aggregates_ended_spans() -> None:
    processor = SpanTimingProcessor()
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("package-test")

    with tracer.start_as_current_span("http.request"):
        pass
    with tracer.start_as_current_span("http.request"):
        pass

    buckets = processor.snapshot(reset=True)

    assert len(buckets) == 1
    assert buckets[0].key == "package-test:http.request"
    assert buckets[0].count == 2
    assert buckets[0].total_elapsed_seconds > 0
    assert buckets[0].max_elapsed_seconds > 0
    assert processor.snapshot() == ()


def test_span_timing_processor_accepts_custom_bucket_key() -> None:
    def server_address(span: ReadableSpan) -> str:
        return str((span.attributes or {}).get("server.address", ""))

    processor = SpanTimingProcessor(key=server_address)
    provider = TracerProvider()
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("package-test")

    with tracer.start_as_current_span("GET", attributes={"server.address": "api.example"}):
        pass

    assert processor.snapshot()[0].key == "api.example"


def test_span_timing_recorder_installs_processor_on_provider() -> None:
    recorder = create_span_timing_recorder()
    tracer = recorder.tracer_provider.get_tracer("package-test")

    with tracer.start_as_current_span("provider.call"):
        pass

    buckets = recorder.snapshot(reset=True)

    assert len(buckets) == 1
    assert buckets[0].key == "package-test:provider.call"
    assert buckets[0].count == 1
    assert recorder.snapshot() == ()


def test_span_timing_recorder_accepts_existing_provider_and_custom_key() -> None:
    provider = TracerProvider()
    recorder = create_span_timing_recorder(tracer_provider=provider, key=lambda span: span.name)

    assert recorder.tracer_provider is provider
    with provider.get_tracer("package-test").start_as_current_span("client.latency"):
        pass

    assert recorder.snapshot()[0].key == "client.latency"


def test_langchain_callbacks_uses_sdk_handler_outside_proxy_mode() -> None:
    from langchain_core.callbacks import BaseCallbackHandler
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

    callback = BaseCallbackHandler()
    telemetry = LangfuseTelemetry(MagicMock(), public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    callbacks = telemetry.langchain_callbacks(proxy_mode=False, callbacks=[callback])

    assert isinstance(callbacks[0], LangfuseCallbackHandler)
    assert callbacks[1] is callback


def test_langchain_callbacks_uses_proxy_callbacks_in_proxy_mode() -> None:
    from langchain_core.callbacks import BaseCallbackHandler

    cache_callback = BaseCallbackHandler()
    tool_callback = BaseCallbackHandler()
    telemetry = LangfuseTelemetry(MagicMock(), public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    callbacks = telemetry.langchain_callbacks(proxy_mode=True, callbacks=[cache_callback], proxy_callbacks=[tool_callback])

    assert callbacks == [cache_callback, tool_callback]


def test_telemetry_exposes_bound_runtime_helpers() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    callback = telemetry.tool_span_callback()
    assert isinstance(callback, ToolSpanCallback)

    with patch("digintlab_langfuse.client.capture_trace_runtime_context", return_value=None) as runtime_context:
        assert telemetry.trace_runtime_context(run_id="run-1", metadata={"tenant": "acme"}) is None

    runtime_context.assert_called_once_with(
        client,
        session_id=None,
        user_id=None,
        run_id="run-1",
        metadata={"tenant": "acme"},
    )

    with patch("digintlab_langfuse.client.observe_node_span") as node_span:
        telemetry.observe_node_span("graph_node", enabled=True, kind=ObservationKind.AGENT)

    node_span.assert_called_once()
    assert node_span.call_args.kwargs["enabled"] is True
    assert node_span.call_args.kwargs["kind"] is ObservationKind.AGENT
    assert node_span.call_args.kwargs["client_getter"]() is client

    with patch("digintlab_langfuse.client.emit_langfuse_guardrail_observation") as guardrail:
        telemetry.emit_guardrail_observation("blocked", output={"ok": False}, observation_input={"call": 1}, status_message="blocked")

    guardrail.assert_called_once()
    assert guardrail.call_args.kwargs["output"] == {"ok": False}
    assert guardrail.call_args.kwargs["observation_input"] == {"call": 1}
    assert guardrail.call_args.kwargs["client_getter"]() is client

    with patch(
        "digintlab_langfuse.client.with_litellm_proxy_trace_metadata",
        return_value={"extra_body": {"metadata": {"trace_id": "trace-1"}}},
    ) as trace_metadata:
        assert telemetry.with_litellm_proxy_trace_metadata({"model": "anthropic/claude"}) == {"extra_body": {"metadata": {"trace_id": "trace-1"}}}

    trace_metadata.assert_called_once()
    assert trace_metadata.call_args.kwargs["client_getter"]() is client


def test_trace_call_adds_attributes_and_optional_root_observation() -> None:
    client = MagicMock()
    root_observation = MagicMock()
    client.start_as_current_observation.return_value = root_observation
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    with (
        patch("digintlab_langfuse.client.propagate_attributes") as propagate_attributes,
        telemetry.trace_call(name="run", session_id="run-1", user_id="analyst", root_observation=True),
    ):
        pass

    propagate_attributes.assert_called_once_with(session_id="run-1", user_id="analyst")
    client.start_as_current_observation.assert_called_once_with(name="run", as_type="agent")
    root_observation.__enter__.assert_called_once()
    root_observation.__exit__.assert_called_once()


def test_trace_call_forwards_public_trace_options() -> None:
    client = MagicMock()
    observation_context = MagicMock()
    client.start_as_current_observation.return_value = observation_context
    telemetry = LangfuseTelemetry(client, public_key="pk", instrument_httpx=False, instrument_botocore=False)

    with patch("digintlab_langfuse.client.propagate_attributes") as propagate:
        propagation_context = MagicMock()
        propagate.return_value = propagation_context

        with telemetry.trace_call(
            name="run",
            session_id="run-1",
            user_id="analyst",
            root_observation=True,
            tags=("smoke", "ci"),
            metadata={"suite": "package"},
            version="2026.06.21",
            trace_name="custom-trace",
            trace_id="trace-1",
            parent_span_id="parent-1",
            root_input={"prompt": "x"},
            root_metadata={"root": "metadata"},
        ):
            pass

    propagate.assert_called_once_with(
        session_id="run-1",
        user_id="analyst",
        trace_name="custom-trace",
        tags=["smoke", "ci"],
        metadata={"suite": "package"},
        version="2026.06.21",
    )
    client.start_as_current_observation.assert_called_once()
    call_kwargs = client.start_as_current_observation.call_args.kwargs
    trace_context = call_kwargs["trace_context"]
    assert trace_context == {"trace_id": "trace-1", "parent_span_id": "parent-1"}
    assert call_kwargs == {
        "trace_context": trace_context,
        "name": "run",
        "as_type": "agent",
        "input": {"prompt": "x"},
        "metadata": {"root": "metadata"},
        "version": "2026.06.21",
    }


def test_trace_call_logs_root_start_failure_and_runs_body() -> None:
    client = MagicMock()
    observation_context = MagicMock()
    observation_context.__enter__.side_effect = RuntimeError("telemetry start failed")
    client.start_as_current_observation.return_value = observation_context
    telemetry = LangfuseTelemetry(client, public_key="pk", instrument_httpx=False, instrument_botocore=False)
    body_ran = False

    with (
        patch("digintlab_langfuse.client.propagate_attributes"),
        patch("digintlab_langfuse.client.logger", create=True) as logger,
        telemetry.trace_call(name="run", session_id="run-1", user_id="analyst", root_observation=True),
    ):
        body_ran = True

    assert body_ran
    logger.exception.assert_called_once()


def test_trace_call_logs_root_end_failure_without_failing_body() -> None:
    client = MagicMock()
    observation_context = MagicMock()
    observation_context.__exit__.side_effect = RuntimeError("telemetry end failed")
    client.start_as_current_observation.return_value = observation_context
    telemetry = LangfuseTelemetry(client, public_key="pk", instrument_httpx=False, instrument_botocore=False)

    with (
        patch("digintlab_langfuse.client.propagate_attributes"),
        patch("digintlab_langfuse.client.logger", create=True) as logger,
        telemetry.trace_call(name="run", session_id="run-1", user_id="analyst", root_observation=True),
    ):
        pass

    logger.exception.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["creation", "entry", "exit"])
def test_trace_call_propagation_failures_do_not_replace_application_behavior(failure_stage: str) -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk", instrument_httpx=False, instrument_botocore=False)
    propagation_context = MagicMock()
    propagate = MagicMock(return_value=propagation_context)
    if failure_stage == "creation":
        propagate.side_effect = RuntimeError("propagation creation failed")
    elif failure_stage == "entry":
        propagation_context.__enter__.side_effect = RuntimeError("propagation entry failed")
    else:
        propagation_context.__exit__.side_effect = RuntimeError("propagation exit failed")
    body_ran = False

    with (
        patch("digintlab_langfuse.client.propagate_attributes", propagate),
        patch("digintlab_langfuse.client.logger", create=True) as logger,
        telemetry.trace_call(name="run", session_id="run-1", user_id="analyst", root_observation=False),
    ):
        body_ran = True

    assert body_ran
    logger.exception.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_stops_instrumentation_and_waits_for_sdk_shutdown() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    with patch.object(LangfuseTelemetry, "uninstrument") as uninstrument:
        await telemetry.shutdown()

    uninstrument.assert_called_once_with()
    client.shutdown.assert_called_once()


@pytest.mark.asyncio
async def test_shutdown_flushes_even_when_uninstrument_fails() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    with (
        patch.object(LangfuseTelemetry, "uninstrument", side_effect=RuntimeError("teardown boom")),
        pytest.raises(RuntimeError, match="teardown boom"),
    ):
        await telemetry.shutdown()

    client.shutdown.assert_called_once()


def _current_observation_context(mock_lf: MagicMock, span: MagicMock) -> MagicMock:
    context = MagicMock()
    context.__enter__.return_value = span
    mock_lf.start_as_current_observation.return_value = context
    return context


def test_observe_node_span_creates_span_for_graph_node() -> None:
    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    context = _current_observation_context(mock_lf, mock_span)

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("correlate", enabled=True):
        pass

    mock_lf.start_as_current_observation.assert_called_once_with(name="correlate", as_type="span")
    context.__exit__.assert_called_once()


@pytest.mark.parametrize("kind", list(ObservationKind))
def test_observe_node_span_uses_explicit_kind(kind: ObservationKind) -> None:
    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    _current_observation_context(mock_lf, mock_span)

    with (
        patch("digintlab_langfuse.spans.get_client", return_value=mock_lf),
        observe_node_span(
            "judge_validate",
            enabled=True,
            kind=kind,
        ),
    ):
        pass

    mock_lf.start_as_current_observation.assert_called_once_with(name="judge_validate", as_type=kind.value)


def test_guardrail_observation_uses_active_node_as_parent() -> None:
    node_span = MagicMock()
    node_span.id = "node-span-123"
    guardrail_span = MagicMock()
    guardrail_span.id = "guardrail-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    _current_observation_context(mock_lf, node_span)
    node_span.start_observation.return_value = guardrail_span

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("crowdstrike_agent", enabled=True):
        emit_guardrail_observation(
            "duplicate-tool-call",
            output={"action": "blocked_duplicate_tool_call", "blocked": True},
            status_message="Duplicate tool call blocked",
        )

    guardrail_call = node_span.start_observation.call_args
    assert guardrail_call.kwargs["name"] == "duplicate-tool-call"
    assert guardrail_call.kwargs["as_type"] == "guardrail"
    assert "trace_context" not in guardrail_call.kwargs
    assert guardrail_call.kwargs["output"]["blocked"] is True
    mock_lf.start_observation.assert_not_called()
    guardrail_span.end.assert_called_once()


def test_guardrail_observation_remains_child_of_node_inside_generation() -> None:
    exporter = InMemorySpanExporter()
    lf = Langfuse(public_key="pk-nested-guardrail-test", secret_key="sk-test", span_exporter=exporter)

    try:
        with (
            lf.start_as_current_observation(name="root", as_type="agent"),
            observe_node_span("node", enabled=True, client_getter=lambda: lf),
        ):
            node_id = lf.get_current_observation_id()
            with observe_generation("generation", client_getter=lambda: lf):
                emit_guardrail_observation("guardrail", output={"blocked": True}, client_getter=lambda: lf)

        lf.flush()
        guardrail = next(span for span in exporter.get_finished_spans() if span.name == "guardrail")

        assert guardrail.parent is not None
        assert f"{guardrail.parent.span_id:016x}" == node_id
        assert guardrail.end_time is not None
        assert "langfuse.internal.as_root" not in (guardrail.attributes or {})
    finally:
        lf.shutdown()


@pytest.mark.asyncio
async def test_guardrail_observation_is_noop_after_node_closes_in_copied_context() -> None:
    exporter = InMemorySpanExporter()
    lf = Langfuse(public_key="pk-closed-node-guardrail-test", secret_key="sk-test", span_exporter=exporter)
    release = asyncio.Event()

    async def emit_after_release() -> None:
        await release.wait()
        emit_guardrail_observation("late-guardrail", output={"blocked": True}, client_getter=lambda: lf)

    try:
        with lf.start_as_current_observation(name="root", as_type="agent"):
            with observe_node_span("node", enabled=True, client_getter=lambda: lf):
                task = asyncio.create_task(emit_after_release())
                await asyncio.sleep(0)

            release.set()
            await task

        lf.flush()
        assert all(span.name != "late-guardrail" for span in exporter.get_finished_spans())
    finally:
        lf.shutdown()


def test_observe_node_span_disabled_is_noop() -> None:
    mock_lf = MagicMock()

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("triage", enabled=False):
        pass

    mock_lf.start_as_current_observation.assert_not_called()


def test_observe_node_span_without_active_trace_is_noop() -> None:
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = None

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("triage", enabled=True):
        pass

    mock_lf.start_as_current_observation.assert_not_called()


def test_observe_node_span_with_malformed_trace_id_is_noop() -> None:
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = 123

    with observe_node_span("triage", enabled=True, client_getter=lambda: mock_lf):
        pass

    mock_lf.start_as_current_observation.assert_not_called()


def test_observe_node_span_marks_error_level_on_exception() -> None:
    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    context = _current_observation_context(mock_lf, mock_span)
    message = "boom"

    with (
        patch("digintlab_langfuse.spans.get_client", return_value=mock_lf),
        pytest.raises(RuntimeError, match="boom"),
        observe_node_span("plan", enabled=True),
    ):
        raise RuntimeError(message)

    mock_span.update.assert_called_once()
    assert mock_span.update.call_args.kwargs["level"] == "ERROR"
    context.__exit__.assert_called_once()


def test_observe_node_span_logs_start_failure_and_runs_body() -> None:
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    context = MagicMock()
    context.__enter__.side_effect = RuntimeError("telemetry start failed")
    mock_lf.start_as_current_observation.return_value = context
    body_ran = False

    with (
        patch("digintlab_langfuse.spans.logger", create=True) as logger,
        observe_node_span("plan", enabled=True, client_getter=lambda: mock_lf),
    ):
        body_ran = True

    assert body_ran
    logger.exception.assert_called_once()


def test_observe_node_span_preserves_application_error_when_telemetry_fails() -> None:
    message = "application failed"
    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_span.update.side_effect = RuntimeError("telemetry update failed")
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    context = _current_observation_context(mock_lf, mock_span)
    context.__exit__.side_effect = RuntimeError("telemetry end failed")

    with (
        patch("digintlab_langfuse.spans.logger", create=True) as logger,
        pytest.raises(ValueError, match=message),
        observe_node_span("plan", enabled=True, client_getter=lambda: mock_lf),
    ):
        raise ValueError(message)

    assert logger.exception.call_count == 2


def test_guardrail_observation_logs_start_and_end_failures() -> None:
    node_span = MagicMock()
    node_span.id = "node-span-123"
    guardrail_span = MagicMock()
    guardrail_span.end.side_effect = RuntimeError("telemetry end failed")
    node_span.start_observation.side_effect = [RuntimeError("telemetry start failed"), guardrail_span]
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    _current_observation_context(mock_lf, node_span)

    with (
        patch("digintlab_langfuse.spans.logger", create=True) as logger,
        observe_node_span("node", enabled=True, client_getter=lambda: mock_lf),
    ):
        emit_guardrail_observation("start-failure", output={}, client_getter=lambda: mock_lf)
        emit_guardrail_observation("end-failure", output={}, client_getter=lambda: mock_lf)

    assert logger.exception.call_count == 2


def test_observe_node_span_sets_active_parent_observation_id_while_open() -> None:
    mock_span = MagicMock()
    mock_span.id = "node-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    _current_observation_context(mock_lf, mock_span)

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("crowdstrike_agent", enabled=True):
        assert active_parent_observation_id() == "node-span-123"

    assert active_parent_observation_id() is None


def test_active_parent_observation_id_ignores_malformed_sdk_id() -> None:
    mock_span = MagicMock()
    mock_span.id = 123
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    _current_observation_context(mock_lf, mock_span)

    with observe_node_span("node", enabled=True, client_getter=lambda: mock_lf):
        assert active_parent_observation_id() is None


def test_active_parent_observation_is_scoped_to_owning_client() -> None:
    owner_span = MagicMock()
    owner_span.id = "owner-node"
    owner = MagicMock()
    owner.get_current_trace_id.return_value = "owner-trace"
    _current_observation_context(owner, owner_span)

    other = MagicMock()
    other.get_current_trace_id.return_value = "other-trace"
    other.get_current_observation_id.return_value = "other-parent"

    with observe_node_span("node", enabled=True, client_getter=lambda: owner):
        assert active_parent_observation_id(owner) == "owner-node"
        assert active_parent_observation_id(other) is None
        runtime_context = trace_runtime_context(other)
        assert runtime_context is not None
        assert runtime_context.parent_observation_id == "other-parent"

        proxy_kwargs = with_litellm_proxy_trace_metadata({}, client_getter=lambda: other)
        extra_body = proxy_kwargs["extra_body"]
        assert isinstance(extra_body, dict)
        metadata = extra_body["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["parent_observation_id"] == "other-parent"

        callback = ToolSpanCallback(client_getter=lambda: other)
        callback.on_tool_start({"name": "tool"}, "input", run_id=uuid4())
        assert other.start_observation.call_args.kwargs["trace_context"] == {
            "trace_id": "other-trace",
            "parent_span_id": "other-parent",
        }


@pytest.mark.asyncio
async def test_active_parent_observation_propagates_to_created_task() -> None:
    mock_span = MagicMock()
    mock_span.id = "node-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "root-span"
    _current_observation_context(mock_lf, mock_span)

    async def child_metadata() -> dict[str, JsonValue]:
        result = with_litellm_proxy_trace_metadata({})
        extra_body = result["extra_body"]
        assert isinstance(extra_body, dict)
        raw = extra_body["metadata"]
        assert isinstance(raw, dict)
        return {str(k): v for k, v in raw.items()}

    with (
        patch("digintlab_langfuse.spans.get_client", return_value=mock_lf),
        patch("digintlab_langfuse.litellm.get_client", return_value=mock_lf),
        observe_node_span("crowdstrike_agent", enabled=True),
    ):
        metadata = await asyncio.create_task(child_metadata())

    assert metadata["parent_observation_id"] == "node-span-123"


def test_observe_generation_marks_observation_error_on_exception() -> None:
    lf = MagicMock()
    lf.get_current_trace_id.return_value = "trace-1"
    generation = MagicMock()
    generation_context = MagicMock()
    generation_context.__enter__.return_value = generation
    lf.start_as_current_observation.return_value = generation_context
    message = "provider boom"

    with pytest.raises(RuntimeError, match=message), observe_generation("llm", client_getter=lambda: lf):
        raise RuntimeError(message)

    generation.update.assert_called_once_with(level=ObservationLevel.ERROR.value, status_message=message)


def test_observe_generation_preserves_application_error_when_telemetry_fails() -> None:
    message = "application failed"
    lf = MagicMock()
    lf.get_current_trace_id.return_value = "trace-1"
    generation = MagicMock()
    generation.update.side_effect = RuntimeError("telemetry update failed")
    generation_context = MagicMock()
    generation_context.__enter__.return_value = generation
    generation_context.__exit__.side_effect = RuntimeError("telemetry end failed")
    lf.start_as_current_observation.return_value = generation_context

    with (
        patch("digintlab_langfuse.spans.logger", create=True) as logger,
        pytest.raises(ValueError, match=message),
        observe_generation("llm", client_getter=lambda: lf),
    ):
        raise ValueError(message)

    assert logger.exception.call_count == 2


def _json_object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _extra_body_metadata(result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _json_object(_json_object(result["extra_body"])["metadata"])


def test_litellm_proxy_trace_metadata_uses_only_extra_body() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = "obs-456"

    result = with_litellm_proxy_trace_metadata({"metadata": {"tenant": "acme"}}, client_getter=lambda: fake_lf)

    assert "metadata" not in result
    assert _extra_body_metadata(result) == {
        "tenant": "acme",
        "existing_trace_id": "trace-123",
        "trace_id": "trace-123",
        "parent_observation_id": "obs-456",
    }


def test_litellm_proxy_trace_metadata_prefers_active_node_parent_observation() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = "root-obs"

    result = with_litellm_proxy_trace_metadata({}, client_getter=lambda: fake_lf, active_parent_getter=lambda: "node-obs")

    assert _extra_body_metadata(result)["parent_observation_id"] == "node-obs"


def test_litellm_proxy_trace_metadata_uses_run_id_as_session_id() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None

    result = with_litellm_proxy_trace_metadata({"metadata": {"run_id": "run-123"}}, client_getter=lambda: fake_lf)

    assert _extra_body_metadata(result)["trace_id"] == "trace-123"
    assert _extra_body_metadata(result)["session_id"] == "run-123"


def test_litellm_proxy_trace_metadata_reads_run_manager_metadata() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None
    run_manager = MagicMock()
    run_manager.metadata = {"run_id": "run-123", "trace_user_id": "analyst"}

    result = with_litellm_proxy_trace_metadata({}, run_manager, client_getter=lambda: fake_lf)

    assert _extra_body_metadata(result) == {
        "run_id": "run-123",
        "trace_user_id": "analyst",
        "existing_trace_id": "trace-123",
        "trace_id": "trace-123",
        "session_id": "run-123",
    }


def test_litellm_proxy_trace_metadata_preserves_extra_body_metadata() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None

    result = with_litellm_proxy_trace_metadata(
        {"metadata": {"run_id": "run-123"}, "extra_body": {"metadata": {"request_id": "req-1"}}},
        client_getter=lambda: fake_lf,
    )

    assert _extra_body_metadata(result) == {
        "request_id": "req-1",
        "run_id": "run-123",
        "existing_trace_id": "trace-123",
        "trace_id": "trace-123",
        "session_id": "run-123",
    }


def test_litellm_proxy_trace_metadata_leaves_kwargs_unchanged_without_active_trace() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = None

    kwargs: dict[str, JsonValue] = {"metadata": {"tenant": "acme"}}
    result = with_litellm_proxy_trace_metadata(kwargs, client_getter=lambda: fake_lf)

    assert result is kwargs


def test_litellm_proxy_trace_metadata_rejects_non_object_extra_body() -> None:

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None

    with pytest.raises(TypeError, match="extra_body must be a JSON object"):
        with_litellm_proxy_trace_metadata({"extra_body": "invalid"}, client_getter=lambda: fake_lf)


def test_tool_span_callback_emits_default_tool_observation() -> None:
    tool_span = MagicMock()
    tool_span.id = "tool-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "parent-span"
    mock_lf.start_observation.return_value = tool_span

    run_id = uuid4()
    callback = ToolSpanCallback(client_getter=lambda: mock_lf)
    callback.on_tool_start(
        {"name": "query_ngsiem"},
        '{"query":"#event_simpleName=ProcessRollup2"}',
        run_id=run_id,
        inputs={"query": "#event_simpleName=ProcessRollup2", "limit": 10},
    )
    callback.on_tool_end({"aid": "abc"}, run_id=run_id)

    start_call = mock_lf.start_observation.call_args
    assert start_call.kwargs["name"] == "query_ngsiem"
    assert start_call.kwargs["as_type"] == "tool"
    assert start_call.kwargs["trace_context"] == {"trace_id": "trace-123", "parent_span_id": "parent-span"}
    assert start_call.kwargs["input"]["inputs"]["limit"] == 10
    tool_span.update.assert_called_once_with(output={"status": "ok", "output": {"aid": "abc"}})
    tool_span.end.assert_called_once()


@pytest.mark.parametrize(
    ("summary", "expected_update"),
    [
        pytest.param({"output": {"ok": True}}, {"output": {"ok": True}}, id="output"),
        pytest.param(
            {"output": {"ok": True}, "level": ObservationLevel.WARNING},
            {"output": {"ok": True}, "level": "WARNING"},
            id="level",
        ),
        pytest.param(
            {"output": {"ok": True}, "status_message": "completed"},
            {"output": {"ok": True}, "status_message": "completed"},
            id="status-message",
        ),
        pytest.param(
            {"output": {"ok": True}, "level": ObservationLevel.WARNING, "status_message": "completed"},
            {"output": {"ok": True}, "level": "WARNING", "status_message": "completed"},
            id="level-and-status-message",
        ),
    ],
)
def test_tool_span_callback_forwards_present_summary_fields(
    summary: ToolObservationResult,
    expected_update: dict[str, object],
) -> None:
    tool_span = MagicMock()
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "parent-span"
    mock_lf.start_observation.return_value = tool_span
    callback = ToolSpanCallback(client_getter=lambda: mock_lf, summarize_output=lambda _output: summary)
    run_id = uuid4()

    callback.on_tool_start({"name": "query"}, "{}", run_id=run_id)
    callback.on_tool_end("result", run_id=run_id)

    tool_span.update.assert_called_once_with(**expected_update)
    tool_span.end.assert_called_once_with()


def test_tool_span_callback_does_not_create_a_logical_root_without_a_parent() -> None:
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = None
    callback = ToolSpanCallback(client_getter=lambda: mock_lf)

    callback.on_tool_start({"name": "query_ngsiem"}, "{}", run_id=uuid4())

    mock_lf.start_observation.assert_not_called()


def test_tool_span_callback_marks_tool_error_event() -> None:
    tool_span = MagicMock()
    tool_span.id = "tool-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "parent-span"
    mock_lf.start_observation.return_value = tool_span

    run_id = uuid4()
    callback = ToolSpanCallback(client_getter=lambda: mock_lf)
    callback.on_tool_start({"name": "query_ngsiem"}, "{}", run_id=run_id)
    callback.on_tool_error(RuntimeError("boom"), run_id=run_id)

    kwargs = tool_span.update.call_args.kwargs
    assert kwargs["output"]["status"] == "error"
    assert kwargs["output"]["error"] == "RuntimeError"
    assert kwargs["level"] == "ERROR"
    assert kwargs["status_message"] == "RuntimeError: boom"
    tool_span.end.assert_called_once()


def test_tool_span_callback_logs_start_failure() -> None:
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "parent-span"
    mock_lf.start_observation.side_effect = RuntimeError("telemetry start failed")
    callback = ToolSpanCallback(client_getter=lambda: mock_lf)

    with patch("digintlab_langfuse.callbacks.logger", create=True) as logger:
        callback.on_tool_start({"name": "query"}, "{}", run_id=uuid4())

    logger.exception.assert_called_once()


def test_tool_span_callback_logs_current_observation_lookup_failure() -> None:
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.side_effect = RuntimeError("telemetry lookup failed")
    callback = ToolSpanCallback(client_getter=lambda: mock_lf)

    with patch("digintlab_langfuse.callbacks.logger", create=True) as logger:
        callback.on_tool_start({"name": "query"}, "{}", run_id=uuid4())

    logger.exception.assert_called_once()
    mock_lf.start_observation.assert_not_called()


def test_tool_span_callback_logs_update_and_end_failures() -> None:
    tool_span = MagicMock()
    tool_span.update.side_effect = RuntimeError("telemetry update failed")
    tool_span.end.side_effect = RuntimeError("telemetry end failed")
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "parent-span"
    mock_lf.start_observation.return_value = tool_span
    callback = ToolSpanCallback(client_getter=lambda: mock_lf)
    run_id = uuid4()
    callback.on_tool_start({"name": "query"}, "{}", run_id=run_id)

    with patch("digintlab_langfuse.callbacks.logger", create=True) as logger:
        callback.on_tool_end({"result": "ok"}, run_id=run_id)

    assert logger.exception.call_count == 2
    tool_span.end.assert_called_once()


def test_emit_score_supports_boolean_text_and_metadata() -> None:
    lf = MagicMock()
    score = BooleanScore(name="policy_passed", value=True)
    assert_type(score.value, bool)
    assert_type(NumericScore(name="confidence", value=0.9).value, float)

    emit_score(
        lf,
        score=score,
        target=ScoreTarget(
            run_id="run-1",
            trace_id="trace-1",
            observation_id="obs-1",
            comment="passed",
            metadata={"suite": "smoke"},
        ),
    )

    lf.create_score.assert_called_once_with(
        trace_id="trace-1",
        observation_id="obs-1",
        name="policy_passed",
        value=1.0,
        data_type="BOOLEAN",
        comment="passed",
        metadata={"run_id": "run-1", "session_id": "run-1", "suite": "smoke"},
    )


def test_langfuse_telemetry_emits_typed_scores() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    telemetry.emit_score(score=BooleanScore(name="policy_passed", value=False), target=ScoreTarget(run_id="run-2"))
    telemetry.score_current_trace(score=NumericScore(name="confidence", value=0.91))
    telemetry.score_current_span(score=BooleanScore(name="valid", value=False))

    client.create_score.assert_called_once_with(
        name="policy_passed",
        value=0.0,
        data_type="BOOLEAN",
        session_id="run-2",
        metadata={"run_id": "run-2", "session_id": "run-2"},
    )
    client.score_current_trace.assert_called_once_with(name="confidence", value=0.91, data_type="NUMERIC")
    client.score_current_span.assert_called_once_with(name="valid", value=0.0, data_type="BOOLEAN")


def test_prompt_helpers_fetch_and_create_prompts() -> None:
    lf = MagicMock()
    fetched_prompt = get_prompt(
        lf,
        ChatPromptRequest(
            name="incident-correlate",
            label="staging",
            cache_ttl_seconds=300,
            fallback=[{"role": "system", "content": "Fallback"}],
        ),
    )
    assert_type(fetched_prompt, ChatPromptClient)
    assert ChatPromptRequest(name="kind-check").prompt_kind is PromptKind.CHAT

    lf.get_prompt.assert_called_once_with(
        "incident-correlate",
        type="chat",
        version=None,
        label="staging",
        cache_ttl_seconds=300,
        fallback=[{"role": "system", "content": "Fallback"}],
        max_retries=None,
        fetch_timeout_seconds=None,
    )

    created_prompt = create_or_update_prompt(
        lf,
        TextManagedPrompt(
            name="incident-correlate",
            prompt="Write a report for {{incident}}",
            labels=("production",),
            tags=("incident-response",),
            config={"temperature": 0},
            commit_message="baseline",
        ),
    )
    assert_type(created_prompt, TextPromptClient)
    assert TextManagedPrompt(name="kind-check", prompt="x").prompt_kind is PromptKind.TEXT

    lf.create_prompt.assert_called_once_with(
        name="incident-correlate",
        type="text",
        prompt="Write a report for {{incident}}",
        labels=["production"],
        tags=["incident-response"],
        config={"temperature": 0},
        commit_message="baseline",
    )


def test_dataset_and_experiment_helpers_delegate_to_langfuse_sdk() -> None:
    lf = MagicMock()
    dataset_config = DatasetConfig(
        name="incident/smoke",
        description="Smoke cases",
        metadata={"owner": "ir"},
        input_schema={"type": "object"},
        expected_output_schema={"type": "object"},
    )
    create_dataset(lf, dataset_config)

    lf.create_dataset.assert_called_once_with(
        name="incident/smoke",
        description="Smoke cases",
        metadata={"owner": "ir"},
        input_schema={"type": "object"},
        expected_output_schema={"type": "object"},
    )

    create_dataset_item(
        lf,
        dataset_name="incident/smoke",
        input={"alert": "x"},
        expected_output={"verdict": "benign"},
        metadata={"fixture": "x"},
        source_trace_id="trace-1",
        source_observation_id="obs-1",
        id="item-1",
    )

    lf.create_dataset_item.assert_called_once_with(
        dataset_name="incident/smoke",
        input={"alert": "x"},
        expected_output={"verdict": "benign"},
        metadata={"fixture": "x"},
        source_trace_id="trace-1",
        source_observation_id="obs-1",
        status=None,
        id="item-1",
    )

    def task(*, item: object, **_kwargs: object) -> str:
        return str(item)

    run_local_experiment(
        lf,
        name="smoke",
        data=[{"input": {"alert": "x"}, "expected_output": {"verdict": "benign"}}],
        task=task,
        run_name="smoke-main",
        metadata={"variant": "main"},
        max_concurrency=4,
    )

    lf.run_experiment.assert_called_once_with(
        name="smoke",
        run_name="smoke-main",
        description=None,
        data=[{"input": {"alert": "x"}, "expected_output": {"verdict": "benign"}}],
        task=task,
        evaluators=[],
        run_evaluators=[],
        max_concurrency=4,
        metadata={"variant": "main"},
    )


def test_local_experiment_preserves_optional_item_metadata_shapes() -> None:
    lf = MagicMock()

    def task(*, item: object, **_kwargs: object) -> str:
        return str(item)

    run_local_experiment(
        lf,
        name="metadata-shapes",
        data=[{}, {"metadata": None}, {"metadata": {"priority": 1}}],
        task=task,
    )

    assert lf.run_experiment.call_args.kwargs["data"] == [{}, {"metadata": None}, {"metadata": {"priority": 1}}]
