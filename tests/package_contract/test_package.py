"""Standalone package contract tests for digintlab_langfuse."""

import ast
from pathlib import Path
from typing import assert_type
from unittest.mock import MagicMock, patch

import pytest
from langfuse.api.commons.types import ObservationLevel
from langfuse.model import ChatPromptClient, TextPromptClient
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

import digintlab_langfuse
from digintlab_langfuse.callbacks import ToolObservationResult, ToolSpanCallback
from digintlab_langfuse.client import LangfuseClientConfig, LangfuseTelemetry, init_langfuse_client
from digintlab_langfuse.experiments import (
    DatasetConfig,
    create_dataset,
    create_dataset_item,
    run_local_experiment,
)
from digintlab_langfuse.litellm import with_langfuse_trace_metadata
from digintlab_langfuse.prompts import (
    ChatPromptRequest,
    PromptKind,
    TextManagedPrompt,
    create_or_update_prompt,
    get_prompt,
)
from digintlab_langfuse.scores import BooleanScore, NumericScore, ScoreTarget, emit_score, score_elapsed
from digintlab_langfuse.spans import ObservationKind, emit_guardrail_observation, observe_generation, observe_node_span, scrub_auth_header
from digintlab_langfuse.timing import SpanTimingProcessor, create_span_timing_recorder, elapsed_timer
from digintlab_langfuse.types import JsonValue

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "digintlab_langfuse"
EXPECTED_ROOT_EXPORTS = [
    "LANGFUSE_DEFAULT_BASE_URL",
    "ChatManagedPrompt",
    "ChatPromptRequest",
    "DatasetConfig",
    "ElapsedTimer",
    "LangfuseClientConfig",
    "LangfuseTelemetry",
    "ManagedPrompt",
    "ObservationKind",
    "PromptKind",
    "PromptRequest",
    "BooleanScore",
    "CategoricalScore",
    "CorrectionScore",
    "NumericScore",
    "Score",
    "ScoreDataType",
    "ScoreTarget",
    "SpanTimingBucket",
    "SpanTimingProcessor",
    "SpanTimingRecorder",
    "TextManagedPrompt",
    "TextPromptRequest",
    "TextScore",
    "ToolObservationResult",
    "ToolSpanCallback",
    "active_parent_observation_id",
    "create_dataset",
    "create_dataset_item",
    "create_or_update_prompt",
    "create_span_timing_recorder",
    "current_trace_id",
    "default_span_timing_key",
    "emit_boolean_score",
    "emit_categorical_score",
    "emit_guardrail_observation",
    "emit_numeric_score",
    "emit_score",
    "emit_text_score",
    "elapsed_timer",
    "get_prompt",
    "init_langfuse_client",
    "init_langfuse_telemetry",
    "langchain_prompt_metadata",
    "observe_generation",
    "observe_node_span",
    "run_local_experiment",
    "score_current_span",
    "score_current_span_boolean",
    "score_current_span_categorical",
    "score_current_span_numeric",
    "score_current_span_text",
    "score_current_trace",
    "score_current_trace_boolean",
    "score_current_trace_categorical",
    "score_current_trace_numeric",
    "score_current_trace_text",
    "score_elapsed",
    "shutdown_langfuse_client",
    "with_langfuse_trace_metadata",
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


def test_root_reexports_observability_helpers() -> None:
    assert digintlab_langfuse.ToolObservationResult is ToolObservationResult
    assert digintlab_langfuse.ToolSpanCallback is ToolSpanCallback
    assert digintlab_langfuse.ObservationKind is ObservationKind
    assert digintlab_langfuse.emit_guardrail_observation is emit_guardrail_observation
    assert digintlab_langfuse.observe_node_span is observe_node_span
    assert digintlab_langfuse.with_langfuse_trace_metadata is with_langfuse_trace_metadata
    assert digintlab_langfuse.create_span_timing_recorder is create_span_timing_recorder
    assert digintlab_langfuse.score_elapsed is score_elapsed


def test_client_config_forwards_langfuse_runtime_options() -> None:
    def should_export_span(_span: ReadableSpan) -> bool:
        return True

    config = LangfuseClientConfig(
        public_key="pk",
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
        init_langfuse_client(config)

    langfuse_cls.assert_called_once_with(
        public_key="pk",
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


def test_httpx_auth_redaction_can_be_disabled_by_applications() -> None:
    client = MagicMock()
    with patch("digintlab_langfuse.client.HTTPXClientInstrumentor") as httpx_cls:
        telemetry = LangfuseTelemetry(
            client,
            public_key="pk",
            instrument_httpx=True,
            instrument_botocore=False,
            redact_http_auth_headers=False,
        )
        telemetry.instrument()

    httpx_cls.return_value.instrument.assert_called_once_with()


def test_init_instruments_selected_project_hooks() -> None:
    from digintlab_langfuse.client import init_langfuse_telemetry

    client = MagicMock()
    with (
        patch("digintlab_langfuse.client.init_langfuse_client", return_value=client),
        patch("digintlab_langfuse.client.HTTPXClientInstrumentor") as httpx_instrumentor,
        patch("digintlab_langfuse.client.BotocoreInstrumentor") as botocore_instrumentor,
    ):
        telemetry = init_langfuse_telemetry(
            LangfuseClientConfig(public_key="pk", secret_key="sk"),
            instrument_botocore=True,
        )

    assert telemetry.client is client
    httpx_instrumentor.return_value.instrument.assert_called_once()
    botocore_instrumentor.return_value.instrument.assert_called_once()


def test_configured_tracer_provider_reaches_process_instrumentors() -> None:
    from digintlab_langfuse.client import init_langfuse_telemetry

    client = MagicMock()
    tracer_provider = TracerProvider()
    with (
        patch("digintlab_langfuse.client.init_langfuse_client", return_value=client),
        patch("digintlab_langfuse.client.HTTPXClientInstrumentor") as httpx_instrumentor,
        patch("digintlab_langfuse.client.BotocoreInstrumentor") as botocore_instrumentor,
    ):
        init_langfuse_telemetry(
            LangfuseClientConfig(public_key="pk", secret_key="sk", tracer_provider=tracer_provider),
            instrument_botocore=True,
        )

    httpx_instrumentor.return_value.instrument.assert_called_once_with(request_hook=scrub_auth_header, tracer_provider=tracer_provider)
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

    with patch("digintlab_langfuse.client.current_langfuse_trace_id", return_value="trace-1") as current_trace_id:
        assert telemetry.current_trace_id() == "trace-1"

    current_trace_id.assert_called_once_with(client)

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

    with patch("digintlab_langfuse.client.with_langfuse_trace_metadata", return_value={"metadata": {"trace_id": "trace-1"}}) as trace_metadata:
        assert telemetry.with_litellm_trace_metadata({"model": "anthropic/claude"}) == {"metadata": {"trace_id": "trace-1"}}

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


def test_observe_node_span_creates_span_for_graph_node() -> None:
    from digintlab_langfuse.spans import observe_node_span

    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.start_observation.return_value = mock_span

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("correlate", enabled=True):
        pass

    mock_lf.start_observation.assert_called_once_with(name="correlate", as_type="span")
    mock_span.end.assert_called_once()


def test_observe_node_span_uses_explicit_kind() -> None:
    from digintlab_langfuse.spans import ObservationKind, observe_node_span

    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.start_observation.return_value = mock_span

    with (
        patch("digintlab_langfuse.spans.get_client", return_value=mock_lf),
        observe_node_span(
            "judge_validate",
            enabled=True,
            kind=ObservationKind.EVALUATOR,
        ),
    ):
        pass

    mock_lf.start_observation.assert_called_once_with(name="judge_validate", as_type="evaluator")


def test_guardrail_observation_uses_active_node_as_parent() -> None:
    from digintlab_langfuse.spans import emit_guardrail_observation, observe_node_span

    node_span = MagicMock()
    node_span.id = "node-span-123"
    guardrail_span = MagicMock()
    guardrail_span.id = "guardrail-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.start_observation.side_effect = [node_span, guardrail_span]

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("crowdstrike_agent", enabled=True):
        emit_guardrail_observation(
            "duplicate-tool-call",
            output={"action": "blocked_duplicate_tool_call", "blocked": True},
            status_message="Duplicate tool call blocked",
        )

    guardrail_call = mock_lf.start_observation.call_args_list[1]
    assert guardrail_call.kwargs["name"] == "duplicate-tool-call"
    assert guardrail_call.kwargs["as_type"] == "guardrail"
    assert guardrail_call.kwargs["trace_context"] == {"trace_id": "trace-123", "parent_span_id": "node-span-123"}
    assert guardrail_call.kwargs["output"]["blocked"] is True
    guardrail_span.end.assert_called_once()


def test_observe_node_span_disabled_is_noop() -> None:
    from digintlab_langfuse.spans import observe_node_span

    mock_lf = MagicMock()

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("triage", enabled=False):
        pass

    mock_lf.start_observation.assert_not_called()


def test_observe_node_span_without_active_trace_is_noop() -> None:
    from digintlab_langfuse.spans import observe_node_span

    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = None

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("triage", enabled=True):
        pass

    mock_lf.start_observation.assert_not_called()


def test_observe_node_span_marks_error_level_on_exception() -> None:
    from digintlab_langfuse.spans import observe_node_span

    mock_span = MagicMock()
    mock_span.id = "span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.start_observation.return_value = mock_span
    message = "boom"

    with (
        patch("digintlab_langfuse.spans.get_client", return_value=mock_lf),
        pytest.raises(RuntimeError, match="boom"),
        observe_node_span("plan", enabled=True),
    ):
        raise RuntimeError(message)

    mock_span.update.assert_called_once()
    assert mock_span.update.call_args.kwargs["level"] == "ERROR"
    mock_span.end.assert_called_once()


def test_observe_node_span_sets_active_parent_observation_id_while_open() -> None:
    from digintlab_langfuse.spans import active_parent_observation_id, observe_node_span

    mock_span = MagicMock()
    mock_span.id = "node-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.start_observation.return_value = mock_span

    with patch("digintlab_langfuse.spans.get_client", return_value=mock_lf), observe_node_span("crowdstrike_agent", enabled=True):
        assert active_parent_observation_id() == "node-span-123"

    assert active_parent_observation_id() is None


@pytest.mark.asyncio
async def test_active_parent_observation_propagates_to_created_task() -> None:
    import asyncio

    from digintlab_langfuse.litellm import with_langfuse_trace_metadata
    from digintlab_langfuse.spans import observe_node_span

    mock_span = MagicMock()
    mock_span.id = "node-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
    mock_lf.get_current_observation_id.return_value = "root-span"
    mock_lf.start_observation.return_value = mock_span

    async def child_metadata() -> dict[str, JsonValue]:
        result = with_langfuse_trace_metadata({})
        raw = result["metadata"]
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


def _json_object(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def _metadata(result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _json_object(result["metadata"])


def _extra_body_metadata(result: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _json_object(_json_object(result["extra_body"])["metadata"])


def test_litellm_trace_metadata_adds_langfuse_proxy_metadata() -> None:
    from digintlab_langfuse.litellm import with_langfuse_trace_metadata

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = "obs-456"

    result = with_langfuse_trace_metadata({"metadata": {"tenant": "acme"}}, client_getter=lambda: fake_lf)

    metadata = _metadata(result)
    assert metadata == {
        "tenant": "acme",
        "existing_trace_id": "trace-123",
        "trace_id": "trace-123",
        "parent_observation_id": "obs-456",
    }
    assert _extra_body_metadata(result) == metadata


def test_litellm_trace_metadata_prefers_active_node_parent_observation() -> None:
    from digintlab_langfuse.litellm import with_langfuse_trace_metadata

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = "root-obs"

    result = with_langfuse_trace_metadata({}, client_getter=lambda: fake_lf, active_parent_getter=lambda: "node-obs")

    assert _metadata(result)["parent_observation_id"] == "node-obs"


def test_litellm_trace_metadata_uses_run_id_as_session_id() -> None:
    from digintlab_langfuse.litellm import with_langfuse_trace_metadata

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None

    result = with_langfuse_trace_metadata({"metadata": {"run_id": "run-123"}}, client_getter=lambda: fake_lf)

    assert _metadata(result)["trace_id"] == "trace-123"
    assert _metadata(result)["session_id"] == "run-123"
    assert _extra_body_metadata(result)["session_id"] == "run-123"


def test_litellm_trace_metadata_reads_run_manager_metadata() -> None:
    from digintlab_langfuse.litellm import with_langfuse_trace_metadata

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None
    run_manager = MagicMock()
    run_manager.metadata = {"run_id": "run-123", "trace_user_id": "analyst"}

    result = with_langfuse_trace_metadata({}, run_manager, client_getter=lambda: fake_lf)

    metadata = _metadata(result)
    assert metadata == {
        "run_id": "run-123",
        "trace_user_id": "analyst",
        "existing_trace_id": "trace-123",
        "trace_id": "trace-123",
        "session_id": "run-123",
    }
    assert _extra_body_metadata(result) == metadata


def test_litellm_trace_metadata_preserves_extra_body_metadata() -> None:
    from digintlab_langfuse.litellm import with_langfuse_trace_metadata

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = "trace-123"
    fake_lf.get_current_observation_id.return_value = None

    result = with_langfuse_trace_metadata(
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


def test_litellm_trace_metadata_leaves_kwargs_unchanged_without_active_trace() -> None:
    from digintlab_langfuse.litellm import with_langfuse_trace_metadata

    fake_lf = MagicMock()
    fake_lf.get_current_trace_id.return_value = None

    kwargs: dict[str, JsonValue] = {"metadata": {"tenant": "acme"}}
    result = with_langfuse_trace_metadata(kwargs, client_getter=lambda: fake_lf)

    assert result is kwargs


def test_tool_span_callback_emits_default_tool_observation() -> None:
    from uuid import uuid4

    from digintlab_langfuse.callbacks import ToolSpanCallback

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


def test_tool_span_callback_marks_tool_error_event() -> None:
    from uuid import uuid4

    from digintlab_langfuse.callbacks import ToolSpanCallback

    tool_span = MagicMock()
    tool_span.id = "tool-span-123"
    mock_lf = MagicMock()
    mock_lf.get_current_trace_id.return_value = "trace-123"
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


def test_langfuse_telemetry_exposes_bound_score_helpers() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    telemetry.emit_numeric_score("latency_seconds", 1.25, run_id="run-1", trace_id="trace-1")

    client.create_score.assert_called_once_with(
        name="latency_seconds",
        value=1.25,
        data_type="NUMERIC",
        trace_id="trace-1",
        metadata={"run_id": "run-1", "session_id": "run-1"},
    )

    client.reset_mock()
    telemetry.emit_score(score=BooleanScore(name="policy_passed", value=False), target=ScoreTarget(run_id="run-2"))

    client.create_score.assert_called_once_with(
        name="policy_passed",
        value=0.0,
        data_type="BOOLEAN",
        session_id="run-2",
        metadata={"run_id": "run-2", "session_id": "run-2"},
    )

    client.reset_mock()
    telemetry.score_current_trace_numeric("confidence", 0.91)
    telemetry.score_current_span_boolean("valid", False)

    client.score_current_trace.assert_called_once_with(name="confidence", value=0.91, data_type="NUMERIC")
    client.score_current_span.assert_called_once_with(name="valid", value=0.0, data_type="BOOLEAN")


def test_langfuse_telemetry_keeps_typed_score_escape_hatch() -> None:
    client = MagicMock()
    telemetry = LangfuseTelemetry(client, public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    telemetry.emit_score(score=BooleanScore(name="policy_passed", value=False), target=ScoreTarget(run_id="run-2"))
    telemetry.score_current_trace(score=NumericScore(name="confidence", value=0.91))

    client.create_score.assert_called_once_with(
        name="policy_passed",
        value=0.0,
        data_type="BOOLEAN",
        session_id="run-2",
        metadata={"run_id": "run-2", "session_id": "run-2"},
    )
    client.score_current_trace.assert_called_once_with(name="confidence", value=0.91, data_type="NUMERIC")


def test_langfuse_telemetry_rejects_mixed_score_targets() -> None:
    telemetry = LangfuseTelemetry(MagicMock(), public_key="pk-test", instrument_httpx=False, instrument_botocore=False)

    with pytest.raises(ValueError, match="either target or run_id/trace_id"):
        telemetry.emit_numeric_score("latency_seconds", 1.25, run_id="run-1", target=ScoreTarget(trace_id="trace-1"))


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
