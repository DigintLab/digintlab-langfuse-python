"""Standalone package contract tests for digintlab_langfuse."""

import ast
from pathlib import Path
from typing import assert_type
from unittest.mock import MagicMock, patch

import pytest
from langfuse.api.commons.types import ObservationLevel
from langfuse.model import ChatPromptClient, TextPromptClient
from opentelemetry.sdk.trace import ReadableSpan

import digintlab_langfuse
from digintlab_langfuse.client import LangfuseClientConfig, LangfuseTelemetry, init_langfuse_client
from digintlab_langfuse.experiments import (
    DatasetConfig,
    create_dataset,
    create_dataset_item,
    run_local_experiment,
)
from digintlab_langfuse.prompts import (
    ChatPromptRequest,
    PromptKind,
    TextManagedPrompt,
    create_or_update_prompt,
    get_prompt,
)
from digintlab_langfuse.scores import BooleanScore, NumericScore, ScoreTarget, emit_score
from digintlab_langfuse.spans import observe_generation

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "digintlab_langfuse"
EXPECTED_ROOT_EXPORTS = [
    "LANGFUSE_DEFAULT_BASE_URL",
    "ChatManagedPrompt",
    "ChatPromptRequest",
    "DatasetConfig",
    "LangfuseClientConfig",
    "LangfuseTelemetry",
    "ManagedPrompt",
    "PromptKind",
    "PromptRequest",
    "BooleanScore",
    "CategoricalScore",
    "CorrectionScore",
    "NumericScore",
    "Score",
    "ScoreDataType",
    "ScoreTarget",
    "TextManagedPrompt",
    "TextPromptRequest",
    "TextScore",
    "create_dataset",
    "create_dataset_item",
    "create_or_update_prompt",
    "emit_boolean_score",
    "emit_categorical_score",
    "emit_numeric_score",
    "emit_score",
    "emit_text_score",
    "get_prompt",
    "init_langfuse_client",
    "init_langfuse_telemetry",
    "langchain_prompt_metadata",
    "observe_generation",
    "run_local_experiment",
    "score_current_span_boolean",
    "score_current_span_categorical",
    "score_current_span_numeric",
    "score_current_span_text",
    "score_current_trace_boolean",
    "score_current_trace_categorical",
    "score_current_trace_numeric",
    "score_current_trace_text",
    "shutdown_langfuse_client",
]


def test_package_does_not_import_analaize_internals() -> None:
    """The reusable library must not depend on the application package."""
    offenders: list[str] = []
    for path in sorted(PACKAGE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("analaize."):
                offenders.append(f"{path.name}: from {node.module} import ...")
            if isinstance(node, ast.Import):
                offenders.extend(f"{path.name}: import {alias.name}" for alias in node.names if alias.name.startswith("analaize."))

    assert offenders == []


def test_root_exports_only_public_library_surface() -> None:
    assert set(digintlab_langfuse.__all__) == set(EXPECTED_ROOT_EXPORTS)


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


def test_httpx_auth_redaction_can_be_disabled_for_analaize() -> None:
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
