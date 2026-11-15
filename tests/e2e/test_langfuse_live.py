"""Authenticated end-to-end validation against a dedicated Langfuse project."""

import asyncio
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from time import monotonic
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from langfuse import Langfuse
from langfuse.api.commons.types import ObservationV2
from langfuse.api.core import ApiError
from langfuse.api.scores.types import ScoreV3
from opentelemetry.sdk.trace import TracerProvider

from digintlab_langfuse import (
    LangfuseClientConfig,
    LangfuseTelemetry,
    SensitiveDataRedactionConfig,
    init_langfuse_telemetry,
)
from digintlab_langfuse.json_values import JsonValue, to_json_value
from digintlab_langfuse.scores import BooleanScore, NumericScore
from digintlab_langfuse.spans import current_trace_id, observe_generation

E2E_TIMEOUT_SECONDS = 120.0
TRANSIENT_HTTP_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_NO_OBSERVATION_NAMES: frozenset[str] = frozenset()
pytestmark = [pytest.mark.e2e, pytest.mark.asyncio]


@dataclass(frozen=True, slots=True)
class _E2ESettings:
    public_key: str
    secret_key: str
    base_url: str
    allow_insecure_http: bool


@dataclass(slots=True)
class _LiveProject:
    telemetry: LangfuseTelemetry
    trace_ids: list[str] = field(default_factory=list)

    def new_trace_id(self, run_id: str) -> str:
        trace_id = Langfuse.create_trace_id(seed=f"authenticated-e2e:{run_id}")
        self.trace_ids.append(trace_id)
        return trace_id


def _required_e2e_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} must be set when RUN_LANGFUSE_E2E=1")
    return value


def _e2e_settings() -> _E2ESettings:
    if os.environ.get("RUN_LANGFUSE_E2E") != "1":
        pytest.skip("set RUN_LANGFUSE_E2E=1 to run authenticated Langfuse tests")
    return _E2ESettings(
        public_key=_required_e2e_environment("LANGFUSE_E2E_PUBLIC_KEY"),
        secret_key=_required_e2e_environment("LANGFUSE_E2E_SECRET_KEY"),
        base_url=_required_e2e_environment("LANGFUSE_E2E_BASE_URL"),
        allow_insecure_http=os.environ.get("LANGFUSE_E2E_ALLOW_INSECURE_HTTP") == "1",
    )


@asynccontextmanager
async def _open_live_project(
    *,
    sensitive_data_redaction: SensitiveDataRedactionConfig | None = None,
) -> AsyncIterator[_LiveProject]:
    settings = _e2e_settings()
    telemetry = init_langfuse_telemetry(
        LangfuseClientConfig(
            public_key=settings.public_key,
            secret_key=settings.secret_key,
            base_url=settings.base_url,
            timeout=10,
            environment="digintlab-e2e",
            sample_rate=1.0,
            debug=False,
            tracer_provider=TracerProvider(),
            sensitive_data_redaction=sensitive_data_redaction,
            allow_insecure_http=settings.allow_insecure_http,
        )
    )
    project = _LiveProject(telemetry=telemetry)
    try:
        assert await asyncio.to_thread(telemetry.client.auth_check)
        yield project
    finally:
        await _cleanup_live_project(project)


async def _delete_live_trace(telemetry: LangfuseTelemetry, trace_id: str) -> None:
    deadline = monotonic() + E2E_TIMEOUT_SECONDS
    delay_seconds = 0.5
    while True:
        try:
            await telemetry.client.async_api.trace.delete(trace_id)
        except ApiError as exc:
            retryable = exc.status_code == 404 or exc.status_code in TRANSIENT_HTTP_STATUS_CODES
            if not retryable:
                raise
            if monotonic() >= deadline:
                if exc.status_code == 404:
                    return
                raise
            await asyncio.sleep(delay_seconds)
            delay_seconds = min(delay_seconds * 2, 5.0)
        else:
            return


async def _cleanup_live_project(project: _LiveProject) -> None:
    try:
        await project.telemetry.flush()
        for trace_id in project.trace_ids:
            await _delete_live_trace(project.telemetry, trace_id)
    finally:
        await project.telemetry.shutdown()


async def test_live_project_cleanup_flushes_and_retries_deletion_before_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    delete_attempts = 0

    async def delete_trace(_trace_id: str) -> None:
        nonlocal delete_attempts
        delete_attempts += 1
        events.append("delete")
        if delete_attempts == 1:
            raise ApiError(status_code=404)

    telemetry = MagicMock()
    telemetry.flush = AsyncMock(side_effect=lambda: events.append("flush"))
    telemetry.client.async_api.trace.delete = AsyncMock(side_effect=delete_trace)
    telemetry.shutdown = AsyncMock(side_effect=lambda: events.append("shutdown"))
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    await _cleanup_live_project(_LiveProject(telemetry=telemetry, trace_ids=["trace-id"]))

    assert events == ["flush", "delete", "delete", "shutdown"]
    sleep.assert_awaited_once_with(0.5)
    observation = MagicMock(input=None, output=None, metadata=None)
    observations: dict[str, ObservationV2] = {"root": observation}

    assert not _observations_ready(
        observations,
        expected_names=frozenset({"root"}),
        required_input_names=frozenset({"root"}),
    )

    observation.input = {"ready": True}

    assert _observations_ready(
        observations,
        expected_names=frozenset({"root"}),
        required_input_names=frozenset({"root"}),
    )
    assert not _observations_ready(
        observations,
        expected_names=frozenset({"root"}),
        required_output_names=frozenset({"root"}),
        required_metadata_names=frozenset({"root"}),
    )

    observation.output = {"ready": True}
    observation.metadata = {"ready": True}

    assert _observations_ready(
        observations,
        expected_names=frozenset({"root"}),
        required_output_names=frozenset({"root"}),
        required_metadata_names=frozenset({"root"}),
    )


def _api_json_value(value: object) -> JsonValue:
    """Normalize the v2 API's raw JSON strings at the third-party boundary."""
    return to_json_value(json.loads(value) if isinstance(value, str) else value)


def _query_window() -> tuple[datetime, datetime]:
    now = datetime.now(UTC)
    return now - timedelta(minutes=15), now + timedelta(minutes=15)


def _observations_ready(
    observations: dict[str, ObservationV2],
    *,
    expected_names: frozenset[str],
    required_input_names: frozenset[str] = _NO_OBSERVATION_NAMES,
    required_output_names: frozenset[str] = _NO_OBSERVATION_NAMES,
    required_metadata_names: frozenset[str] = _NO_OBSERVATION_NAMES,
) -> bool:
    if not expected_names <= observations.keys():
        return False
    return (
        all(observations[name].input is not None for name in required_input_names)
        and all(observations[name].output is not None for name in required_output_names)
        and all(observations[name].metadata is not None for name in required_metadata_names)
    )


async def _wait_for_observations(
    telemetry: LangfuseTelemetry,
    *,
    trace_id: str,
    expected_names: frozenset[str],
    required_input_names: frozenset[str] = _NO_OBSERVATION_NAMES,
    required_output_names: frozenset[str] = _NO_OBSERVATION_NAMES,
    required_metadata_names: frozenset[str] = _NO_OBSERVATION_NAMES,
    window_start: datetime,
    window_end: datetime,
) -> dict[str, ObservationV2]:
    deadline = monotonic() + E2E_TIMEOUT_SECONDS
    delay_seconds = 0.5
    latest: dict[str, ObservationV2] = {}
    while monotonic() < deadline:
        try:
            pages = telemetry.client.aiter_observation_pages(
                trace_id=trace_id,
                fields="core,basic,io,metadata,trace_context",
                expand_metadata="authorization,e2e_run_id",
                from_start_time=window_start,
                to_start_time=window_end,
            )
            observations = [observation async for page in pages for observation in page.data]
        except ApiError as exc:
            if exc.status_code not in TRANSIENT_HTTP_STATUS_CODES:
                raise
        else:
            latest = {observation.name: observation for observation in observations if observation.name is not None}
            if _observations_ready(
                latest,
                expected_names=expected_names,
                required_input_names=required_input_names,
                required_output_names=required_output_names,
                required_metadata_names=required_metadata_names,
            ):
                return latest
        await asyncio.sleep(delay_seconds)
        delay_seconds = min(delay_seconds * 2, 5.0)

    pytest.fail(
        f"Langfuse did not materialize observations {sorted(expected_names)} with their required fields within {E2E_TIMEOUT_SECONDS:.0f} seconds"
    )


async def _wait_for_scores(
    telemetry: LangfuseTelemetry,
    *,
    trace_id: str,
    expected_names: frozenset[str],
    window_start: datetime,
    window_end: datetime,
) -> dict[str, ScoreV3]:
    deadline = monotonic() + E2E_TIMEOUT_SECONDS
    delay_seconds = 0.5
    latest: dict[str, ScoreV3] = {}
    while monotonic() < deadline:
        try:
            response = await telemetry.client.async_api.scores.get_many(
                trace_id=trace_id,
                fields="subject",
                limit=20,
                from_timestamp=window_start,
                to_timestamp=window_end,
            )
        except ApiError as exc:
            if exc.status_code not in TRANSIENT_HTTP_STATUS_CODES:
                raise
        else:
            latest = {score.name: score for score in response.data}
            if expected_names <= latest.keys():
                return latest
        await asyncio.sleep(delay_seconds)
        delay_seconds = min(delay_seconds * 2, 5.0)

    pytest.fail(f"Langfuse did not materialize scores {sorted(expected_names)} within {E2E_TIMEOUT_SECONDS:.0f} seconds")


def _assert_redacted_observations(
    observations: dict[str, ObservationV2],
    *,
    root_name: str,
    child_name: str,
    trace_id: str,
    session_id: str,
    run_id: str,
    fake_secret: str,
) -> None:
    root = observations[root_name]
    child = observations[child_name]
    assert root.trace_id == trace_id
    assert child.trace_id == trace_id
    assert root.end_time is not None
    assert child.end_time is not None
    assert root.type == "AGENT"
    assert child.type == "SPAN"
    assert root.is_root_observation is True
    assert child.parent_observation_id == root.id
    assert root.session_id == session_id
    assert root.user_id == "digintlab-e2e"
    assert root.environment == "digintlab-e2e"
    assert _api_json_value(root.input) == {"probe": run_id, "password": "[REDACTED]", "email": "[REDACTED]"}
    root_metadata = _api_json_value(root.metadata)
    assert isinstance(root_metadata, dict)
    assert root_metadata["authorization"] == "[REDACTED]"
    assert root_metadata["e2e_run_id"] == run_id
    assert fake_secret not in root.model_dump_json()


async def test_authenticated_trace_ingestion_and_opt_in_redaction() -> None:
    run_id = uuid4().hex
    root_name = f"digintlab-e2e-redacted-root-{run_id}"
    child_name = f"digintlab-e2e-redacted-child-{run_id}"
    session_id = f"digintlab-e2e-session-{run_id}"
    fake_secret = f"synthetic-secret-{run_id}"
    window_start, window_end = _query_window()
    async with _open_live_project(sensitive_data_redaction=SensitiveDataRedactionConfig()) as project:
        trace_id = project.new_trace_id(run_id)
        with project.telemetry.trace_call(
            name=root_name,
            trace_name=root_name,
            trace_id=trace_id,
            session_id=session_id,
            user_id="digintlab-e2e",
            tags=("digintlab-e2e",),
            metadata={"e2e_run_id": run_id},
            root_observation=True,
            root_input={"probe": run_id, "password": fake_secret, "email": "synthetic@example.com"},
            root_metadata={"authorization": f"Bearer {fake_secret}"},
        ):
            assert current_trace_id(project.telemetry.client) == trace_id
            with project.telemetry.observe_node_span(child_name, enabled=True):
                pass

        await project.telemetry.flush()
        observations = await _wait_for_observations(
            project.telemetry,
            trace_id=trace_id,
            expected_names=frozenset({root_name, child_name}),
            required_input_names=frozenset({root_name}),
            required_metadata_names=frozenset({root_name}),
            window_start=window_start,
            window_end=window_end,
        )
        _assert_redacted_observations(
            observations,
            root_name=root_name,
            child_name=child_name,
            trace_id=trace_id,
            session_id=session_id,
            run_id=run_id,
            fake_secret=fake_secret,
        )


async def test_sensitive_data_redaction_is_off_by_default_on_the_server() -> None:
    run_id = uuid4().hex
    root_name = f"digintlab-e2e-unredacted-root-{run_id}"
    session_id = f"digintlab-e2e-session-{run_id}"
    synthetic_canary = f"synthetic-canary-{run_id}"
    synthetic_email = f"synthetic-{run_id}@example.com"
    window_start, window_end = _query_window()
    async with _open_live_project() as project:
        trace_id = project.new_trace_id(run_id)
        with project.telemetry.trace_call(
            name=root_name,
            trace_id=trace_id,
            session_id=session_id,
            user_id="digintlab-e2e",
            root_observation=True,
            root_input={"password": synthetic_canary, "email": synthetic_email},
            root_metadata={"authorization": f"Bearer {synthetic_canary}"},
        ):
            pass

        await project.telemetry.flush()
        observations = await _wait_for_observations(
            project.telemetry,
            trace_id=trace_id,
            expected_names=frozenset({root_name}),
            required_input_names=frozenset({root_name}),
            required_metadata_names=frozenset({root_name}),
            window_start=window_start,
            window_end=window_end,
        )
        root = observations[root_name]
        assert _api_json_value(root.input) == {"password": synthetic_canary, "email": synthetic_email}
        root_metadata = _api_json_value(root.metadata)
        assert isinstance(root_metadata, dict)
        assert root_metadata["authorization"] == f"Bearer {synthetic_canary}"
        assert synthetic_canary in root.model_dump_json()


async def test_generation_guardrail_hierarchy_and_scores_materialize() -> None:
    run_id = uuid4().hex
    names = {kind: f"digintlab-e2e-{kind}-{run_id}" for kind in ("root", "node", "generation", "guardrail")}
    trace_score_name = f"digintlab-e2e-confidence-{run_id}"
    span_score_name = f"digintlab-e2e-valid-{run_id}"
    window_start, window_end = _query_window()
    async with _open_live_project() as project:
        trace_id = project.new_trace_id(run_id)
        with (
            project.telemetry.trace_call(
                name=names["root"],
                trace_id=trace_id,
                session_id=f"digintlab-e2e-session-{run_id}",
                user_id="digintlab-e2e",
                root_observation=True,
            ),
            project.telemetry.observe_node_span(names["node"], enabled=True),
            observe_generation(
                names["generation"],
                model="digintlab-e2e-model",
                observation_input={"prompt": run_id},
                client_getter=lambda: project.telemetry.client,
            ) as generation,
        ):
            assert generation is not None
            generation.update(output={"answer": run_id})
            project.telemetry.score_current_trace(score=NumericScore(name=trace_score_name, value=0.91))
            project.telemetry.score_current_span(score=BooleanScore(name=span_score_name, value=True))
            project.telemetry.emit_guardrail_observation(names["guardrail"], output={"allowed": True})

        await project.telemetry.flush()
        observations = await _wait_for_observations(
            project.telemetry,
            trace_id=trace_id,
            expected_names=frozenset(names.values()),
            required_output_names=frozenset({names["generation"]}),
            window_start=window_start,
            window_end=window_end,
        )
        scores = await _wait_for_scores(
            project.telemetry,
            trace_id=trace_id,
            expected_names=frozenset({trace_score_name, span_score_name}),
            window_start=window_start,
            window_end=window_end,
        )
        root = observations[names["root"]]
        node = observations[names["node"]]
        generation_observation = observations[names["generation"]]
        guardrail = observations[names["guardrail"]]
        assert node.parent_observation_id == root.id
        assert generation_observation.parent_observation_id == node.id
        assert guardrail.parent_observation_id == node.id
        assert generation_observation.type == "GENERATION"
        assert guardrail.type == "GUARDRAIL"
        assert _api_json_value(generation_observation.output) == {"answer": run_id}
        trace_score = scores[trace_score_name]
        span_score = scores[span_score_name]
        assert trace_score.data_type == "NUMERIC"
        assert trace_score.value == pytest.approx(0.91)
        assert trace_score.subject is not None
        assert trace_score.subject.kind == "trace"
        assert trace_score.subject.id == trace_id
        assert span_score.data_type == "BOOLEAN"
        assert span_score.value is True
        assert span_score.subject is not None
        assert span_score.subject.kind == "observation"
        assert span_score.subject.id == generation_observation.id
