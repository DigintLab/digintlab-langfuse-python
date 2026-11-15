"""Offline regression checks for upstream score delivery and Bedrock usage telemetry."""

import json
import warnings
from uuid import uuid4

import botocore.session
import httpx
import pytest
from botocore.client import ConverseRequest, ConverseResponse
from botocore.stub import Stubber
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes.gen_ai_attributes import (
    GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS,
    GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS,
)

from digintlab_langfuse import LangfuseClientConfig, init_langfuse_telemetry
from digintlab_langfuse.scores import NumericScore, ScoreTarget


@pytest.mark.asyncio
async def test_flush_delivers_pending_scores_without_waiting_for_batch_interval() -> None:
    """A library flush should drain SDK score tasks, not just OTEL spans."""
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(202, json={"message": "Accepted"})

    # The pinned Langfuse SDK calls archived backoff 2.2.1 from its score worker.
    # Record the known upstream warning; keep all unexpected warnings actionable.
    with warnings.catch_warnings(record=True) as observed_warnings:
        warnings.simplefilter("always", DeprecationWarning)
        with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
            telemetry = init_langfuse_telemetry(
                LangfuseClientConfig(
                    public_key=f"pk-score-flush-{uuid4()}",
                    secret_key="test-secret",
                    httpx_client=http_client,
                    span_exporter=InMemorySpanExporter(),
                    flush_at=100,
                    flush_interval=60.0,
                ),
            )
            async with telemetry:
                telemetry.emit_score(
                    score=NumericScore(name="quality", value=0.9),
                    target=ScoreTarget(trace_id="a" * 32),
                )
                await telemetry.flush()
                score_requests = [request for request in requests if request.url.path == "/api/public/scores"]
                assert len(score_requests) == 1
                scores = json.loads(score_requests[0].content)
                assert len(scores) == 1
                assert scores[0]["traceId"] == "a" * 32
                assert scores[0]["name"] == "quality"
                assert scores[0]["value"] == 0.9
                assert scores[0]["dataType"] == "NUMERIC"
    
    unexpected_warnings = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in observed_warnings
        if not (
            issubclass(warning.category, DeprecationWarning)
            and warning.filename.replace("\\", "/").endswith("/backoff/_decorator.py")
            and "asyncio.iscoroutinefunction" in str(warning.message)
        )
    ]
    assert not unexpected_warnings

@pytest.mark.asyncio
async def test_bedrock_converse_preserves_prompt_cache_usage_in_exported_span() -> None:
    """The package's opt-in botocore instrumentation retains upstream cache counters."""
    exporter = InMemorySpanExporter()
    telemetry = init_langfuse_telemetry(
        LangfuseClientConfig(
            public_key=f"pk-bedrock-cache-{uuid4()}",
            secret_key="test-secret",
            tracer_provider=TracerProvider(),
            span_exporter=exporter,
        ),
        instrument_botocore=True,
    )
    session = botocore.session.get_session()
    client = session.create_client(
        "bedrock-runtime",
        region_name="us-east-1",
        aws_access_key_id="test-access-key",
        aws_secret_access_key="test-secret-key",
    )
    parameters: ConverseRequest = {
        "modelId": "amazon.nova-micro-v1:0",
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
    }
    response: ConverseResponse = {
        "output": {"message": {"role": "assistant", "content": [{"text": "hello"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": 20,
            "outputTokens": 5,
            "totalTokens": 25,
            "cacheReadInputTokens": 7,
            "cacheWriteInputTokens": 13,
        },
        "metrics": {"latencyMs": 1},
    }
    async with telemetry:
        with Stubber(client) as stubber:
            stubber.add_response("converse", response, parameters)
            result = client.converse(**parameters)
        assert result["usage"]["cacheReadInputTokens"] == 7
        await telemetry.flush()

    spans = exporter.get_finished_spans()
    matching = [span for span in spans if span.attributes is not None and GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS in span.attributes]
    assert len(matching) == 1
    assert matching[0].attributes is not None
    assert matching[0].attributes[GEN_AI_USAGE_CACHE_READ_INPUT_TOKENS] == 7
    assert matching[0].attributes[GEN_AI_USAGE_CACHE_CREATION_INPUT_TOKENS] == 13
