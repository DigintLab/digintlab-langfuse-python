"""Network-free integration tests against the real Langfuse SDK exporter."""

import json
from uuid import uuid4

import pytest
from langfuse import Langfuse, get_client
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from digintlab_langfuse import (
    LangfuseClientConfig,
    SensitiveDataRedactionConfig,
    init_langfuse_telemetry,
)
from digintlab_langfuse._lifecycle import flush_client, shutdown_client
from digintlab_langfuse.client import _init_langfuse_client
from digintlab_langfuse.spans import emit_guardrail_observation


@pytest.mark.asyncio
async def test_real_sdk_exports_closed_parented_and_optionally_redacted_observations() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    public_key = f"pk-integration-{uuid4()}"
    telemetry = init_langfuse_telemetry(
        LangfuseClientConfig(
            public_key=public_key,
            secret_key="integration-secret",
            span_exporter=exporter,
            tracer_provider=provider,
            sensitive_data_redaction=SensitiveDataRedactionConfig(),
        ),
        instrument_httpx=False,
    )
    alias = get_client(public_key=public_key)

    async with telemetry:
        with telemetry.trace_call(
            name="integration-root",
            session_id="session-1",
            user_id="user-1",
            root_observation=True,
            root_input={"email": "person@example.com", "password": "secret"},
            root_metadata={"failure": "Bearer root-secret"},
        ):
            with telemetry.observe_node_span("integration-node", enabled=True):
                emit_guardrail_observation(
                    "integration-guardrail",
                    output={"blocked": False},
                    client_getter=lambda: alias,
                )
            with provider.get_tracer("litellm").start_as_current_span(
                "raw-generation",
                attributes={
                    "user.email": "person@example.com",
                    "authorization": "raw-secret",
                    "gen_ai.prompt": '{"password":"plain-secret"}',
                },
            ):
                pass

    spans = {span.name: span for span in exporter.get_finished_spans()}
    root = spans["integration-root"]
    node = spans["integration-node"]
    guardrail = spans["integration-guardrail"]
    raw_generation = spans["raw-generation"]

    assert root.end_time is not None
    assert node.end_time is not None
    assert node.parent is not None
    assert node.parent.span_id == root.context.span_id
    assert guardrail.parent is not None
    assert guardrail.parent.span_id == node.context.span_id
    assert root.attributes is not None
    assert json.loads(str(root.attributes["langfuse.observation.input"])) == {
        "email": "[REDACTED]",
        "password": "[REDACTED]",
    }
    assert root.attributes["langfuse.observation.metadata.failure"] == "[REDACTED]"
    assert raw_generation.attributes is not None
    assert raw_generation.attributes["user.email"] == "[REDACTED]"
    assert raw_generation.attributes["authorization"] == "[REDACTED]"
    assert json.loads(str(raw_generation.attributes["gen_ai.prompt"])) == {"password": "[REDACTED]"}


@pytest.mark.asyncio
async def test_sdk_aliases_share_lifecycle_state() -> None:
    public_key = f"pk-alias-{uuid4()}"
    client = _init_langfuse_client(
        LangfuseClientConfig(
            public_key=public_key,
            secret_key="integration-secret",
            span_exporter=InMemorySpanExporter(),
        )
    )
    alias = get_client(public_key=public_key)

    assert alias is not client
    await shutdown_client(client)
    with pytest.raises(RuntimeError, match="after shutdown"):
        await flush_client(alias)


def test_initializer_rejects_a_preexisting_same_key_sdk_owner() -> None:
    exporter = InMemorySpanExporter()
    public_key = f"pk-external-{uuid4()}"
    external = Langfuse(
        public_key=public_key,
        secret_key="external-secret",
        base_url="https://external.example",
        span_exporter=exporter,
    )
    try:
        with pytest.raises(RuntimeError, match="outside digintlab_langfuse"):
            init_langfuse_telemetry(
                LangfuseClientConfig(
                    public_key=public_key,
                    secret_key="library-secret",
                    sensitive_data_redaction=SensitiveDataRedactionConfig(),
                )
            )
    finally:
        external.shutdown()
