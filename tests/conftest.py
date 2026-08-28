# ruff: noqa: INP001
"""Shared isolation for process-global telemetry state."""

from collections.abc import Iterator

import pytest
from langfuse._client.resource_manager import LangfuseResourceManager

from digintlab_langfuse import _instrumentation as instrumentation_module
from digintlab_langfuse import _lifecycle as lifecycle_module


@pytest.fixture(autouse=True)
def isolate_process_telemetry_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep process-global SDK and instrumentation ownership isolated per test."""
    monkeypatch.setattr(lifecycle_module, "_initialized_public_keys", set())
    monkeypatch.setattr(instrumentation_module, "_process_instrumentation_leases", {})
    yield

    with LangfuseResourceManager._lock:
        resources = tuple(LangfuseResourceManager._instances.values())
        LangfuseResourceManager._instances.clear()
    for resource in resources:
        if not resource._shutdown:
            resource.shutdown()
