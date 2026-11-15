"""Contract tests for the private Langfuse SDK internals this package relies on.

Each failure names the attribute, the dependent module, and the required action so a
Langfuse version bump surfaces breakage here instead of at runtime.
"""

import importlib
import threading
from uuid import uuid4

from langfuse import Langfuse, get_client
from langfuse._client.resource_manager import LangfuseResourceManager
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from digintlab_langfuse import LangfuseClientConfig
from digintlab_langfuse.client import _init_langfuse_client

_UPGRADE_HINT = "update digintlab_langfuse before bumping langfuse"


def _message(attribute: str, module: str, expectation: str) -> str:
    return f"{attribute} {expectation}; digintlab_langfuse.{module} depends on it: {_UPGRADE_HINT}."


def _build_client() -> tuple[str, Langfuse]:
    public_key = f"pk-internals-{uuid4()}"
    client = _init_langfuse_client(
        LangfuseClientConfig(
            public_key=public_key,
            secret_key="internals-secret",
            span_exporter=InMemorySpanExporter(),
        )
    )
    return public_key, client


def test_resource_manager_lock_is_a_reentrant_context_manager() -> None:
    lock = LangfuseResourceManager._lock
    expectation = "must be a reentrant lock usable as a context manager (initialize_client holds it while Langfuse() re-acquires it)"
    message = _message("LangfuseResourceManager._lock", "_lifecycle:58", expectation)

    assert isinstance(lock, type(threading.RLock())), message
    with lock, lock:
        pass


def test_resource_manager_instances_is_a_dict_keyed_by_public_key() -> None:
    public_key, _client = _build_client()
    instances = LangfuseResourceManager._instances

    assert isinstance(instances, dict), _message("LangfuseResourceManager._instances", "_lifecycle:74", "must be a dict")
    assert public_key in instances, _message("LangfuseResourceManager._instances", "_lifecycle:83", "must be keyed by the client's public key")
    assert isinstance(instances[public_key], LangfuseResourceManager), _message(
        "LangfuseResourceManager._instances", "_lifecycle:83", "must map public keys to LangfuseResourceManager"
    )


def test_resource_manager_instances_supports_key_deletion() -> None:
    public_key, _client = _build_client()
    instances = LangfuseResourceManager._instances

    with LangfuseResourceManager._lock:
        del instances[public_key]

    assert not instances, _message("LangfuseResourceManager._instances", "_lifecycle:84", "must support `del` by public key")


def test_langfuse_resources_attribute_is_the_shared_resource_manager() -> None:
    public_key, client = _build_client()
    alias = get_client(public_key=public_key)

    assert hasattr(client, "_resources"), _message("Langfuse._resources", "_lifecycle:38 and spans:131", "must exist on Langfuse")
    assert isinstance(client._resources, LangfuseResourceManager), _message(
        "Langfuse._resources", "_lifecycle:38 and spans:131", "must be a LangfuseResourceManager"
    )
    assert alias is not client, _message("get_client()", "_lifecycle:38", "must return a distinct facade for an existing key")
    assert alias._resources is client._resources, _message(
        "Langfuse._resources", "_lifecycle:38 and spans:131", "must be shared by every facade of one public key"
    )


def test_resource_manager_shutdown_flag_and_method() -> None:
    _public_key, client = _build_client()
    resources = client._resources

    assert resources is not None
    flag = resources._shutdown
    assert flag is False, _message("LangfuseResourceManager._shutdown", "tests/conftest.py cleanup", "must be a bool, False for a live client")
    assert callable(resources.shutdown), _message("LangfuseResourceManager.shutdown", "tests/conftest.py cleanup", "must be callable")

    resources.shutdown()

    assert resources._shutdown is True, _message(
        "LangfuseResourceManager._shutdown", "tests/conftest.py cleanup", "must become True after shutdown()"
    )


def test_langfuse_observation_wrapper_is_importable_from_private_module() -> None:
    module = importlib.import_module("langfuse._client.span")

    assert hasattr(module, "LangfuseObservationWrapper"), _message(
        "langfuse._client.span.LangfuseObservationWrapper", "callbacks:31", "must be importable"
    )
