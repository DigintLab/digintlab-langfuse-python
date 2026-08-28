"""Langfuse SDK resource ownership and asynchronous lifecycle operations."""

import asyncio
import math
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from threading import Lock, RLock
from weakref import WeakKeyDictionary

from langfuse import Langfuse
from langfuse._client.resource_manager import LangfuseResourceManager

DEFAULT_OPERATION_TIMEOUT_SECONDS = 30.0

_client_initialization_lock = RLock()
_initialized_public_keys: set[str] = set()
_client_lifecycle_states_lock = Lock()


@dataclass(slots=True)
class ClientLifecycleState:
    """Shared lifecycle state for SDK facades backed by the same resources."""

    guard: Lock = field(default_factory=Lock)
    executor: ThreadPoolExecutor = field(default_factory=lambda: ThreadPoolExecutor(max_workers=1, thread_name_prefix="langfuse-lifecycle"))
    flush_future: Future[None] | None = None
    shutdown_future: Future[None] | None = None
    shutdown_complete: bool = False


_client_lifecycle_states: WeakKeyDictionary[object, ClientLifecycleState] = WeakKeyDictionary()


def lifecycle_state(lf: Langfuse) -> ClientLifecycleState:
    """Return lifecycle state shared by all facades for one SDK resource owner."""
    resources = lf._resources  # noqa: SLF001 - exact-pinned SDK facades share this lifecycle owner.
    owner: object = resources if resources is not None else lf
    with _client_lifecycle_states_lock:
        state = _client_lifecycle_states.get(owner)
        if state is None:
            state = ClientLifecycleState()
            _client_lifecycle_states[owner] = state
        return state


def validate_operation_timeout(timeout_seconds: float) -> float:
    """Return a valid caller-visible lifecycle timeout."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        message = "Telemetry operation timeouts must be finite and greater than zero."
        raise ValueError(message)
    return timeout_seconds


def initialize_client(public_key: str, factory: Callable[[], Langfuse]) -> Langfuse:
    """Create the sole Langfuse SDK resource owner for this process."""
    with _client_initialization_lock, LangfuseResourceManager._lock:  # noqa: SLF001 - closes the SDK singleton race.
        _reserve_public_key(public_key)
        try:
            return factory()
        except BaseException:
            _initialized_public_keys.discard(public_key)
            raise


def _reserve_public_key(public_key: str) -> None:
    if public_key in _initialized_public_keys:
        message = f"Langfuse client {public_key!r} has already been initialized in this process."
        raise RuntimeError(message)
    if _initialized_public_keys:
        message = "Only one Langfuse project may own telemetry resources in a process."
        raise RuntimeError(message)
    if LangfuseResourceManager._instances:  # noqa: SLF001 - SDK exposes no public ownership registry.
        message = "A Langfuse SDK resource was initialized outside digintlab_langfuse; explicit ownership cannot be guaranteed."
        raise RuntimeError(message)
    _initialized_public_keys.add(public_key)


def release_failed_owner(public_key: str) -> None:
    """Discard a successfully shut down client whose owner setup failed."""
    with _client_initialization_lock, LangfuseResourceManager._lock:  # noqa: SLF001 - SDK exposes no public ownership registry.
        if public_key in LangfuseResourceManager._instances:  # noqa: SLF001 - remove the successfully shut down failed owner.
            del LangfuseResourceManager._instances[public_key]  # noqa: SLF001 - remove the successfully shut down failed owner.
        _initialized_public_keys.discard(public_key)


def _run_shutdown(lf: Langfuse, state: ClientLifecycleState) -> None:
    lf.shutdown()
    with state.guard:
        state.shutdown_complete = True
    state.executor.shutdown(wait=False)


def _schedule_flush(lf: Langfuse) -> Future[None]:
    state = lifecycle_state(lf)
    with state.guard:
        if state.shutdown_complete:
            message = "Cannot flush Langfuse telemetry after shutdown."
            raise RuntimeError(message)
        if state.shutdown_future is not None:
            message = "Cannot flush Langfuse telemetry after shutdown has started."
            raise RuntimeError(message)
        if state.flush_future is None or state.flush_future.done():
            state.flush_future = state.executor.submit(lf.flush)
        return state.flush_future


def _schedule_shutdown(lf: Langfuse) -> Future[None]:
    state = lifecycle_state(lf)
    with state.guard:
        existing = state.shutdown_future
        if existing is None or (existing.done() and existing.exception() is not None):
            existing = state.executor.submit(_run_shutdown, lf, state)
            state.shutdown_future = existing
        return existing


def _release_failed_owner_after_shutdown(public_key: str, future: Future[None]) -> None:
    if future.cancelled() or future.exception() is not None:
        return
    release_failed_owner(public_key)


def shutdown_failed_client(
    lf: Langfuse,
    *,
    public_key: str,
    timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS,
) -> None:
    """Bound failed-initialization cleanup and release ownership after shutdown."""
    timeout = validate_operation_timeout(timeout_seconds)
    future = _schedule_shutdown(lf)
    try:
        future.result(timeout=timeout)
    except TimeoutError:
        future.add_done_callback(partial(_release_failed_owner_after_shutdown, public_key))
        raise
    release_failed_owner(public_key)


async def _await_future(future: Future[None], *, timeout_seconds: float) -> None:
    await asyncio.wait_for(asyncio.shield(asyncio.wrap_future(future)), timeout=timeout_seconds)


async def flush_client(lf: Langfuse, *, timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS) -> None:
    """Flush Langfuse telemetry within a caller-visible deadline."""
    timeout = validate_operation_timeout(timeout_seconds)
    await _await_future(_schedule_flush(lf), timeout_seconds=timeout)


async def shutdown_client(lf: Langfuse, *, timeout_seconds: float = DEFAULT_OPERATION_TIMEOUT_SECONDS) -> None:
    """Shutdown Langfuse telemetry without blocking the event loop indefinitely."""
    timeout = validate_operation_timeout(timeout_seconds)
    await _await_future(_schedule_shutdown(lf), timeout_seconds=timeout)


def shutdown_started(lf: Langfuse) -> bool:
    """Return whether shutdown has started for this SDK resource owner."""
    state = lifecycle_state(lf)
    with state.guard:
        return state.shutdown_future is not None
