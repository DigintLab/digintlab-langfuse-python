"""OpenTelemetry span timing aggregation helpers."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock

from opentelemetry import context as context_api
from opentelemetry.sdk.trace import ReadableSpan, Span, TracerProvider
from opentelemetry.sdk.trace.export import SpanProcessor

SPAN_TIMING_OVERFLOW_KEY = "__overflow__"
DEFAULT_MAX_SPAN_TIMING_BUCKETS = 1_000


def _monotonic() -> float:
    return time.monotonic()


@dataclass(slots=True)
class ElapsedTimer:
    """Small monotonic elapsed-time recorder for local code sections."""

    _started_at: float = field(default_factory=_monotonic)
    _stopped_at: float | None = None

    @property
    def elapsed_seconds(self) -> float:
        """Return stopped elapsed time, or live elapsed time while running."""
        stopped_at = self._stopped_at if self._stopped_at is not None else time.monotonic()
        return stopped_at - self._started_at

    def rounded(self, digits: int = 3) -> float:
        """Return elapsed seconds rounded for metric payloads."""
        return round(self.elapsed_seconds, digits)

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self._stopped_at is None:
            self._stopped_at = time.monotonic()
        return self.elapsed_seconds


@dataclass(frozen=True, slots=True)
class SpanTimingBucket:
    """Aggregated elapsed time for a group of ended OpenTelemetry spans."""

    key: str
    count: int
    total_elapsed_seconds: float
    max_elapsed_seconds: float


@contextmanager
def elapsed_timer() -> Iterator[ElapsedTimer]:
    """Measure elapsed seconds for a local code block."""
    timer = ElapsedTimer()
    try:
        yield timer
    finally:
        timer.stop()


def default_span_timing_key(span: ReadableSpan) -> str:
    """Return the default span timing bucket key for a completed span."""
    scope = span.instrumentation_scope
    return f"{scope.name}:{span.name}" if scope and scope.name else span.name


def _span_elapsed_seconds(span: ReadableSpan) -> float | None:
    if span.start_time is None or span.end_time is None or span.end_time < span.start_time:
        return None
    return (span.end_time - span.start_time) / 1_000_000_000


class SpanTimingProcessor(SpanProcessor):
    """Collect elapsed-time buckets for ended OpenTelemetry spans."""

    def __init__(
        self,
        *,
        key: Callable[[ReadableSpan], str] = default_span_timing_key,
        max_buckets: int = DEFAULT_MAX_SPAN_TIMING_BUCKETS,
    ) -> None:
        if max_buckets < 1:
            message = "Span timing max_buckets must be at least 1."
            raise ValueError(message)
        self._key = key
        self._max_buckets = max_buckets
        self._buckets: dict[str, SpanTimingBucket] = {}
        self._lock = Lock()

    def on_start(self, span: Span, parent_context: context_api.Context | None = None) -> None:
        """No-op start hook required by the OpenTelemetry processor protocol."""

    def on_end(self, span: ReadableSpan) -> None:
        """Aggregate one completed span duration."""
        elapsed = _span_elapsed_seconds(span)
        if elapsed is None:
            return
        key = self._key(span)
        if not key:
            return

        with self._lock:
            if key not in self._buckets and len(self._buckets) >= self._max_buckets:
                key = SPAN_TIMING_OVERFLOW_KEY
            if key in self._buckets:
                current = self._buckets[key]
                self._buckets[key] = SpanTimingBucket(
                    key=key,
                    count=current.count + 1,
                    total_elapsed_seconds=current.total_elapsed_seconds + elapsed,
                    max_elapsed_seconds=max(current.max_elapsed_seconds, elapsed),
                )
            else:
                self._buckets[key] = SpanTimingBucket(
                    key=key,
                    count=1,
                    total_elapsed_seconds=elapsed,
                    max_elapsed_seconds=elapsed,
                )

    def snapshot(self, *, reset: bool = False) -> tuple[SpanTimingBucket, ...]:
        """Return current buckets, optionally resetting accumulated timings."""
        with self._lock:
            buckets = tuple(self._buckets[key] for key in sorted(self._buckets))
            if reset:
                self._buckets.clear()
            return buckets

    def reset(self) -> None:
        """Clear accumulated timing buckets."""
        with self._lock:
            self._buckets.clear()

    def shutdown(self) -> None:
        """Release accumulated timings during provider shutdown."""
        self.reset()

    def force_flush(self, _timeout_millis: int = 30_000) -> bool:
        """No buffered export work exists, so flushing always succeeds."""
        return True


@dataclass(frozen=True, slots=True)
class SpanTimingRecorder:
    """Tracer provider plus timing processor installed on it."""

    tracer_provider: TracerProvider
    processor: SpanTimingProcessor

    def snapshot(self, *, reset: bool = False) -> tuple[SpanTimingBucket, ...]:
        """Return current timing buckets, optionally resetting accumulated timings."""
        return self.processor.snapshot(reset=reset)

    def reset(self) -> None:
        """Clear accumulated timing buckets."""
        self.processor.reset()


def create_span_timing_recorder(
    *,
    key: Callable[[ReadableSpan], str] = default_span_timing_key,
    tracer_provider: TracerProvider | None = None,
    max_buckets: int = DEFAULT_MAX_SPAN_TIMING_BUCKETS,
) -> SpanTimingRecorder:
    """Create or extend a tracer provider with a span timing processor."""
    provider = tracer_provider or TracerProvider()
    processor = SpanTimingProcessor(key=key, max_buckets=max_buckets)
    provider.add_span_processor(processor)
    return SpanTimingRecorder(tracer_provider=provider, processor=processor)
