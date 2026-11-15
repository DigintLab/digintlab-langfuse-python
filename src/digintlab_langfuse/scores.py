"""Score helpers for Langfuse trace, observation, span, and session scores."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

from langfuse import Langfuse

from digintlab_langfuse.json_values import JsonValue, to_json_value
from digintlab_langfuse.timing import ElapsedTimer, elapsed_timer

__all__ = [
    "BooleanScore",
    "CategoricalScore",
    "CorrectionScore",
    "NumericScore",
    "Score",
    "ScoreTarget",
    "ScoreValue",
    "TextScore",
    "emit_score",
    "score_current_span",
    "score_current_trace",
    "score_elapsed",
]

type SDKNumericScoreDataType = Literal["NUMERIC", "BOOLEAN"]
type SDKStringScoreDataType = Literal["CATEGORICAL", "TEXT", "CORRECTION"]
type ScoreValue = float | str | bool


class Score[TValue: ScoreValue](Protocol):
    """Typed score payload that knows how to emit itself to Langfuse."""

    @property
    def name(self) -> str:
        """Return the Langfuse score name."""
        ...

    @property
    def value(self) -> TValue:
        """Return the typed score value."""
        ...

    def emit(self, lf: Langfuse, target: "ScoreTarget") -> None:
        """Emit this score to Langfuse."""
        ...

    def score_current_trace(self, lf: Langfuse) -> None:
        """Attach this score to the active Langfuse trace."""
        ...

    def score_current_span(self, lf: Langfuse) -> None:
        """Attach this score to the active Langfuse span or generation."""
        ...


class CommonScoreKwargs(TypedDict, total=False):
    """Optional keyword arguments shared by Langfuse score overloads."""

    session_id: str
    dataset_run_id: str
    trace_id: str
    observation_id: str
    score_id: str
    comment: str
    config_id: str
    metadata: dict[str, JsonValue]


class NumericScoreKwargs(CommonScoreKwargs):
    """Keyword arguments for numeric Langfuse score overloads."""

    name: str
    value: float
    data_type: SDKNumericScoreDataType


class StringScoreKwargs(CommonScoreKwargs):
    """Keyword arguments for string Langfuse score overloads."""

    name: str
    value: str
    data_type: SDKStringScoreDataType


@dataclass(frozen=True, slots=True)
class ScoreTarget:
    """Trace, observation, session, and metadata target for one Langfuse score."""

    run_id: str | None = None
    trace_id: str | None = None
    observation_id: str | None = None
    session_id: str | None = None
    dataset_run_id: str | None = None
    score_id: str | None = None
    comment: str | None = None
    config_id: str | None = None
    metadata: Mapping[str, JsonValue] | None = None

    @property
    def resolved_session_id(self) -> str | None:
        """Return the explicit session id, or run id when scoring a session."""
        return self.session_id or (self.run_id if self.run_id and not self.trace_id else None)

    @property
    def score_metadata(self) -> dict[str, JsonValue] | None:
        """Return merged score metadata with run/session ids when available."""
        merged: dict[str, JsonValue] = {}
        if self.run_id:
            merged["run_id"] = self.run_id
            merged["session_id"] = self.run_id
        if self.metadata:
            merged.update({str(key): to_json_value(value) for key, value in self.metadata.items()})
        return merged or None

    def add_to(self, kwargs: CommonScoreKwargs) -> None:
        """Populate optional Langfuse score kwargs while omitting absent fields."""
        session_id = self.resolved_session_id
        if session_id:
            kwargs["session_id"] = session_id
        if self.dataset_run_id:
            kwargs["dataset_run_id"] = self.dataset_run_id
        if self.trace_id:
            kwargs["trace_id"] = self.trace_id
        if self.observation_id:
            kwargs["observation_id"] = self.observation_id
        if self.score_id:
            kwargs["score_id"] = self.score_id
        if self.comment is not None:
            kwargs["comment"] = self.comment
        if self.config_id:
            kwargs["config_id"] = self.config_id
        metadata = self.score_metadata
        if metadata is not None:
            kwargs["metadata"] = metadata


@dataclass(frozen=True, slots=True)
class NumericScore(Score[float]):
    """Numeric Langfuse score."""

    name: str
    value: float

    def emit(self, lf: Langfuse, target: ScoreTarget) -> None:
        """Emit this numeric score to Langfuse."""
        kwargs: NumericScoreKwargs = {"name": self.name, "value": self.value, "data_type": "NUMERIC"}
        target.add_to(kwargs)
        lf.create_score(**kwargs)

    def score_current_trace(self, lf: Langfuse) -> None:
        """Attach this numeric score to the active Langfuse trace."""
        lf.score_current_trace(name=self.name, value=self.value, data_type="NUMERIC")

    def score_current_span(self, lf: Langfuse) -> None:
        """Attach this numeric score to the active Langfuse span or generation."""
        lf.score_current_span(name=self.name, value=self.value, data_type="NUMERIC")


@dataclass(frozen=True, slots=True)
class BooleanScore(Score[bool]):
    """Boolean Langfuse score."""

    name: str
    value: bool

    def emit(self, lf: Langfuse, target: ScoreTarget) -> None:
        """Emit this boolean score to Langfuse."""
        kwargs: NumericScoreKwargs = {"name": self.name, "value": 1.0 if self.value else 0.0, "data_type": "BOOLEAN"}
        target.add_to(kwargs)
        lf.create_score(**kwargs)

    def score_current_trace(self, lf: Langfuse) -> None:
        """Attach this boolean score to the active Langfuse trace."""
        lf.score_current_trace(name=self.name, value=1.0 if self.value else 0.0, data_type="BOOLEAN")

    def score_current_span(self, lf: Langfuse) -> None:
        """Attach this boolean score to the active Langfuse span or generation."""
        lf.score_current_span(name=self.name, value=1.0 if self.value else 0.0, data_type="BOOLEAN")


@dataclass(frozen=True, slots=True)
class CategoricalScore(Score[str]):
    """Categorical Langfuse score."""

    name: str
    value: str

    def emit(self, lf: Langfuse, target: ScoreTarget) -> None:
        """Emit this categorical score to Langfuse."""
        kwargs: StringScoreKwargs = {"name": self.name, "value": self.value, "data_type": "CATEGORICAL"}
        target.add_to(kwargs)
        lf.create_score(**kwargs)

    def score_current_trace(self, lf: Langfuse) -> None:
        """Attach this categorical score to the active Langfuse trace."""
        lf.score_current_trace(name=self.name, value=self.value, data_type="CATEGORICAL")

    def score_current_span(self, lf: Langfuse) -> None:
        """Attach this categorical score to the active Langfuse span or generation."""
        lf.score_current_span(name=self.name, value=self.value, data_type="CATEGORICAL")


@dataclass(frozen=True, slots=True)
class TextScore(Score[str]):
    """Text Langfuse score."""

    name: str
    value: str

    def emit(self, lf: Langfuse, target: ScoreTarget) -> None:
        """Emit this text score to Langfuse."""
        kwargs: StringScoreKwargs = {"name": self.name, "value": self.value, "data_type": "TEXT"}
        target.add_to(kwargs)
        lf.create_score(**kwargs)

    def score_current_trace(self, lf: Langfuse) -> None:
        """Attach this text score to the active Langfuse trace."""
        lf.score_current_trace(name=self.name, value=self.value, data_type="TEXT")

    def score_current_span(self, lf: Langfuse) -> None:
        """Attach this text score to the active Langfuse span or generation."""
        lf.score_current_span(name=self.name, value=self.value, data_type="TEXT")


@dataclass(frozen=True, slots=True)
class CorrectionScore(Score[str]):
    """Correction Langfuse score."""

    name: str
    value: str

    def emit(self, lf: Langfuse, target: ScoreTarget) -> None:
        """Emit this correction score to Langfuse."""
        kwargs: StringScoreKwargs = {"name": self.name, "value": self.value, "data_type": "CORRECTION"}
        target.add_to(kwargs)
        lf.create_score(**kwargs)

    def score_current_trace(self, lf: Langfuse) -> None:
        """Attach this correction score to the active Langfuse trace."""
        lf.score_current_trace(name=self.name, value=self.value, data_type="CORRECTION")

    def score_current_span(self, lf: Langfuse) -> None:
        """Attach this correction score to the active Langfuse span or generation."""
        lf.score_current_span(name=self.name, value=self.value, data_type="CORRECTION")


DEFAULT_SCORE_TARGET = ScoreTarget()


def emit_score[TValue: ScoreValue](
    lf: Langfuse,
    *,
    score: Score[TValue],
    target: ScoreTarget = DEFAULT_SCORE_TARGET,
) -> None:
    """Emit a typed Langfuse score."""
    score.emit(lf, target)


@contextmanager
def score_elapsed(
    lf: Langfuse,
    *,
    name: str,
    target: ScoreTarget = DEFAULT_SCORE_TARGET,
    digits: int = 3,
) -> Iterator[ElapsedTimer]:
    """Measure a block and emit its elapsed seconds as a numeric Langfuse score."""
    with elapsed_timer() as timer:
        yield timer
    emit_score(lf, score=NumericScore(name=name, value=timer.rounded(digits)), target=target)


def score_current_trace[TValue: ScoreValue](lf: Langfuse, *, score: Score[TValue]) -> None:
    """Attach a typed score to the trace active in the current context."""
    score.score_current_trace(lf)


def score_current_span[TValue: ScoreValue](lf: Langfuse, *, score: Score[TValue]) -> None:
    """Attach a typed score to the active Langfuse span or generation."""
    score.score_current_span(lf)
