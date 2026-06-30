"""Generic numeric budget checks for observed runtime, cost, and quality measurements."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from digintlab_langfuse.types import JsonValue


@dataclass(frozen=True, slots=True)
class NumericBudgetViolation:
    """One missing or over-budget numeric measurement."""

    name: str
    ceiling: float
    observed: float | None = None

    @property
    def message(self) -> str:
        """Return a compact diagnostic suitable for CI output."""
        if self.observed is None:
            return f"{self.name} missing"
        return f"{self.observed} > {self.ceiling}"


def _numeric_value(value: JsonValue, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        message = f"Budget ceiling {name!r} must be a number."
        raise TypeError(message)
    return float(value)


def validate_numeric_budget_ceilings(
    raw: Mapping[str, JsonValue],
    *,
    allowed_names: Collection[str] | None = None,
) -> dict[str, float]:
    """Return validated numeric ceilings, rejecting unknown or non-numeric budget entries."""
    if allowed_names is not None:
        unknown_names = frozenset(raw) - frozenset(allowed_names)
        if unknown_names:
            message = f"Unknown budget ceiling names: {', '.join(sorted(unknown_names))}."
            raise ValueError(message)

    return {name: _numeric_value(value, name=name) for name, value in raw.items()}


def check_numeric_budgets(
    measurements: Mapping[str, float],
    ceilings: Mapping[str, float],
) -> tuple[NumericBudgetViolation, ...]:
    """Return missing or exceeded numeric budget violations."""
    violations: list[NumericBudgetViolation] = []
    for name, ceiling in ceilings.items():
        if name not in measurements:
            violations.append(NumericBudgetViolation(name=name, ceiling=ceiling))
            continue
        observed = measurements[name]
        if observed > ceiling:
            violations.append(NumericBudgetViolation(name=name, ceiling=ceiling, observed=observed))
    return tuple(violations)
