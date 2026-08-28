"""Generic numeric budget checks for observed runtime, cost, and quality measurements."""

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from digintlab_langfuse.json_values import JsonValue


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
    numeric = float(value)
    if not math.isfinite(numeric):
        message = f"Budget value {name!r} must be finite."
        raise ValueError(message)
    return numeric


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
        validated_ceiling = _numeric_value(ceiling, name=name)
        if name not in measurements:
            violations.append(NumericBudgetViolation(name=name, ceiling=validated_ceiling))
            continue
        observed = _numeric_value(measurements[name], name=name)
        if observed > validated_ceiling:
            violations.append(NumericBudgetViolation(name=name, ceiling=validated_ceiling, observed=observed))
    return tuple(violations)
