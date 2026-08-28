"""Recursive JSON values and normalization helpers."""

import math
from collections.abc import Mapping, Sequence
from enum import Enum

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


def to_json_value(value: object) -> JsonValue:
    """Convert an arbitrary value into a recursive JSON value."""
    match value:
        case Enum():
            return to_json_value(value.value)
        case float() if not math.isfinite(value):
            message = "JSON floating-point values must be finite."
            raise ValueError(message)
        case None | str() | int() | float() | bool():
            return value
        case Mapping():
            return {str(key): to_json_value(item) for key, item in value.items()}
        case Sequence() if not isinstance(value, str | bytes | bytearray):
            return [to_json_value(item) for item in value]
        case _:
            return str(value)
