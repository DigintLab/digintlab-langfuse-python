"""Opt-in generic masking for common sensitive fields and PII strings."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING

from langfuse.types import (
    MaskFunction,
    MaskOtelSpansFunction,
    MaskOtelSpansParams,
    MaskOtelSpansResult,
    OtelSpanPatch,
)

if TYPE_CHECKING:
    from opentelemetry.util.types import AttributeValue
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine

    from digintlab_langfuse.json_values import JsonValue

DEFAULT_SENSITIVE_FIELD_NAMES = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "authorization",
        "bearer",
        "client_secret",
        "credential",
        "credentials",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "secret_key",
        "ssws",
        "token",
    }
)
DEFAULT_PII_ENTITIES = frozenset(
    {
        "CREDIT_CARD",
        "CRYPTO",
        "EMAIL_ADDRESS",
        "IBAN_CODE",
        "IP_ADDRESS",
        "MAC_ADDRESS",
        "PHONE_NUMBER",
        "US_SSN",
    }
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+")
_REDACTION_EXTRA_ERROR = 'PII redaction requires the optional dependency: uv add "digintlab-langfuse-python[redaction]".'


def _normalized_field_name(name: str) -> str:
    snake_case_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"[^a-z0-9]+", "_", snake_case_name.casefold()).strip("_")


@dataclass(frozen=True, slots=True)
class SensitiveDataRedactionConfig:
    """Explicit policy for the optional generic sensitive-data masker."""

    replacement: str = "[REDACTED]"
    sensitive_field_names: frozenset[str] = DEFAULT_SENSITIVE_FIELD_NAMES
    pii_entities: frozenset[str] = DEFAULT_PII_ENTITIES
    pii_score_threshold: float = 0.4
    redact_bearer_tokens: bool = True

    def __post_init__(self) -> None:
        if not self.replacement:
            message = "Sensitive-data redaction replacement must not be empty."
            raise ValueError(message)
        object.__setattr__(
            self,
            "sensitive_field_names",
            frozenset(_normalized_field_name(name) for name in self.sensitive_field_names),
        )
        if not 0 <= self.pii_score_threshold <= 1:
            message = "Presidio score threshold must be between zero and one."
            raise ValueError(message)
        normalized_entities = frozenset(entity.strip().upper() for entity in self.pii_entities)
        if "" in normalized_entities:
            message = "Presidio entity names must not be empty."
            raise ValueError(message)
        object.__setattr__(self, "pii_entities", normalized_entities)
        if not normalized_entities:
            return
        supported_entities = frozenset(_presidio_analyzer().get_supported_entities("en"))
        if unsupported_entities := normalized_entities - supported_entities:
            names = ", ".join(sorted(unsupported_entities))
            message = f"Unsupported Presidio entities: {names}."
            raise ValueError(message)


@cache
def _presidio_analyzer() -> AnalyzerEngine:
    """Return the process-wide local analyzer without loading an NLP model."""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NoOpNlpEngine
        from presidio_analyzer.predefined_recognizers import EmailRecognizer
        from tldextract import TLDExtract
    except ModuleNotFoundError as error:
        raise RuntimeError(_REDACTION_EXTRA_ERROR) from error

    domain_extractor = TLDExtract(cache_dir=None, suffix_list_urls=())

    class OfflineEmailRecognizer(EmailRecognizer):
        def __init__(self) -> None:
            super().__init__(name="OfflineEmailRecognizer")

        def validate_result(self, pattern_text: str) -> bool:
            return domain_extractor(pattern_text).fqdn != ""

    nlp_engine = NoOpNlpEngine(models=[{"lang_code": "en", "model_name": "no-op"}])
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    analyzer.registry.remove_recognizer("EmailRecognizer", language="en")
    analyzer.registry.add_recognizer(OfflineEmailRecognizer())
    return analyzer


@cache
def _presidio_anonymizer() -> AnonymizerEngine:
    """Return the process-wide Presidio anonymizer."""
    try:
        from presidio_anonymizer import AnonymizerEngine
    except ModuleNotFoundError as error:
        raise RuntimeError(_REDACTION_EXTRA_ERROR) from error
    return AnonymizerEngine()  # type: ignore[no-untyped-call]  # Presidio's constructor lacks annotations.


def _redact_pii(value: str, config: SensitiveDataRedactionConfig) -> str:
    if not config.pii_entities:
        return value
    results = _presidio_analyzer().analyze(
        text=value,
        language="en",
        entities=sorted(config.pii_entities),
        score_threshold=config.pii_score_threshold,
    )
    if not results:
        return value
    anonymizer = _presidio_anonymizer()
    from presidio_anonymizer.entities import OperatorConfig
    from presidio_anonymizer.entities import RecognizerResult as AnonymizerRecognizerResult

    anonymizer_results = [
        AnonymizerRecognizerResult(entity_type=result.entity_type, start=result.start, end=result.end, score=result.score) for result in results
    ]
    operators = {"DEFAULT": OperatorConfig("replace", {"new_value": config.replacement})}
    return anonymizer.anonymize(text=value, analyzer_results=anonymizer_results, operators=operators).text


def _redact_string(value: str, config: SensitiveDataRedactionConfig) -> str:
    if config.redact_bearer_tokens:
        value = _BEARER_PATTERN.sub(lambda _match: config.replacement, value)
    return _redact_pii(value, config)


def _is_sensitive_field_name(name: str, config: SensitiveDataRedactionConfig) -> bool:
    normalized = _normalized_field_name(name)
    if normalized in config.sensitive_field_names:
        return True
    return any(_normalized_field_name(part) in config.sensitive_field_names for part in re.split(r"[./]", name))


def redact_sensitive_data(value: object, config: SensitiveDataRedactionConfig) -> JsonValue:
    """Return a JSON value with configured sensitive fields and strings masked."""
    match value:
        case Enum():
            return redact_sensitive_data(value.value, config)
        case float() if not math.isfinite(value):
            return config.replacement
        case None | bool() | int() | float():
            return value
        case str():
            text = value
        case Mapping():
            result: dict[str, JsonValue] = {}
            for raw_key, item in value.items():
                key = str(raw_key)
                if _is_sensitive_field_name(key, config):
                    result[key] = config.replacement
                else:
                    result[key] = redact_sensitive_data(item, config)
            return result
        case Sequence() if not isinstance(value, str | bytes | bytearray):
            return [redact_sensitive_data(item, config) for item in value]
        case _:
            text = str(value)
    return _redact_string(text, config)


def create_sensitive_data_mask(config: SensitiveDataRedactionConfig) -> MaskFunction:
    """Create a Langfuse mask callback from an explicit redaction policy."""

    def mask(*, data: object, **_kwargs: object) -> JsonValue:
        return redact_sensitive_data(data, config)

    return mask


def _redact_otel_string(value: str, config: SensitiveDataRedactionConfig) -> str:
    if not value.lstrip().startswith(("{", "[")):
        return _redact_string(value, config)
    try:
        parsed = json.loads(value)
        redacted_json = redact_sensitive_data(parsed, config)
    except json.JSONDecodeError:
        return _redact_string(value, config)
    return json.dumps(redacted_json, ensure_ascii=False, separators=(",", ":")) if redacted_json != parsed else value


def _redact_otel_attribute(
    *,
    name: str,
    value: AttributeValue,
    config: SensitiveDataRedactionConfig,
) -> AttributeValue:
    if _is_sensitive_field_name(name, config):
        return config.replacement
    if isinstance(value, str):
        return _redact_otel_string(value, config)
    if isinstance(value, tuple | list) and all(isinstance(item, str) for item in value):
        return tuple(_redact_otel_string(item, config) for item in value)
    return value


def create_sensitive_data_otel_mask(config: SensitiveDataRedactionConfig) -> MaskOtelSpansFunction:
    """Create export-stage masking for Langfuse and third-party OpenTelemetry attributes."""

    def mask_otel_spans(*, params: MaskOtelSpansParams) -> MaskOtelSpansResult:
        patches = {}
        for identifier, span in params.spans.items():
            replacements: dict[str, AttributeValue] = {}
            for name, value in span.attributes.items():
                redacted = _redact_otel_attribute(name=name, value=value, config=config)
                if redacted != value:
                    replacements[name] = redacted
            if replacements:
                patches[identifier] = OtelSpanPatch(set_attributes=replacements)
        return MaskOtelSpansResult(span_patches=patches)

    return mask_otel_spans
