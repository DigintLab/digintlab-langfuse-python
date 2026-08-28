import subprocess
import sys

import pytest

from digintlab_langfuse.redaction import (
    SensitiveDataRedactionConfig,
    _presidio_analyzer,
    redact_sensitive_data,
)


def test_presidio_redacts_common_rule_based_pii() -> None:
    config = SensitiveDataRedactionConfig(
        pii_entities=frozenset({"CREDIT_CARD", "EMAIL_ADDRESS", "IP_ADDRESS", "PHONE_NUMBER"}),
    )

    redacted = redact_sensitive_data(
        "Email person@example.com from 192.168.1.1, call +39 02 12345678, card 4111111111111111.",
        config,
    )

    assert redacted == "Email [REDACTED] from [REDACTED], call [REDACTED], card [REDACTED]."


def test_presidio_entity_selection_is_explicit() -> None:
    config = SensitiveDataRedactionConfig(pii_entities=frozenset({"EMAIL_ADDRESS"}))

    redacted = redact_sensitive_data("person@example.com from 192.168.1.1", config)

    assert redacted == "[REDACTED] from 192.168.1.1"


def test_empty_presidio_entity_selection_disables_pii_detection() -> None:
    config = SensitiveDataRedactionConfig(pii_entities=frozenset())

    redacted = redact_sensitive_data("person@example.com with Bearer secret-token", config)

    assert redacted == "person@example.com with [REDACTED]"


def test_presidio_score_threshold_is_configurable() -> None:
    text = "Call +39 02 12345678"

    assert redact_sensitive_data(text, SensitiveDataRedactionConfig(pii_score_threshold=0.4)) == "Call [REDACTED]"
    assert redact_sensitive_data(text, SensitiveDataRedactionConfig(pii_score_threshold=0.5)) == text


@pytest.mark.parametrize("score_threshold", [-0.01, 1.01])
def test_presidio_score_threshold_must_be_a_probability(score_threshold: float) -> None:
    with pytest.raises(ValueError, match="score threshold"):
        SensitiveDataRedactionConfig(pii_score_threshold=score_threshold)


def test_presidio_analyzer_does_not_load_an_nlp_model() -> None:
    from presidio_analyzer.nlp_engine import NoOpNlpEngine

    analyzer = _presidio_analyzer()

    assert analyzer is _presidio_analyzer()
    assert isinstance(analyzer.nlp_engine, NoOpNlpEngine)


def test_importing_package_does_not_eagerly_load_presidio() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import digintlab_langfuse; assert 'presidio_analyzer' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_presidio_email_validation_uses_the_bundled_suffix_snapshot() -> None:
    email_recognizers = [recognizer for recognizer in _presidio_analyzer().registry.recognizers if "EMAIL_ADDRESS" in recognizer.supported_entities]

    assert [type(recognizer).__name__ for recognizer in email_recognizers] == ["OfflineEmailRecognizer"]


def test_presidio_entity_names_are_validated_and_normalized() -> None:
    config = SensitiveDataRedactionConfig(pii_entities=frozenset({" email_address "}))

    assert config.pii_entities == frozenset({"EMAIL_ADDRESS"})

    with pytest.raises(ValueError, match="Unsupported Presidio entities: UNKNOWN"):
        SensitiveDataRedactionConfig(pii_entities=frozenset({"UNKNOWN"}))
