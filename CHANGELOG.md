# Changelog

All notable changes to this package are documented here.

## 0.3.0 - Unreleased

### Added

- Async flush and context-managed shutdown for telemetry owners.
- Opt-in sensitive-data redaction for Langfuse and OpenTelemetry payloads, with local Presidio PII detection.
- Safe shared ownership of optional HTTPX and botocore instrumentation.

### Changed

- Moved JSON helpers from `digintlab_langfuse.types` to `digintlab_langfuse.json_values`.
- Made process instrumentation opt-in and restricted each process to one Langfuse project owner.
- Strengthened configuration, lifecycle, and numeric-value validation.
- Split process instrumentation and SDK lifecycle ownership into private modules.
- Replaced type-specific score convenience methods with the typed `Score` and `ScoreTarget` API.
- Limited support to Python 3.13–3.14 and Langfuse 4.14.2.
