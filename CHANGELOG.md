# Changelog

All notable changes to this package are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.5.1] - 2026-11-18

### Changed

- Updated locked dependencies: filelock from 3.32.4 to 4.2.1, msgpack from 1.2.2 to 1.2.3, setuptools from 84.0.0 to 84.1.0, and urllib3 from 2.7.0 to 2.8.0.

## [0.5.0] - 2026-11-15

### Added

- Tag-triggered release workflow that checks the tag against the package version, runs the checks, builds and attests the wheel and sdist, and publishes the GitHub release with notes from this changelog.

### Changed

- Relaxed the Langfuse requirement from an exact pin to the patch releases of the tested minor version (`langfuse>=5.2.0,<5.3`).
- Shortened the README to a module overview and a limitations section.

## [0.4.0] - 2026-11-15

### Added

- Added support for Python 3.15.
- Declared each public module's exports with `__all__` and documented the API stability and versioning policy.

### Changed

- Licensed the package under LGPL-3.0-or-later.
- Pinned Langfuse 5.2.0. Clients created through `init_langfuse_telemetry` pick up the SDK's LangChain `CallbackHandler` fixes: root-run propagated attributes now reach child runs under `ainvoke`, Langfuse prompts are deregistered when a chain errors, and `message.name` is kept in generation input.
- Documented installation from tagged GitHub releases; the package is not published on PyPI.
- Limited the source distribution to the package, tests, docs, and license files.

### Fixed

- Sensitive-data redaction now masks credential fields inside JSON documents carried as strings, such as LangChain tool `input_str` and string tool outputs, in both the Langfuse mask and the OpenTelemetry mask.

## 0.3.0 - 2026-08-28

### Added

- Async flush and context-managed shutdown for telemetry owners.
- Opt-in sensitive-data redaction for Langfuse and OpenTelemetry payloads, with local Presidio PII detection.
- Safe shared ownership of optional HTTPX and botocore instrumentation.

### Changed

- Moved JSON helpers from `digintlab_langfuse.types` to `digintlab_langfuse.json_values`.
- Made process instrumentation opt-in and restricted each process to one Langfuse project owner.
- Reject externally owned HTTPX instrumentation when explicit authorization-header redaction cannot be guaranteed.
- Strengthened configuration, lifecycle, and numeric-value validation.
- Kept trace propagation fail-open and bounded failed-initialization shutdown with the configured deadline.
- Split process instrumentation and SDK lifecycle ownership into private modules.
- Replaced type-specific score convenience methods with the typed `Score` and `ScoreTarget` API.
- Limited support to Python 3.13–3.14 and Langfuse 4.14.2.

[Unreleased]: https://github.com/DigintLab/digintlab-langfuse-python/compare/v0.5.1...HEAD
[0.5.1]: https://github.com/DigintLab/digintlab-langfuse-python/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/DigintLab/digintlab-langfuse-python/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/DigintLab/digintlab-langfuse-python/releases/tag/v0.4.0
