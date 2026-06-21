Working notes for AI agents in this repository. Keep this file operational and concise: quick-start, invariants, and high-impact traps.

## Before Answering Any Question

Read this file before answering questions about the codebase. Check the code and tests before describing behavior.

## Project Snapshot

`digintlab-langfuse-python` is a small typed Python package for Langfuse integration in agent runtimes.

Key modules:

- `src/digintlab_langfuse/client.py`: client construction, process instrumentation, callbacks, trace scopes, and shutdown.
- `src/digintlab_langfuse/spans.py`: node/generation/guardrail observations and HTTP auth-header redaction hook.
- `src/digintlab_langfuse/callbacks.py`: generic LangChain tool observation callback.
- `src/digintlab_langfuse/litellm.py`: active trace metadata forwarding for LiteLLM-compatible kwargs.
- `src/digintlab_langfuse/scores.py`: typed score helpers.
- `src/digintlab_langfuse/prompts.py` and `src/digintlab_langfuse/experiments.py`: thin typed helpers for Langfuse prompts and local experiment runs.

## Commands

```bash
task install
task format
task check
task test
uv build
```

## Critical Invariants

- Python 3.13+, `ruff` with strict rules, `mypy` strict mode.
- This package must not import application packages or application domain types.
- Keep the public API package-generic. Do not add app-specific names, defaults, schemas, or examples.
- Avoid `Any`; do not use loose `object`, `.get()`, `cast()`, or `__getattr__` to hide known structures. Prefer `TypedDict`, dataclasses, enums, `JsonValue`, and explicit unions.
- Prefer `StrEnum` for string-valued enums. Do not use `.value` unless a third-party API needs a plain string and the call site is clearer with it.
- Use match statements for closed-shape branching when it improves exhaustiveness or readability. Do not replace simple direct calls with noisy branching.
- Prefer generic types when they simplify ownership of typed values across helpers.
- Do not add shims, old-name aliases, compatibility layers, or wrappers that only forward arguments.
- No redaction policy should be hardcoded for all applications beyond explicit package defaults. Applications choose their own adapter defaults.

## Typing And Lint Discipline

- `mypy` runs in strict mode on both `src/` and `tests/`.
- Use `type: ignore[...]` only at real third-party typing boundaries, with the narrowest specific code.
- Use `# noqa:<rule>` only when unavoidable and target the exact rule.
- Prefer fixing the underlying type or lint issue over suppressing it.
- Keep `py.typed` present so downstream users receive type information.

## Maintenance Discipline

- Keep the package small and reusable; avoid application workflow code.
- Add new public helpers only when they represent reusable Langfuse behavior and are covered by package tests.
- Delete duplication before adding options or modes.
- A wrapper is allowed only when it enforces a boundary, adapts a required third-party signature, centralizes behavior used by multiple call sites, or gives one non-trivial operation a tested name.
- README examples and PR text should stay package-generic and safe for a public repository.

## When Editing

- Test-driven design is required for behavior changes: write or move the failing test first, then implementation, then validation.
- For pure documentation or metadata changes, run the relevant lightweight checks and explain why no behavior test was needed.
- Run `task format check test` before handing off changes that touch code or tests.
- Run `uv build` when packaging metadata, package data, or build configuration changes.
- Align with existing module boundaries; do not introduce new abstraction layers for one call site.

## Git Workflow

- Never push directly to `main`; work on a feature branch and open a PR.
- Squash related work before opening or updating the PR.
- Commit messages should name the subsystem and change in the imperative.
- Avoid generic messages like `fix`, `wip`, or `cleanup`.
