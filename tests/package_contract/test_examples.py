"""Offline checks that keep the scripts in `examples/` importable and runnable."""

import importlib.util
import py_compile
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import httpx
import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from digintlab_langfuse import LangfuseClientConfig, SensitiveDataRedactionConfig

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"
EXAMPLE_PATHS = sorted(path for path in EXAMPLES_DIR.glob("*.py") if path.name != "__init__.py")
ENV_NAMES = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")


def _load(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"_example_{path.stem}", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline_config(http_client: httpx.Client, sensitive_data_redaction: SensitiveDataRedactionConfig | None = None) -> LangfuseClientConfig:
    return LangfuseClientConfig(
        public_key=f"pk-example-{uuid4()}",
        secret_key="test-secret",
        httpx_client=http_client,
        span_exporter=InMemorySpanExporter(),
        flush_at=100,
        flush_interval=60.0,
        sensitive_data_redaction=sensitive_data_redaction,
    )


def test_examples_exist() -> None:
    assert {path.stem for path in EXAMPLE_PATHS} == {"basic_tracing", "langchain_tool_callback", "redaction"}


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda path: path.stem)
def test_example_compiles_and_exposes_entry_points(path: Path, tmp_path: Path) -> None:
    py_compile.compile(str(path), cfile=str(tmp_path / f"{path.stem}.pyc"), doraise=True)
    module = _load(path)
    assert callable(module.run)
    assert callable(module.config_from_env)


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda path: path.stem)
def test_example_exits_with_clear_message_without_credentials(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    module = _load(path)
    with pytest.raises(SystemExit) as exit_info:
        module.config_from_env()
    message = str(exit_info.value)
    assert all(name in message for name in ENV_NAMES)


@pytest.mark.parametrize("path", EXAMPLE_PATHS, ids=lambda path: path.stem)
def test_example_reads_credentials_from_environment(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-env")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-env")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "https://langfuse.example.com")
    config = _load(path).config_from_env()
    assert isinstance(config, LangfuseClientConfig)
    assert config.public_key == "pk-env"
    assert config.base_url == "https://langfuse.example.com"


# The pinned Langfuse SDK calls archived backoff 2.2.1 from its score worker, which uses a deprecated asyncio helper on Python 3.14+.
@pytest.mark.filterwarnings("ignore:'asyncio.iscoroutinefunction' is deprecated:DeprecationWarning:backoff._decorator")
@pytest.mark.asyncio
async def test_basic_tracing_example_runs_offline() -> None:
    module = _load(EXAMPLES_DIR / "basic_tracing.py")
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(202, json={"message": "Accepted"}))) as http_client:
        await module.run(_offline_config(http_client))


@pytest.mark.asyncio
async def test_langchain_tool_callback_example_runs_offline() -> None:
    module = _load(EXAMPLES_DIR / "langchain_tool_callback.py")
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(202, json={"message": "Accepted"}))) as http_client:
        assert await module.run(_offline_config(http_client)) == 5


@pytest.mark.asyncio
async def test_redaction_example_runs_offline() -> None:
    module = _load(EXAMPLES_DIR / "redaction.py")
    with httpx.Client(transport=httpx.MockTransport(lambda _request: httpx.Response(202, json={"message": "Accepted"}))) as http_client:
        redacted = await module.run(_offline_config(http_client, module.redaction_policy()))
    assert redacted == {"api_key": "[REDACTED]", "note": "Contact [REDACTED]"}
