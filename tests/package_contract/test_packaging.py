"""Packaging contract tests."""

import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


def test_wheel_contains_py_typed(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    dist_dir = tmp_path / "dist"
    uv = shutil.which("uv")
    assert uv is not None

    result = subprocess.run(  # noqa: S603
        [uv, "build", "--wheel", "--no-build-isolation", "--out-dir", str(dist_dir)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    with ZipFile(wheels[0]) as wheel:
        assert "digintlab_langfuse/py.typed" in wheel.namelist()

    constraints = tmp_path / "constraints.txt"
    venv_dir = tmp_path / "venv"
    setup_commands = (
        # Pin the clean install to uv.lock so it reuses cached builds instead of resolving and compiling the newest releases.
        [uv, "export", "--locked", "--no-dev", "--no-emit-project", "--no-hashes", "--no-header", "--output-file", str(constraints)],
        [uv, "venv", "--python", sys.executable, str(venv_dir)],
        [uv, "pip", "install", "--python", str(venv_dir), "--constraint", str(constraints), str(wheels[0])],
    )
    for command in setup_commands:
        setup_result = subprocess.run(command, cwd=repo_root, text=True, capture_output=True, check=False)  # noqa: S603
        assert setup_result.returncode == 0, setup_result.stdout + setup_result.stderr

    import_result = subprocess.run(  # noqa: S603
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            (
                "import digintlab_langfuse\n"
                "from digintlab_langfuse import SensitiveDataRedactionConfig\n"
                "from digintlab_langfuse.json_values import to_json_value\n"
                "assert digintlab_langfuse.__all__ and to_json_value({'ok': True}) == {'ok': True}\n"
                "assert SensitiveDataRedactionConfig(pii_entities=frozenset()).pii_entities == frozenset()\n"
                "try:\n"
                "    SensitiveDataRedactionConfig()\n"
                "except RuntimeError as error:\n"
                "    assert 'digintlab-langfuse-python[redaction]' in str(error)\n"
                "else:\n"
                "    raise AssertionError('PII redaction unexpectedly available in the base installation')\n"
            ),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stdout + import_result.stderr
