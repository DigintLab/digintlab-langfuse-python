"""Packaging contract tests."""

import shutil
import subprocess
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

    import_result = subprocess.run(  # noqa: S603
        [
            uv,
            "run",
            "--isolated",
            "--no-project",
            "--with",
            str(wheels[0]),
            "python",
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
