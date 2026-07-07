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
        [uv, "build", "--wheel", "--out-dir", str(dist_dir)],
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
