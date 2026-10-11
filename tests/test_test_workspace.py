"""Exercise pytest from a new checkout without any ignored data directory."""
import os
from pathlib import Path
import subprocess
import sys


def test_fresh_checkout_creates_its_own_test_workspace(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "fresh-checkout"
    project.mkdir()
    (project / "conftest.py").write_bytes((root / "tests/conftest.py").read_bytes())
    (project / "test_fresh.py").write_text(
        "def test_tmp_path(tmp_path):\n"
        "    assert tmp_path.is_dir()\n"
        "    assert 'data/test-work' in tmp_path.as_posix()\n", encoding="utf-8")
    environment = {**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"}
    process = subprocess.run([sys.executable, "-m", "pytest", "-q", "--rootdir", str(project), "-p", "no:cacheprovider"],
        cwd=project, env=environment, capture_output=True, text=True, timeout=45)
    assert process.returncode == 0, process.stdout + process.stderr
    assert (project / "data/test-work").is_dir()
