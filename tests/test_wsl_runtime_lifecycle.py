"""Exercise startup commands offline, without touching Docker or systemd."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(sys.platform == "win32", reason="Linux runtime shell contract")
@pytest.mark.parametrize("image_present", [True, False])
def test_start_reuses_installed_image_and_restores_enabled_compute(tmp_path, image_present):
    scripts = tmp_path / "infra/wsl/scripts"
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / "infra/wsl/scripts/start.sh", scripts / "start.sh")
    for name in ("migrate_runtime_to_linux_volumes.sh", "migrate_xiaomachi_data_volume.sh", "status.sh"):
        (scripts / name).write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (scripts.parent / ".env").write_text("QQ_PLATFORM=snowluma\nENABLE_GPU=1\n", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = f"""#!{sys.executable}
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['CALL_LOG'], 'a') as log:
    log.write(json.dumps([Path(sys.argv[0]).name, *args]) + '\\n')
if args == ['image', 'inspect', 'xiaomachi-bot:local']:
    sys.exit(0 if os.environ['IMAGE_PRESENT'] == '1' else 1)
"""
    for name in ("docker", "systemctl", "curl"):
        path = bin_dir / name
        path.write_text(stub, encoding="utf-8")
        path.chmod(0o755)
    log = tmp_path / "calls.jsonl"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"],
               CALL_LOG=str(log), IMAGE_PRESENT="1" if image_present else "0")
    result = subprocess.run(["bash", str(scripts / "start.sh")], env=env,
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    builds = [row for row in calls if row[-2:] == ["build", "xiaomachi"]]
    assert len(builds) == (0 if image_present else 1)
    assert ["docker", "volume", "inspect", "xiaomachi-llbot-data"] not in calls
    private = next(i for i, row in enumerate(calls) if row[-1] == "xiaomachi-private")
    compute = calls.index(["systemctl", "start", "xiaomachi-sandbox.service"])
    assert private < compute
