"""Repeat real Harness + synthetic QQ/space acceptance without live services."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basetemp", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-X", "utf8", "-m", "pytest", "tests/agent/test_harness_runtime.py",
               "tests/agent/test_mature_composition.py", "tests/agent/test_mature_boundaries.py",
               "tests/private_chat/test_agent_service.py", "-q", "-p", "no:cacheprovider",
               "--junitxml=" + str(output / "scenarios.xml")]
    if args.basetemp is not None:
        command.append("--basetemp=" + str(args.basetemp))
    completed = subprocess.run(command, cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True)
    (output / "pytest.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    cases = []
    if (output / "scenarios.xml").exists():
        for row in ET.parse(output / "scenarios.xml").iter("testcase"):
            cases.append({"name": row.attrib["name"], "seconds": float(row.attrib.get("time", 0)),
                          "status": "failed" if row.find("failure") is not None or row.find("error") is not None else
                                    "skipped" if row.find("skipped") is not None else "passed"})
    report = {"status": "PASS" if completed.returncode == 0 and cases and all(c["status"] == "passed" for c in cases) else "FAIL",
              "recorded_at": datetime.now(timezone.utc).isoformat(), "python": sys.version.split()[0],
              "framework": "deepseek-harness", "sdk": version("deepseek-harness-sdk"),
              "runtime": version("deepseek-harness-runtime-bin"), "command": command,
              "real_runtime": True, "platform_gateway": "synthetic QQ/space", "model": "local synthetic native SSE",
              "paid_model_requests": 0, "real_qq_writes": 0, "production_data_changes": 0,
              "scope": "Group, space and private execution mechanisms and business boundaries only; no claim about real-model autonomous intelligence",
              "cases": cases, "exit_code": completed.returncode}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "cases": len(cases), "report": str(output / "report.json")}))
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
