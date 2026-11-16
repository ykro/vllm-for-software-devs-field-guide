"""The Field Guide check tells the student what to clone when no unit is next to it."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_verifier_with_nothing_cloned_says_what_to_clone(tmp_path):
    repo = tmp_path / "vllm-for-software-devs-field-guide"
    (repo / "scripts").mkdir(parents=True)
    (repo / "benchmarks").mkdir()
    shutil.copy(ROOT / "scripts" / "verify_field_guide.py", repo / "scripts")
    run = subprocess.run([sys.executable, "scripts/verify_field_guide.py"], cwd=repo,
                         capture_output=True, text=True, timeout=60, check=False)
    assert run.returncode == 1
    assert "0 checked, 5 skipped." in run.stdout
    assert "SOME UNITS FAILED." not in run.stdout
    assert ("git clone https://github.com/ykro/first-vllm-server-with-gemma.git "
            "../first-vllm-server-with-gemma") in run.stdout
