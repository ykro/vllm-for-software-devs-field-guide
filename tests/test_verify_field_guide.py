"""The Field Guide check tells the student what to clone when no unit is next to it."""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.skipif(sys.version_info < (3, 11), reason="before 3.11 the verifier asks for a newer Python")
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


def test_verifier_names_the_python_it_needs():
    spec = importlib.util.spec_from_file_location("verify", ROOT / "scripts" / "verify_field_guide.py")
    verify = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verify)
    assert verify.old_python_message((3, 11, 0)) is None
    assert "Python 3.9 is too old" in verify.old_python_message((3, 9, 6))
    assert "3.11 or newer" in verify.old_python_message((3, 9, 6))
