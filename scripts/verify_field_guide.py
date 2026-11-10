"""Checks the vLLM Field Guide repos that are cloned next to this one.

For each unit it looks for the standard layout (five experiments, a judge, tests, three
diagrams, deploy scripts, README) and runs its judge against the built-in mock server, so
it needs no GPU. Units you have not cloned are reported as skipped.

    git clone https://github.com/ykro/first-vllm-server-with-gemma ../first-vllm-server-with-gemma
    python scripts/verify_field_guide.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PARENT = ROOT.parent
UNITS = [
    ("1", "first-vllm-server-with-gemma", "verify_first_server.py"),
    ("2", "tokens-memory-and-quantization-in-vllm", "verify_memory.py"),
    ("3", "many-users-one-gpu-with-vllm", "verify_many_users.py"),
    ("4", "structured-outputs-and-tools-with-vllm", "verify_structured.py"),
    ("5", "vllm-in-production-on-cloud-run", "verify_production.py"),
]


def layout_ok(repo: Path, judge: str) -> list[str]:
    missing = []
    if len(list((repo / "experiments").glob("0[1-5]_*.py"))) != 5:
        missing.append("experiments/01..05")
    for rel in (judge, "README.md", "Makefile", "gcp/deploy_vllm.sh"):
        if not (repo / rel).exists():
            missing.append(rel)
    if not list((repo / "tests").glob("test_*.py")):
        missing.append("tests/")
    return missing


def diagrams_ok(repo: Path) -> bool:
    return len(list((repo / "diagrams").glob("*.svg"))) >= 3


def main() -> int:
    python = sys.executable
    rows, ok_all = [], True
    for code, slug, judge in UNITS:
        repo = PARENT / slug
        if not repo.exists():
            rows.append((code, slug, "skipped (not cloned)", None))
            continue
        missing = layout_ok(repo, judge)
        run = subprocess.run([python, judge, "--mock"], check=False, cwd=repo, capture_output=True,
                             text=True, timeout=600)
        passed = run.returncode == 0 and "ALL CHECKS PASSED" in run.stdout
        status = "PASS" if passed and not missing else "FAIL"
        detail = "layout ok, mock judge passed" if status == "PASS" else (
            f"missing {', '.join(missing)}" if missing else "mock judge failed")
        if status == "PASS" and not diagrams_ok(repo):
            detail = "mock judge passed, diagrams pending"
        ok_all &= status == "PASS"
        rows.append((code, slug, detail, status))

    print("=" * 96)
    print("vLLM for Software Devs Field Guide · repo check (mock mode, no GPU)")
    print("=" * 96)
    for code, slug, detail, status in rows:
        print(f"{code:3} | {slug:42} | {detail:34} | {status or '-'}")
    print("-" * 96)
    checked = [r for r in rows if r[3]]
    print(f"{len(checked)} checked, {len(rows) - len(checked)} skipped.")
    (ROOT / "benchmarks" / "field_guide_check.json").write_text(json.dumps(
        [{"unit": c, "repo": s, "detail": d, "status": st} for c, s, d, st in rows], indent=2))
    print("ALL CHECKED UNITS PASSED." if ok_all and checked else "SOME UNITS FAILED.")
    return 0 if ok_all and checked else 1


if __name__ == "__main__":
    sys.exit(main())
