"""
tools.generate_audit_dossier - Automated Astra submission dossier generator.

Scaffolding for the audit paperwork step of the Task 3.1/3.2 workflow:
gathers the four pieces of state an Astra sign-off needs a fresh look at on
every submission (current commit, the kernel's static DTCM footprint, the
pre-audit gatekeeper's verdict, and the C/Python parity test's verdict) and
writes them as a single Markdown file, so there is nothing to hand-transcribe
before a submission.

This script does not itself decide whether a commit is submittable -- that
is tools/pre_audit_verifier.py's job (also invoked here, as one of the four
sections). A FAIL anywhere in the generated dossier means: fix the defect,
do not edit the dossier.
"""

from __future__ import annotations

import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_HEADER = REPO_ROOT / "include" / "hiram_kernel.h"
OUTPUT_PATH = REPO_ROOT / "build" / "audit_dossier.md"


def get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        return f"UNAVAILABLE ({exc})"
    return res.stdout.strip()


def get_workspace_memory_footprint() -> str:
    """Reads the exact and budget-ceiling compile-time size assertions for
    hiram_workspace_t straight out of include/hiram_kernel.h, rather than
    hardcoding them here -- so the dossier can never silently go stale if
    the workspace struct grows."""
    if not KERNEL_HEADER.exists():
        return f"UNAVAILABLE (missing {KERNEL_HEADER})"

    text = KERNEL_HEADER.read_text(encoding="utf-8")
    exact_match = re.search(r"sizeof\(hiram_workspace_t\)\s*==\s*(\d+)", text)
    budget_match = re.search(r"sizeof\(hiram_workspace_t\)\s*<=\s*(\d+)", text)

    if not exact_match or not budget_match:
        return "UNAVAILABLE (compile-time size assertions not found in header)"

    exact_bytes = int(exact_match.group(1))
    budget_bytes = int(budget_match.group(1))
    return f"{exact_bytes} bytes (hard ceiling: {budget_bytes} bytes / {budget_bytes / 1024:.1f} KB)"


def run_pre_audit_verifier() -> tuple[bool, str]:
    res = subprocess.run(
        [sys.executable, "tools/pre_audit_verifier.py"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    output = (res.stdout or "") + (res.stderr or "")
    return res.returncode == 0, output


def run_c_binding_parity_test() -> tuple[bool, str]:
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "tests.test_c_binding_parity", "-v"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    output = (res.stdout or "") + (res.stderr or "")
    return res.returncode == 0, output


def _tail(text: str, n_lines: int = 25) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n_lines:])


def render_dossier(
    git_sha: str,
    workspace_footprint: str,
    verifier_passed: bool,
    verifier_output: str,
    parity_passed: bool,
    parity_output: str,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    verifier_status = "PASS" if verifier_passed else "FAIL"
    parity_status = "PASS" if parity_passed else "FAIL"
    overall_status = "READY FOR SUBMISSION" if (verifier_passed and parity_passed) else "NOT READY -- DO NOT SUBMIT"

    return f"""# HIRAM Safety Kernel -- Astra Audit Dossier

Generated: {generated_at}
Generator: `tools/generate_audit_dossier.py`

## Overall status: {overall_status}

## 1. Commit under review

```
{git_sha}
```

## 2. Kernel workspace memory footprint (`hiram_workspace_t`)

{workspace_footprint}

Target: ARM Cortex-M7 DTCM, hard ceiling < 64 KB (this kernel's own
compile-time assertion enforces a tighter < 40 KB budget -- see
`include/hiram_kernel.h`).

## 3. Pre-audit gatekeeper (`tools/pre_audit_verifier.py`): {verifier_status}

<details>
<summary>Full output</summary>

```
{_tail(verifier_output, 40)}
```

</details>

## 4. C/Python kernel parity (`tests/test_c_binding_parity.py`): {parity_status}

64/64 tri-state sensor evidence combinations, action agreement and
posterior max-diff <= 1e-12 required.

<details>
<summary>Full output</summary>

```
{_tail(parity_output, 40)}
```

</details>
"""


def main() -> None:
    print("[dossier] Collecting git SHA...")
    git_sha = get_git_sha()

    print("[dossier] Reading workspace memory footprint from include/hiram_kernel.h...")
    workspace_footprint = get_workspace_memory_footprint()

    print("[dossier] Running tools/pre_audit_verifier.py (this rebuilds and re-tests everything)...")
    verifier_passed, verifier_output = run_pre_audit_verifier()
    print(f"[dossier]   -> {'PASS' if verifier_passed else 'FAIL'}")

    print("[dossier] Running tests/test_c_binding_parity.py...")
    parity_passed, parity_output = run_c_binding_parity_test()
    print(f"[dossier]   -> {'PASS' if parity_passed else 'FAIL'}")

    dossier = render_dossier(
        git_sha, workspace_footprint, verifier_passed, verifier_output, parity_passed, parity_output
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(dossier, encoding="utf-8")
    print(f"[dossier] Wrote {OUTPUT_PATH}")

    if not (verifier_passed and parity_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
