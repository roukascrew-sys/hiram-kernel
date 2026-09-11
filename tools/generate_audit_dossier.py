"""
tools.generate_audit_dossier - Automated Astra submission dossier generator.

Scaffolding for the audit paperwork step of the Task 3.1/3.2 workflow:
gathers the state an Astra sign-off needs a fresh look at on every
submission (current commit, the kernel's static DTCM footprint, the
pre-audit gatekeeper's verdict, the C/Python parity test's verdict --
including the integer-overflow regression -- the 4-way comparative
benchmark, and the wheel-slip ablation study) and writes them as a single
Markdown file, so there is nothing to hand-transcribe before a submission.

This script does not itself decide whether a commit is submittable -- that
is tools/pre_audit_verifier.py's job (also invoked here, as one of the
sections). A FAIL anywhere in the generated dossier means: fix the defect,
do not edit the dossier.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KERNEL_HEADER = REPO_ROOT / "include" / "hiram_kernel.h"
OUTPUT_PATH = REPO_ROOT / "build" / "audit_dossier.md"
BENCHMARK_JSON_PATH = REPO_ROOT / "build" / "benchmark_results.json"


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


def run_comparative_benchmark() -> tuple[bool, str, dict | None]:
    """Runs tools/run_comparative_benchmark.py (the 4-way controller
    benchmark + wheel-slip ablation study), which as a side effect writes
    build/benchmark_results.json / build/benchmark_results.md. Returns
    (success, captured stdout+stderr, parsed JSON results or None)."""
    res = subprocess.run(
        [sys.executable, "tools/run_comparative_benchmark.py"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    output = (res.stdout or "") + (res.stderr or "")
    benchmark_json = None
    if BENCHMARK_JSON_PATH.exists():
        try:
            benchmark_json = json.loads(BENCHMARK_JSON_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            benchmark_json = None
    return res.returncode == 0, output, benchmark_json


def _tail(text: str, n_lines: int = 25) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n_lines:])


def _render_ablation_section(benchmark_json: dict | None) -> str:
    if benchmark_json is None or "wheel_slip_ablation" not in benchmark_json:
        return "UNAVAILABLE -- build/benchmark_results.json missing or malformed; see section 5's raw output."

    ablation = benchmark_json["wheel_slip_ablation"]
    baseline = ablation["baseline"]
    ablated = ablation["ablated"]
    return f"""Scenario: **{ablation['scenario']}**

| | Baseline (WheelSlipObs observed) | Ablated (WheelSlipObs forced unobserved) |
|---|---|---|
| Collision | {baseline['collision']} | {ablated['collision']} |
| Contact time (s) | {baseline['contact_time']} | {ablated['contact_time']} |
| Contact velocity (m/s) | {baseline['contact_velocity']} | {ablated['contact_velocity']} |
| First brake/coast step | {baseline['first_brake_or_coast_step']} | {ablated['first_brake_or_coast_step']} |

**Finding:** {ablation['finding']}

This isolates whether wheel-slip inference is *causally* load-bearing for
collision avoidance in this scenario (as opposed to merely correlated with
a safe outcome), by re-running the identical episode with WheelSlipObs
forced to HIRAM_UNOBSERVED on every step and comparing outcomes directly --
see `tools/run_comparative_benchmark.py:run_wheel_slip_ablation`."""


def render_dossier(
    git_sha: str,
    workspace_footprint: str,
    verifier_passed: bool,
    verifier_output: str,
    parity_passed: bool,
    parity_output: str,
    benchmark_passed: bool,
    benchmark_output: str,
    benchmark_json: dict | None,
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    verifier_status = "PASS" if verifier_passed else "FAIL"
    parity_status = "PASS" if parity_passed else "FAIL"
    benchmark_status = "PASS" if benchmark_passed else "FAIL"
    overall_status = (
        "READY FOR SUBMISSION"
        if (verifier_passed and parity_passed and benchmark_passed)
        else "NOT READY -- DO NOT SUBMIT"
    )

    ablation_section = _render_ablation_section(benchmark_json)

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

**Scope of the zero-allocation guarantee:** this 48-byte static footprint,
and the "no malloc/free/calloc/realloc/alloca" guarantee `verify_zero_alloc`
enforces, describe **`src/hiram_kernel.c` as compiled for the target
(`build/libhiram_kernel.a` / the on-device build)** only. The Task 3.2
Python ctypes binding (`tools/hiram_c_binding.py`) that talks to
`build/libhiram_kernel.{{dll,so}}` from the host is **not** part of that
guarantee and does perform ordinary heap allocation -- CPython object
allocation for every `ctypes.Structure` instance, list, and dict the
binding constructs per call. That allocation is confined to the host-side
Python reference/verification harness; it is never linked into, or
present in, the on-target firmware image.

## 3. Pre-audit gatekeeper (`tools/pre_audit_verifier.py`): {verifier_status}

<details>
<summary>Full output</summary>

```
{_tail(verifier_output, 40)}
```

</details>

## 4. C/Python kernel parity (`tests/test_c_binding_parity.py`): {parity_status}

64/64 tri-state sensor evidence combinations, action agreement and
posterior max-diff <= 1e-12 required. Also includes the integer
overflow/underflow regression tests below.

<details>
<summary>Full output</summary>

```
{_tail(parity_output, 40)}
```

</details>

### 4a. Integer overflow/underflow vulnerability -- remediation summary

**Finding (Astra, commit 1891234):** `tools/hiram_c_binding.py`'s
`step_c_kernel()` passed its `lidar`/`tof`/`slip` arguments straight into
`ctypes.c_int32` struct fields with no bounds check. A `ctypes.c_int32`
assignment silently *wraps* an out-of-range Python int rather than raising
(`2**32 -> 0`, `-(2**32) -> 0`), so `step_c_kernel(4294967296, 0, 0)` was
indistinguishable from `step_c_kernel(0, 0, 0)`: it returned `status=0`
(`HIRAM_OK`) and action `ACCEL`, bypassing both this wrapper's and the C
kernel's own `HIRAM_ERR_INVALID_EVIDENCE` rejection entirely.

**Remediation:** `HiramCKernel.step()` (called by `step_c_kernel()`) now
validates each sensor argument in full Python-int precision via
`_validate_sensor_value()` *before* any value is narrowed into a
`ctypes.c_int32` field -- rejecting anything outside `{{-1, 0, 1, 2}}`
(including values only invalid because a `c_int32` would silently wrap
them, non-integers, `bool`, and `None`) by raising `ValueError`. Regression
coverage: `tests/test_c_binding_parity.py::test_integer_overflow_underflow_rejection`
and `::test_non_integer_and_none_rejected`, exercised as part of section 4
above.

## 5. Comparative benchmark & wheel-slip ablation (`tools/run_comparative_benchmark.py`): {benchmark_status}

4-way controller comparison (HIRAM Bayesian C kernel vs. Naive Threshold vs.
Conservative Kinematic vs. Always-Brake) across 5 operational regimes, plus
the wheel-slip ablation study (section 5a). Full reproducible results and
per-step traces: `build/benchmark_results.json`; human-readable table:
`build/benchmark_results.md`.

**Timing methodology note:** every latency figure in this section (and in
`build/benchmark_results.json`/`.md`) is an *observed host execution time* --
`time.perf_counter()` wall-clock around each controller's `step()` call, on
whatever development machine ran this script. These are **not** a hard
worst-case-execution-time (WCET) bound for the on-target Cortex-M7 kernel;
WCET analysis for the embedded target is out of scope for this host-side
comparative benchmark.

<details>
<summary>Full output</summary>

```
{_tail(benchmark_output, 60)}
```

</details>

### 5a. Wheel-slip ablation findings

{ablation_section}
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

    print("[dossier] Running tools/run_comparative_benchmark.py (4-way benchmark + wheel-slip ablation)...")
    benchmark_passed, benchmark_output, benchmark_json = run_comparative_benchmark()
    print(f"[dossier]   -> {'PASS' if benchmark_passed else 'FAIL'}")

    dossier = render_dossier(
        git_sha,
        workspace_footprint,
        verifier_passed,
        verifier_output,
        parity_passed,
        parity_output,
        benchmark_passed,
        benchmark_output,
        benchmark_json,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(dossier, encoding="utf-8")
    print(f"[dossier] Wrote {OUTPUT_PATH}")

    if not (verifier_passed and parity_passed and benchmark_passed):
        sys.exit(1)


if __name__ == "__main__":
    main()
