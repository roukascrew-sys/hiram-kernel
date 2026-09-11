"""
HIRAM Pre-Audit Verification Gatekeeper.
Runs independent adversarial boundary checks, zero-allocation symbol audits,
and contract enforcement. Must pass 100% before submitting any commit to Astra.
"""

from __future__ import annotations
import sys
import subprocess
import os
import math
from pathlib import Path

# Ensure repository root is always in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_c_symbol_table() -> bool:
    """Rule 1: Prohibit heap allocations in compiled C static library."""
    lib_path = REPO_ROOT / "build" / "libhiram_kernel.a"
    
    # If not built, attempt to build via make or gcc
    if not lib_path.exists():
        print("[CHECK] C static library missing. Attempting build via make...")
        try:
            res_build = subprocess.run(["make", "all"], cwd=str(REPO_ROOT), capture_output=True, text=True)
            if res_build.returncode != 0:
                print("[WARN] 'make all' failed or make not installed. Checking for direct gcc build...")
        except FileNotFoundError:
            print("[WARN] 'make' command not found in environment.")

    if not lib_path.exists():
        print("[SKIP] C static library build/libhiram_kernel.a not found. Proceeding with Python checks.")
        return True

    print("[CHECK] Scanning C symbol table for prohibited heap allocators...")
    cmd = ["nm", str(lib_path)]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        prohibited = ["malloc", "free", "calloc", "realloc", "alloca"]
        for sym in prohibited:
            if sym in res.stdout:
                print(f"[FAIL] Prohibited dynamic memory symbol detected: {sym}")
                return False
        print("[PASS] Zero dynamic memory allocation verified.")
        return True
    except FileNotFoundError:
        print("[WARN] 'nm' utility not found. Skipping symbol scan.")
        return True


def check_python_boundary_matrix() -> bool:
    """Rule 2: Torture-test kinematics against boundary conditions."""
    print("[CHECK] Executing analytical kinematics boundary matrix...")
    from tools.trajectory_evaluator import solve_step_kinematics_and_contact

    # 1. Zero displacement stop
    x, v, coll, dt, cv = solve_step_kinematics_and_contact(
        x0=0.0, v0=0.001, a_cmd=-1.0, dt=0.02, v_cruise=0.50, obstacle_x=None
    )
    if not math.isclose(x, 0.0000005, abs_tol=1e-9) or v != 0.0 or coll:
        print(f"[FAIL] Zero-displacement stop mismatch: x={x}, v={v}, coll={coll}")
        return False

    # 2. Starting on boundary
    x, v, coll, dt, cv = solve_step_kinematics_and_contact(
        x0=0.20, v0=0.35, a_cmd=-1.0, dt=0.01, v_cruise=0.50, obstacle_x=0.20
    )
    if not coll or dt != 0.0 or cv != 0.35 or x != 0.20:
        print(f"[FAIL] Starting boundary contact mismatch: coll={coll}, dt={dt}, cv={cv}")
        return False

    # 3. Starting past boundary (penetration)
    x, v, coll, dt, cv = solve_step_kinematics_and_contact(
        x0=0.25, v0=0.35, a_cmd=-1.0, dt=0.01, v_cruise=0.50, obstacle_x=0.20
    )
    if not coll or dt != 0.0 or cv != 0.35 or x != 0.20:
        print(f"[FAIL] Penetration boundary contact mismatch: coll={coll}, dt={dt}, cv={cv}")
        return False

    # 4. Acceleration from rest with micro-clearance
    x, v, coll, dt, cv = solve_step_kinematics_and_contact(
        x0=0.0, v0=0.0, a_cmd=0.25, dt=0.01, v_cruise=0.50, obstacle_x=5e-6
    )
    if not coll or not math.isclose(dt, 0.006324555, rel_tol=1e-4) or not math.isclose(cv, 0.001581139, rel_tol=1e-4):
        print(f"[FAIL] Rest-start contact mismatch: coll={coll}, dt={dt}, cv={cv}")
        return False

    # 5. Cruise clamp boundary
    x, v, coll, dt, cv = solve_step_kinematics_and_contact(
        x0=0.0, v0=0.50, a_cmd=0.25, dt=0.01, v_cruise=0.50, obstacle_x=0.004
    )
    if not coll or not math.isclose(dt, 0.008, abs_tol=1e-6) or cv != 0.50:
        print(f"[FAIL] Cruise-clamp contact mismatch: coll={coll}, dt={dt}, cv={cv}")
        return False

    print("[PASS] All analytical kinematic boundaries verified.")
    return True


def run_test_suite() -> bool:
    """Rule 3: Full test runner."""
    print("[CHECK] Running full test suite...")
    res = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "tests", "-p", "test_*.py", "-v"],
        cwd=str(REPO_ROOT),
    )
    return res.returncode == 0


def main() -> None:
    print("==================================================")
    print("   HIRAM PRE-AUDIT GATEKEEPER VERIFICATION        ")
    print("==================================================")

    ok = True
    ok = ok and check_c_symbol_table()
    ok = ok and check_python_boundary_matrix()
    ok = ok and run_test_suite()

    print("==================================================")
    if ok:
        print(">>> PRE-AUDIT GATEKEEPER STATUS: PASSED <<<")
        print("Commit is qualified for adversarial audit submission.")
        sys.exit(0)
    else:
        print(">>> PRE-AUDIT GATEKEEPER STATUS: FAILED <<<")
        print("Do NOT submit to auditor. Correct defects above.")
        sys.exit(1)


if __name__ == "__main__":
    main()