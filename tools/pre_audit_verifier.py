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


def run_c_build_and_tests() -> bool:
    """Rule 1: Full C pipeline gate -- `make clean && make all` (which itself
    includes verify_zero_alloc), an independent symbol scan of
    build/libhiram_kernel.a, and execution of the compiled C test binary.
    Unlike the previous soft-skip-if-missing behavior, a missing toolchain
    or a build/test failure is now a hard failure of this gate, not
    something to silently pass past."""
    print("[CHECK] Building C kernel via 'make clean && make all'...")
    try:
        res_clean = subprocess.run(["make", "clean"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    except FileNotFoundError:
        print("[FAIL] 'make' command not found in environment.")
        return False
    if res_clean.returncode != 0:
        print("[FAIL] 'make clean' failed:")
        print(res_clean.stdout)
        print(res_clean.stderr)
        return False

    res_build = subprocess.run(["make", "all"], cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(res_build.stdout)
    if res_build.returncode != 0:
        print("[FAIL] 'make all' failed (build and/or verify_zero_alloc):")
        print(res_build.stderr)
        return False
    print("[PASS] make all succeeded (build + verify_zero_alloc).")

    lib_path = REPO_ROOT / "build" / "libhiram_kernel.a"
    if not lib_path.exists():
        print(f"[FAIL] Expected static library not found after build: {lib_path}")
        return False

    print("[CHECK] Independently scanning C symbol table for prohibited heap allocators...")
    try:
        res_nm = subprocess.run(["nm", str(lib_path)], capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print("[FAIL] 'nm' utility not found in environment.")
        return False
    except subprocess.CalledProcessError as exc:
        print(f"[FAIL] 'nm' failed on {lib_path}: {exc.stderr}")
        return False
    prohibited = ["malloc", "free", "calloc", "realloc", "alloca"]
    for sym in prohibited:
        if sym in res_nm.stdout:
            print(f"[FAIL] Prohibited dynamic memory symbol detected: {sym}")
            return False
    print("[PASS] Zero dynamic memory allocation verified.")

    print("[CHECK] Executing C test binary...")
    candidates = [REPO_ROOT / "build" / "test_c_kernel.exe", REPO_ROOT / "build" / "test_c_kernel"]
    test_bin = next((c for c in candidates if c.exists()), None)
    if test_bin is None:
        print(f"[FAIL] C test binary not found at {candidates[0]} or {candidates[1]}")
        return False
    res_test = subprocess.run([str(test_bin)], cwd=str(REPO_ROOT), capture_output=True, text=True)
    print(res_test.stdout)
    if res_test.returncode != 0:
        print("[FAIL] C test binary reported failures:")
        print(res_test.stderr)
        return False
    print("[PASS] C test binary passed all checks.")
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

    if not run_c_build_and_tests():
        print("==================================================")
        print(">>> PRE-AUDIT GATEKEEPER STATUS: FAILED <<<")
        print("C build/test pipeline failed. Do NOT submit to auditor.")
        sys.exit(1)

    ok = True
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