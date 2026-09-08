import ctypes
from fractions import Fraction
import os
import random
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

# Resolve repository paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOOLS_DIR = os.path.join(REPO_ROOT, "tools")
SRC_FILE = os.path.join(REPO_ROOT, "src", "hiram_eval.c")
DLL_FILE = os.path.join(REPO_ROOT, "hiram_eval.dll")
INC_DIR = os.path.join(REPO_ROOT, "include")
INC_HIRAM_DIR = os.path.join(REPO_ROOT, "include", "hiram")

# Add tools/ to sys.path so modules import cleanly
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from hiram_ac import eval_node
from hiram_checker import Rule
from hiram_compiler import compile_from_rules

# -----------------------------------------------------------------------------
# 1. Locate GCC & Compile hiram_eval.c into hiram_eval.dll
# -----------------------------------------------------------------------------


def find_gcc() -> str:
    candidates = [
        shutil.which("gcc"),
        r"C:\Strawberry\c\bin\gcc.exe",
        os.path.expanduser(r"~\w64devkit\bin\gcc.exe"),
        r"C:\w64devkit\bin\gcc.exe",
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError(
        "GCC compiler executable was not found on PATH or standard directories."
    )


def compile_shared_library(src_path: str, dll_path: str):
    gcc = find_gcc()
    print(f"[*] Using C compiler: {gcc}")
    print(f"[*] Compiling {src_path} -> {dll_path} (-O3, -shared)...", end=" ")

    cmd = [
        gcc,
        "-O3",
        "-shared",
        "-Wall",
        f"-I{INC_DIR}",
        f"-I{INC_HIRAM_DIR}",
        "-Wl,--export-all-symbols",
        src_path,
        "-o",
        dll_path,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("\n[!] Compilation Failed:")
        print(res.stderr)
        sys.exit(1)
    print("Done.")


# -----------------------------------------------------------------------------
# 2. ctypes C-ABI Struct & Signature Mappings
# -----------------------------------------------------------------------------


class Rational(ctypes.Structure):
    _fields_ = [
        ("num", ctypes.c_int64),
        ("den", ctypes.c_int64),
    ]

    def to_fraction(self) -> Fraction:
        return Fraction(self.num, self.den)


class Evidence(ctypes.Structure):
    _fields_ = [
        ("observed_mask", ctypes.c_uint32),
        ("values_mask", ctypes.c_uint32),
    ]


class HiramAuditReport(ctypes.Structure):
    _fields_ = [
        ("decision", ctypes.c_int),
        ("hazard_probability", Rational),
        ("hazard_threshold", Rational),
    ]


HIRAM_APPROVED = 0
HIRAM_VETOED_SAFETY_VIOLATION = 1
HIRAM_REJECTED_CONTRADICTION = 2

DECISION_NAMES = {
    HIRAM_APPROVED: "APPROVED",
    HIRAM_VETOED_SAFETY_VIOLATION: "VETOED",
    HIRAM_REJECTED_CONTRADICTION: "CONTRADICTION",
}


def load_hiram_dll(dll_path: str) -> ctypes.CDLL:
    abs_dll = os.path.abspath(dll_path)
    if sys.platform == 'win32':
        for d in [os.path.dirname(abs_dll), r'C:\Users\rouka\w64devkit\bin']:
            if os.path.isdir(d):
                try: os.add_dll_directory(d)
                except Exception: pass
        import shutil
        gcc_bin = shutil.which('gcc')
        if gcc_bin:
            try: os.add_dll_directory(os.path.dirname(os.path.abspath(gcc_bin)))
            except Exception: pass
        try: lib = ctypes.CDLL(abs_dll, winmode=0)
        except TypeError: lib = ctypes.CDLL(abs_dll)
    else:
        lib = ctypes.CDLL(abs_dll)

    lib.hiram_evidence_init.argtypes = [ctypes.POINTER(Evidence)]
    lib.hiram_evidence_init.restype = None

    lib.hiram_evidence_set.argtypes = [
        ctypes.POINTER(Evidence),
        ctypes.c_uint16,
        ctypes.c_bool,
    ]
    lib.hiram_evidence_set.restype = None

    lib.hiram_eval_evidence.argtypes = [ctypes.POINTER(Evidence)]
    lib.hiram_eval_evidence.restype = Rational

    lib.hiram_audit_hazard.argtypes = [ctypes.POINTER(Evidence), Rational]
    lib.hiram_audit_hazard.restype = HiramAuditReport

    return lib


def build_reference_kb():
    rules = [
        Rule(
            "R_DESCENT_CLEAR",
            ["terrain_clear", "altitude_stable"],
            "authorize_descent",
        ),
        Rule(
            "R_STALL_RISK",
            ["low_airspeed", "high_angle_of_attack"],
            "stall_hazard",
        ),
        Rule(
            "R_ICING_RISK", ["subzero_temp", "high_humidity"], "icing_hazard"
        ),
        Rule(
            "R_OVERSPEED_RISK", ["high_mach", "flaps_extended"], "structural_hazard"
        ),
    ]

    priors = {
        "terrain_clear": Fraction(99, 100),
        "altitude_stable": Fraction(95, 100),
        "authorize_descent": Fraction(1, 2),
        "low_airspeed": Fraction(1, 20),
        "high_angle_of_attack": Fraction(1, 10),
        "stall_hazard": Fraction(1, 100),
        "subzero_temp": Fraction(1, 5),
        "high_humidity": Fraction(1, 3),
        "icing_hazard": Fraction(1, 50),
        "high_mach": Fraction(1, 50),
        "flaps_extended": Fraction(1, 4),
        "structural_hazard": Fraction(1, 500),
    }

    return compile_from_rules(rules, priors)


# -----------------------------------------------------------------------------
# 3. Differential Test Execution
# -----------------------------------------------------------------------------


def run_differential_suite(num_trials: int = 1000):
    compile_shared_library(SRC_FILE, DLL_FILE)
    c_lib = load_hiram_dll(DLL_FILE)
    kb = build_reference_kb()

    hazard_var_id = kb.var_to_id["stall_hazard"]
    all_vars = list(range(len(kb.var_to_id)))

    print("=" * 80)
    print("HIRAM CROSS-LANGUAGE DIFFERENTIAL VERIFICATION (PYTHON vs C DLL)")
    print("=" * 80)
    print(f"Total Flight Variables: {len(all_vars)}")
    print(f"Monitored Hazard Var:   'stall_hazard' (ID {hazard_var_id})")
    print(f"Executing {num_trials} randomized differential evaluations...\n")

    py_total_time = 0.0
    c_total_time = 0.0

    stats = {"approved": 0, "vetoed": 0, "contradictions": 0}
    thresh_frac = Fraction(5, 100)
    thresh_c = Rational(thresh_frac.numerator, thresh_frac.denominator)

    for trial in range(1, num_trials + 1):
        py_evidence: Dict[int, bool] = {}
        for v in all_vars:
            if v == hazard_var_id:
                continue
            if random.random() < 0.45:
                py_evidence[v] = random.choice([True, False])

        if trial == 1:
            py_evidence = {}
        elif trial == 2:
            py_evidence = {
                kb.var_to_id["low_airspeed"]: True,
                kb.var_to_id["high_angle_of_attack"]: True,
            }
        elif trial == 3:
            py_evidence = {
                kb.var_to_id["terrain_clear"]: True,
                kb.var_to_id["altitude_stable"]: True,
            }

        # Python Oracle
        t0 = time.perf_counter()
        py_p_e = eval_node(kb.circuit, kb.root_id, py_evidence)
        if py_p_e == Fraction(0, 1):
            py_decision = HIRAM_REJECTED_CONTRADICTION
            py_posterior = Fraction(0, 1)
        else:
            py_p_hazard_and_e = eval_node(
                kb.circuit, kb.root_id, {**py_evidence, hazard_var_id: True}
            )
            py_posterior = py_p_hazard_and_e / py_p_e
            py_decision = (
                HIRAM_VETOED_SAFETY_VIOLATION
                if py_posterior > thresh_frac
                else HIRAM_APPROVED
            )
        t1 = time.perf_counter()
        py_total_time += t1 - t0

        # C Runtime
        ev_c = Evidence()
        c_lib.hiram_evidence_init(ctypes.byref(ev_c))
        for v, val in py_evidence.items():
            c_lib.hiram_evidence_set(ctypes.byref(ev_c), v, val)

        t2 = time.perf_counter()
        c_p_e_struct = c_lib.hiram_eval_evidence(ctypes.byref(ev_c))
        c_report = c_lib.hiram_audit_hazard(ctypes.byref(ev_c), thresh_c)
        t3 = time.perf_counter()
        c_total_time += t3 - t2

        c_p_e = c_p_e_struct.to_fraction()
        c_posterior = c_report.hazard_probability.to_fraction()
        c_decision = c_report.decision

        # Invariant Assertions
        assert (
            c_p_e == py_p_e
        ), f"Trial {trial}: P(E) mismatch! Py={py_p_e} C={c_p_e}"
        assert (
            c_posterior == py_posterior
        ), f"Trial {trial}: Posterior mismatch! Py={py_posterior} C={c_posterior}"
        assert (
            c_decision == py_decision
        ), f"Trial {trial}: Decision mismatch! Py={py_decision} C={c_decision}"

        if c_decision == HIRAM_APPROVED:
            stats["approved"] += 1
        elif c_decision == HIRAM_VETOED_SAFETY_VIOLATION:
            stats["vetoed"] += 1
        elif c_decision == HIRAM_REJECTED_CONTRADICTION:
            stats["contradictions"] += 1

        if trial % 200 == 0:
            print(
                f"  -> Verified {trial:>4}/{num_trials} states... (All exact"
                " fraction matches)"
            )

    print("\n" + "=" * 80)
    print("VERIFICATION OUTCOME: 100% BIT-EXACT MATCH ACROSS ALL METRICS")
    print("=" * 80)
    print(f"Total Trials Executed:   {num_trials}")
    print(f"  - Approved (Safe):     {stats['approved']}")
    print(f"  - Vetoed (Hazard > 5%): {stats['vetoed']}")
    print(f"  - Contradictions:      {stats['contradictions']}")
    print("-" * 80)
    print(
        f"Mean Python Latency:     {(py_total_time / num_trials) * 1e6:8.2f} us"
        " / frame"
    )
    print(
        f"Mean C DLL Latency:      {(c_total_time / num_trials) * 1e6:8.2f} us"
        " / frame"
    )
    print(f"Cross-Language Speedup:  {(py_total_time / c_total_time):8.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    run_differential_suite(1000)
