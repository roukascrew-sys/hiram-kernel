import sys, os, pathlib, ctypes, random, subprocess, time
from fractions import Fraction

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))

from hiram_compiler import compile_from_rules, Rule, CANONICAL_PRIORS, CANONICAL_RULES, build_reference_kb
from hiram_ac import eval_node

class Rational(ctypes.Structure):
    _fields_ = [("num", ctypes.c_int64), ("den", ctypes.c_int64)]

class Evidence(ctypes.Structure):
    _fields_ = [("observed_mask", ctypes.c_uint32), ("values_mask", ctypes.c_uint32)]

class HiramAuditReport(ctypes.Structure):
    _fields_ = [
        ("decision", ctypes.c_int),
        ("hazard_probability", Rational),
        ("hazard_threshold", Rational)
    ]

class HiramContext(ctypes.Structure):
    _fields_ = [
        ("memo", Rational * 81),
        ("execution_flags", ctypes.c_uint32)
    ]

def load_hiram_dll(dll_path=None):
    if dll_path is None:
        dll_path = REPO_ROOT / ("hiram_eval.dll" if sys.platform == "win32" else "libhiram_eval.so")
        cmd = ["gcc", "-O3", "-shared", "-fPIC", "-Iinclude", "-Iinclude/hiram", "src/hiram_eval.c", "-o", str(dll_path)]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    dll_path = pathlib.Path(dll_path)

    if sys.platform == "win32":
        import shutil
        gcc_bin = shutil.which("gcc")
        if gcc_bin:
            try: os.add_dll_directory(os.path.dirname(os.path.abspath(gcc_bin)))
            except Exception: pass
        try: os.add_dll_directory(str(dll_path.parent))
        except Exception: pass
        lib = ctypes.CDLL(str(dll_path), winmode=0)
    else:
        lib = ctypes.CDLL(str(dll_path))

    lib.hiram_evidence_init.argtypes = [ctypes.POINTER(Evidence)]
    lib.hiram_evidence_set.argtypes = [ctypes.POINTER(Evidence), ctypes.c_uint16, ctypes.c_bool]
    lib.hiram_context_init.argtypes = [ctypes.POINTER(HiramContext)]
    lib.hiram_eval_evidence.argtypes = [ctypes.POINTER(Evidence), ctypes.POINTER(HiramContext)]
    lib.hiram_eval_evidence.restype = Rational
    lib.hiram_audit_hazard.argtypes = [ctypes.POINTER(Evidence), Rational, ctypes.POINTER(HiramContext)]
    lib.hiram_audit_hazard.restype = HiramAuditReport
    return lib

def run_differential_suite(num_trials=1000, seed=20260908):
    random.seed(seed)
    lib = load_hiram_dll()
    kb = build_reference_kb()
    hazard_var_id = kb.var_to_id["stall_hazard"]

    print("=" * 80)
    print("HIRAM CROSS-LANGUAGE DIFFERENTIAL VERIFICATION (PYTHON vs C DLL)")
    print("=" * 80)
    print(f"Total Flight Variables: {len(kb.var_to_id)}")
    print(f"Monitored Hazard Var:   'stall_hazard' (ID {hazard_var_id})")
    print(f"Executing {num_trials} randomized differential evaluations (including hazard observations)...")
    print()

    passed = 0
    test_thresholds = [Rational(1, 20), Rational(1, 100), Rational(1, 200), Rational(1, 1)]

    ctx = HiramContext()
    lib.hiram_context_init(ctypes.byref(ctx))

    for trial in range(1, num_trials + 1):
        c_ev = Evidence()
        lib.hiram_evidence_init(ctypes.byref(c_ev))
        py_evidence = {}

        # Un-skip hazard variable: test all 12 flight variables
        for vid in range(len(kb.var_to_id)):
            obs_choice = random.choice([0, 1, 2])
            if obs_choice != 2:
                val = (obs_choice == 1)
                py_evidence[vid] = val
                lib.hiram_evidence_set(ctypes.byref(c_ev), vid, val)

        thresh = random.choice(test_thresholds)
        py_thresh = Fraction(thresh.num, thresh.den)

        # 1. Marginal Evidence Mass P(E)
        c_pe = lib.hiram_eval_evidence(ctypes.byref(c_ev), ctypes.byref(ctx))
        c_p_e = Fraction(c_pe.num, c_pe.den)
        py_p_e = eval_node(kb.circuit, kb.root_id, py_evidence)
        assert c_p_e == py_p_e, f"Trial {trial}: P(E) mismatch! Py={py_p_e} C={c_p_e}"

        # 2. Hazard Posterior Probability P(Hazard | E)
        rep = lib.hiram_audit_hazard(ctypes.byref(c_ev), thresh, ctypes.byref(ctx))
        c_post = Fraction(rep.hazard_probability.num, rep.hazard_probability.den)

        if py_p_e == 0:
            assert rep.decision == 2, f"Trial {trial}: Expected CONTRADICTION (2), got {rep.decision}"
        elif hazard_var_id in py_evidence:
            if not py_evidence[hazard_var_id]:
                py_post = Fraction(0, 1)
            else:
                py_post = Fraction(1, 1)
            assert c_post == py_post, f"Trial {trial}: Explicit hazard posterior mismatch! Py={py_post} C={c_post}"
            expected_decision = 1 if (py_post > py_thresh) else 0
            assert rep.decision == expected_decision, f"Trial {trial}: Decision mismatch on explicit hazard! Expected {expected_decision}, got {rep.decision}"
        else:
            py_p_hazard_and_e = eval_node(kb.circuit, kb.root_id, {**py_evidence, hazard_var_id: True})
            py_post = py_p_hazard_and_e / py_p_e
            assert c_post == py_post, f"Trial {trial}: Posterior mismatch! Py={py_post} C={c_post}"
            expected_decision = 1 if (py_post > py_thresh) else 0
            assert rep.decision == expected_decision, f"Trial {trial}: Decision mismatch! Expected {expected_decision}, got {rep.decision}"

        passed += 1
        if trial % 200 == 0:
            print(f"  -> Verified {trial:>5}/{num_trials} states... (All exact fraction and decision matches)")

    print()
    print("=" * 80)
    print("VERIFICATION OUTCOME: 100% BIT-EXACT MATCH ACROSS ALL METRICS")
    print("=" * 80)

if __name__ == "__main__":
    run_differential_suite(1000)