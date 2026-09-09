"""Differential and contract verification for the multi-hazard vector API.

`hiram_audit_hazard_vector` shipped with no coverage in any suite or harness --
84 lines of safety-critical C, the largest stack consumer in the kernel, wholly
unverified. This file closes that gap and adds three structural checks that the
repository previously relied on an external harness to perform:

* ABI agreement between the ctypes bindings and the compiled C structs. The
  bindings hardcode the memo length; if the model grows, C writes past the end
  of a Python-owned buffer. That is heap corruption inside the verification
  tool, which is the worst place to have it.
* The generated circuit's topological and bounds invariants. The kernel reads
  ctx->memo[child_id] assuming every child precedes its parent. Since the
  context now persists across calls, a violation yields stale-but-plausible
  numbers instead of obvious garbage.
* Cross-API agreement: the vector API's stall posterior must equal the scalar
  API's on identical evidence. Two entry points to the same model that disagree
  is a defect regardless of which one is right.

Run:  python tests/test_vector_binding.py
"""

import ctypes
import itertools
import pathlib
import random
import re
import subprocess
import sys
from fractions import Fraction

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from hiram_ac import eval_node  # noqa: E402
from hiram_compiler import build_reference_kb  # noqa: E402
import test_c_binding  # noqa: E402
from test_c_binding import Evidence, Rational, load_hiram_dll  # noqa: E402

BUILD_DIR = REPO_ROOT / "tmp_test_build"
CIRCUIT_HEADER = REPO_ROOT / "include" / "hiram" / "hiram_circuit_data.h"

APPROVED, VETOED, CONTRADICTION, INVALID_INPUT = 0, 1, 2, 3
MONITORED = ("stall_hazard", "icing_hazard", "structural_hazard")


def header_constant(name: str) -> int:
    """Read a #define from the generated circuit header.

    Parsed rather than hardcoded on purpose: a binding that pins the node count
    by hand keeps compiling after a model change and then lets the kernel write
    past the end of the context it allocated.
    """
    text = CIRCUIT_HEADER.read_text(encoding="utf-8")
    match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"{name} not found in {CIRCUIT_HEADER}")
    return int(match.group(1))


NODE_COUNT = header_constant("CIRCUIT_NODE_COUNT")
VAR_COUNT = 12
MONITORED_HAZARDS_COUNT = 3


class HiramContext(ctypes.Structure):
    _fields_ = [("memo", Rational * NODE_COUNT), ("execution_flags", ctypes.c_uint32)]


class HiramVectorReport(ctypes.Structure):
    _fields_ = [
        ("overall_decision", ctypes.c_int),
        ("veto_mask", ctypes.c_uint32),
        ("hazard_posteriors", Rational * MONITORED_HAZARDS_COUNT),
        ("hazard_thresholds", Rational * MONITORED_HAZARDS_COUNT),
    ]


class HiramAuditReport(ctypes.Structure):
    _fields_ = [
        ("decision", ctypes.c_int),
        ("hazard_probability", Rational),
        ("hazard_threshold", Rational),
    ]


ThresholdArray = Rational * MONITORED_HAZARDS_COUNT


def bind_vector_api(lib):
    # Rebind every context-taking entry point to THIS module's HiramContext.
    # ctypes matches pointer types by class identity, and test_c_binding
    # declares its own structurally identical class.
    lib.hiram_context_init.argtypes = [ctypes.POINTER(HiramContext)]
    lib.hiram_eval_evidence.argtypes = [ctypes.POINTER(Evidence), ctypes.POINTER(HiramContext)]
    lib.hiram_eval_evidence.restype = Rational
    lib.hiram_audit_hazard_vector.argtypes = [
        ctypes.POINTER(Evidence),
        ctypes.POINTER(Rational),
        ctypes.POINTER(HiramContext),
    ]
    lib.hiram_audit_hazard_vector.restype = HiramVectorReport
    lib.hiram_audit_hazard.restype = HiramAuditReport
    lib.hiram_audit_hazard.argtypes = [
        ctypes.POINTER(Evidence),
        Rational,
        ctypes.POINTER(HiramContext),
    ]
    return lib


def frac(r) -> Fraction:
    return Fraction(r.num, r.den)


def evidence_of(mapping):
    ev = Evidence()
    ev.observed_mask = 0
    ev.values_mask = 0
    for var, val in mapping.items():
        ev.observed_mask |= 1 << var
        if val:
            ev.values_mask |= 1 << var
    return ev


# --------------------------------------------------------------------------- #
# 1. ABI agreement between the C structs and the ctypes bindings
# --------------------------------------------------------------------------- #

ABI_PROBE = r"""
#include <stdio.h>
#include <stddef.h>
#include "hiram_circuit_data.h"
int main(void) {
    printf("%zu %zu %zu %zu %zu %d\n",
           sizeof(HiramContext), offsetof(HiramContext, execution_flags),
           sizeof(HiramVectorReport), offsetof(HiramVectorReport, hazard_posteriors),
           offsetof(HiramVectorReport, hazard_thresholds), (int)CIRCUIT_NODE_COUNT);
    return 0;
}
"""


def check_abi_agreement(failures):
    BUILD_DIR.mkdir(exist_ok=True)
    src = BUILD_DIR / "tmp_test_abi_probe.c"
    exe = BUILD_DIR / ("tmp_test_abi_probe.exe" if sys.platform == "win32" else "tmp_test_abi_probe")
    src.write_text(ABI_PROBE, encoding="utf-8")
    subprocess.run(
        ["gcc", "-std=c99", "-I", "include", "-I", "include/hiram", str(src), "-o", str(exe)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    c_ctx, c_flags_off, c_vec, c_post_off, c_thr_off, c_nodes = (int(x) for x in out)

    expect = [
        ("sizeof(HiramContext)", c_ctx, ctypes.sizeof(HiramContext)),
        ("offsetof(HiramContext.execution_flags)", c_flags_off, HiramContext.execution_flags.offset),
        ("sizeof(HiramVectorReport)", c_vec, ctypes.sizeof(HiramVectorReport)),
        ("offsetof(HiramVectorReport.hazard_posteriors)", c_post_off, HiramVectorReport.hazard_posteriors.offset),
        ("offsetof(HiramVectorReport.hazard_thresholds)", c_thr_off, HiramVectorReport.hazard_thresholds.offset),
        ("CIRCUIT_NODE_COUNT", c_nodes, NODE_COUNT),
        # test_c_binding hardcodes its memo length. If a model change makes that
        # stale, the C kernel writes past the end of a Python-owned buffer and
        # corrupts the verification process itself.
        ("sizeof(test_c_binding.HiramContext)", c_ctx,
         ctypes.sizeof(test_c_binding.HiramContext)),
    ]
    print(f"  {'field':<48}{'C':>10}{'ctypes':>10}")
    for name, c_val, py_val in expect:
        ok = c_val == py_val
        print(f"  {name:<48}{c_val:>10}{py_val:>10}   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"ABI mismatch: {name} C={c_val} ctypes={py_val}")


# --------------------------------------------------------------------------- #
# 2. Generated-circuit structural invariants
# --------------------------------------------------------------------------- #

STRUCT_PROBE = r"""
#include <stdio.h>
#include "hiram_circuit_data.h"
int main(void) {
    unsigned back_edges = 0, oob_child = 0, oob_weight = 0, bad_weight = 0, bad_var = 0;
    for (unsigned i = 0; i < CIRCUIT_NODE_COUNT; i++) {
        const CircuitNode* n = &g_circuit_nodes[i];
        if (n->type == NODE_LITERAL && n->var_id >= CIRCUIT_VAR_COUNT) { bad_var++; }
        if (n->type == NODE_PROD || n->type == NODE_SUM) {
            if ((unsigned)n->child_offset + n->child_count > CHILDREN_ARRAY_SIZE) { oob_child++; }
            for (unsigned j = 0; j < n->child_count; j++) {
                if (g_circuit_children[n->child_offset + j] >= i) { back_edges++; }
            }
        }
        if (n->type == NODE_SUM) {
            if ((unsigned)n->weight_offset + n->child_count > WEIGHTS_ARRAY_SIZE) { oob_weight++; }
            for (unsigned j = 0; j < n->child_count; j++) {
                Rational w = g_circuit_weights[n->weight_offset + j];
                if (w.den <= 0 || w.num < 0 || w.num > w.den) { bad_weight++; }
            }
        }
    }
    printf("%u %u %u %u %u %d\n", back_edges, oob_child, oob_weight, bad_weight, bad_var,
           (int)(CIRCUIT_ROOT_ID < CIRCUIT_NODE_COUNT));
    return 0;
}
"""


def check_circuit_structure(failures):
    src = BUILD_DIR / "tmp_test_struct_probe.c"
    exe = BUILD_DIR / ("tmp_test_struct_probe.exe" if sys.platform == "win32" else "tmp_test_struct_probe")
    src.write_text(STRUCT_PROBE, encoding="utf-8")
    subprocess.run(
        ["gcc", "-std=c99", "-I", "include", "-I", "include/hiram", str(src), "-o", str(exe)],
        cwd=REPO_ROOT, check=True, capture_output=True,
    )
    out = subprocess.run([str(exe)], capture_output=True, text=True, check=True).stdout.split()
    back, oob_c, oob_w, bad_w, bad_v, root_ok = (int(x) for x in out)
    checks = [
        ("every child precedes its parent (topological order)", back == 0),
        ("child slices inside CHILDREN_ARRAY_SIZE", oob_c == 0),
        ("weight slices inside WEIGHTS_ARRAY_SIZE", oob_w == 0),
        ("every sum weight is a probability in [0,1]", bad_w == 0),
        ("every literal var_id < CIRCUIT_VAR_COUNT", bad_v == 0),
        ("root index inside the node array", root_ok == 1),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        if not ok:
            failures.append(f"circuit structure: {label}")


# --------------------------------------------------------------------------- #
# 3. Python oracle for the vector API
# --------------------------------------------------------------------------- #


def oracle_vector(kb, evidence, thresholds):
    """Independent expected result for hiram_audit_hazard_vector."""
    p_e = eval_node(kb.circuit, kb.root_id, evidence)
    if p_e == 0:
        return CONTRADICTION, 0, None

    posteriors, veto_mask = [], 0
    for index, name in enumerate(MONITORED):
        var = kb.var_to_id[name]
        if var in evidence:
            post = Fraction(1) if evidence[var] else Fraction(0)
        else:
            post = eval_node(kb.circuit, kb.root_id, {**evidence, var: True}) / p_e
        posteriors.append(post)
        if post > thresholds[index]:
            veto_mask |= 1 << index
    decision = VETOED if veto_mask else APPROVED
    return decision, veto_mask, posteriors


def check_vector_differential(lib, kb, failures, trials=2000, seed=20260908):
    rng = random.Random(seed)
    ctx = HiramContext()
    lib.hiram_context_init(ctypes.byref(ctx))
    choices = [Fraction(1, 20), Fraction(1, 100), Fraction(1, 500), Fraction(1, 1), Fraction(0, 1)]

    stats = {APPROVED: 0, VETOED: 0, CONTRADICTION: 0}
    for trial in range(1, trials + 1):
        evidence = {}
        for var in range(VAR_COUNT):
            state = rng.choice([0, 1, 2])
            if state:
                evidence[var] = state == 2
        thresholds = [rng.choice(choices) for _ in range(MONITORED_HAZARDS_COUNT)]

        arr = ThresholdArray(*[Rational(t.numerator, t.denominator) for t in thresholds])
        ev = evidence_of(evidence)
        rep = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(ctx))

        exp_decision, exp_mask, exp_post = oracle_vector(kb, evidence, thresholds)
        stats[exp_decision] = stats.get(exp_decision, 0) + 1

        if rep.overall_decision != exp_decision:
            failures.append(f"trial {trial}: decision C={rep.overall_decision} py={exp_decision}")
            break
        if exp_decision == CONTRADICTION:
            continue
        if rep.veto_mask != exp_mask:
            failures.append(f"trial {trial}: veto_mask C={rep.veto_mask:#b} py={exp_mask:#b}")
            break
        for h in range(MONITORED_HAZARDS_COUNT):
            got = frac(rep.hazard_posteriors[h])
            if got != exp_post[h]:
                failures.append(
                    f"trial {trial}: hazard {MONITORED[h]} posterior C={got} py={exp_post[h]}")
                break
            if frac(rep.hazard_thresholds[h]) != thresholds[h]:
                failures.append(f"trial {trial}: threshold {h} not echoed back")
                break
        else:
            continue
        break

    print(f"  {trials} randomized trials: approved={stats.get(APPROVED,0)} "
          f"vetoed={stats.get(VETOED,0)} contradiction={stats.get(CONTRADICTION,0)}")
    print(f"  [{'PASS' if not failures else 'FAIL'}] vector API matches the Python oracle")


# --------------------------------------------------------------------------- #
# 4. Cross-API consistency and the invalid-input contract
# --------------------------------------------------------------------------- #


def check_cross_api(lib, kb, failures, seed=4242, trials=400):
    """The vector API's stall posterior must equal the scalar API's."""
    rng = random.Random(seed)
    ctx = HiramContext()
    lib.hiram_context_init(ctypes.byref(ctx))
    thr = Fraction(1, 20)
    mismatches = 0

    for _ in range(trials):
        evidence = {}
        for var in range(VAR_COUNT):
            state = rng.choice([0, 1, 2])
            if state:
                evidence[var] = state == 2
        ev = evidence_of(evidence)
        arr = ThresholdArray(*[Rational(1, 20)] * MONITORED_HAZARDS_COUNT)

        scalar = lib.hiram_audit_hazard(ctypes.byref(ev), Rational(1, 20), ctypes.byref(ctx))
        vector = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(ctx))

        if scalar.decision == CONTRADICTION or vector.overall_decision == CONTRADICTION:
            if scalar.decision != vector.overall_decision:
                mismatches += 1
            continue
        if frac(scalar.hazard_probability) != frac(vector.hazard_posteriors[0]):
            mismatches += 1
            continue
        scalar_vetoes = scalar.decision == VETOED
        vector_vetoes_stall = bool(vector.veto_mask & 1)
        if scalar_vetoes != vector_vetoes_stall:
            mismatches += 1

    ok = mismatches == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] scalar and vector APIs agree on stall "
          f"across {trials} states ({mismatches} mismatches)")
    if not ok:
        failures.append(f"cross-API disagreement on {mismatches}/{trials} states")


def check_invalid_inputs(lib, failures):
    ctx = HiramContext()
    lib.hiram_context_init(ctypes.byref(ctx))
    ok_thr = ThresholdArray(*[Rational(1, 20)] * MONITORED_HAZARDS_COUNT)
    ev = evidence_of({})

    cases = [
        ("NULL context", ev, ok_thr, None),
        ("NULL thresholds", ev, None, ctypes.byref(ctx)),
        ("zero denominator", ev, ThresholdArray(Rational(1, 0), Rational(1, 20), Rational(1, 20)), ctypes.byref(ctx)),
        ("negative denominator", ev, ThresholdArray(Rational(-1, -20), Rational(1, 20), Rational(1, 20)), ctypes.byref(ctx)),
        ("threshold above one", ev, ThresholdArray(Rational(2, 1), Rational(1, 20), Rational(1, 20)), ctypes.byref(ctx)),
        ("negative threshold", ev, ThresholdArray(Rational(-1, 20), Rational(1, 20), Rational(1, 20)), ctypes.byref(ctx)),
    ]
    for label, evidence, thresholds, context in cases:
        rep = lib.hiram_audit_hazard_vector(ctypes.byref(evidence), thresholds, context)
        ok = rep.overall_decision == INVALID_INPUT
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} -> INVALID_INPUT (got {rep.overall_decision})")
        if not ok:
            failures.append(f"invalid input not rejected: {label}")

    # Non-canonical evidence: a value bit set without its observation bit.
    bad = Evidence()
    bad.observed_mask = 0
    bad.values_mask = 1 << 3
    rep = lib.hiram_audit_hazard_vector(ctypes.byref(bad), ok_thr, ctypes.byref(ctx))
    ok = rep.overall_decision == INVALID_INPUT
    print(f"  [{'PASS' if ok else 'FAIL'}] value bit without observation bit -> INVALID_INPUT")
    if not ok:
        failures.append("non-canonical evidence accepted by the vector API")

    # Observation of a variable outside the compiled model.
    oob = Evidence()
    oob.observed_mask = 1 << 20
    oob.values_mask = 0
    rep = lib.hiram_audit_hazard_vector(ctypes.byref(oob), ok_thr, ctypes.byref(ctx))
    ok = rep.overall_decision == INVALID_INPUT
    print(f"  [{'PASS' if ok else 'FAIL'}] observation outside the model -> INVALID_INPUT")
    if not ok:
        failures.append("out-of-model observation accepted by the vector API")


def check_context_reuse(lib, kb, failures):
    """A reused context must not leak state between evaluations.

    The context is now long-lived, so a stale memo entry would survive into the
    next call. A fresh context and a heavily reused one must agree exactly.
    """
    fresh_results, reused_results = [], []
    reused = HiramContext()
    lib.hiram_context_init(ctypes.byref(reused))
    arr = ThresholdArray(*[Rational(1, 20)] * MONITORED_HAZARDS_COUNT)

    rng = random.Random(99)
    states = []
    for _ in range(200):
        evidence = {v: rng.choice([True, False]) for v in range(VAR_COUNT) if rng.random() < 0.5}
        states.append(evidence)

    for evidence in states:
        ev = evidence_of(evidence)
        r = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(reused))
        reused_results.append((r.overall_decision, r.veto_mask,
                               tuple(frac(r.hazard_posteriors[i]) for i in range(3))))

    for evidence in states:
        clean = HiramContext()
        lib.hiram_context_init(ctypes.byref(clean))
        ev = evidence_of(evidence)
        r = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(clean))
        fresh_results.append((r.overall_decision, r.veto_mask,
                              tuple(frac(r.hazard_posteriors[i]) for i in range(3))))

    ok = fresh_results == reused_results
    print(f"  [{'PASS' if ok else 'FAIL'}] a reused context yields results identical to fresh ones")
    if not ok:
        failures.append("context reuse changes results (stale memo leakage)")


def check_uninitialised_context(lib, kb, failures):
    """The kernel must not depend on hiram_context_init having been called.

    Every memo slot is written before it is read only if the circuit is in
    topological order. Feeding a deliberately poisoned context proves the
    kernel overwrites rather than reads stale values.
    """
    poisoned = HiramContext()
    for i in range(NODE_COUNT):
        poisoned.memo[i] = Rational(999983, 7)
    poisoned.execution_flags = 0xDEADBEEF

    clean = HiramContext()
    lib.hiram_context_init(ctypes.byref(clean))
    arr = ThresholdArray(*[Rational(1, 20)] * MONITORED_HAZARDS_COUNT)

    rng = random.Random(7)
    mismatches = 0
    for _ in range(200):
        evidence = {v: rng.choice([True, False]) for v in range(VAR_COUNT) if rng.random() < 0.5}
        ev = evidence_of(evidence)
        a = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(poisoned))
        b = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(clean))
        if (a.overall_decision, a.veto_mask) != (b.overall_decision, b.veto_mask):
            mismatches += 1
        elif any(frac(a.hazard_posteriors[i]) != frac(b.hazard_posteriors[i]) for i in range(3)):
            mismatches += 1

    ok = mismatches == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] a poisoned context is fully overwritten "
          f"({mismatches} divergences)")
    if not ok:
        failures.append("kernel reads stale memo entries from an uninitialised context")


def check_exhaustive_hazard_subspace(lib, kb, failures):
    """Exhaustive sweep over the six variables that drive the three hazards.

    3**6 = 729 configurations of the six variables that feed the three hazard
    rules (low_airspeed, high_angle_of_attack, freezing_temp, high_humidity,
    high_g_load, airframe_stress), each with the three hazard variables left
    unobserved, checked against the Python oracle.
    """
    ctx = HiramContext()
    lib.hiram_context_init(ctypes.byref(ctx))
    drivers = [kb.var_to_id[n] for n in
               ("low_airspeed", "high_angle_of_attack", "freezing_temp",
                "high_humidity", "high_g_load", "airframe_stress")]
    thresholds = [Fraction(1, 20), Fraction(1, 50), Fraction(1, 500)]
    arr = ThresholdArray(*[Rational(t.numerator, t.denominator) for t in thresholds])

    checked = mismatches = 0
    for combo in itertools.product(range(3), repeat=len(drivers)):
        evidence = {v: (s == 2) for v, s in zip(drivers, combo) if s}
        ev = evidence_of(evidence)
        rep = lib.hiram_audit_hazard_vector(ctypes.byref(ev), arr, ctypes.byref(ctx))
        exp_decision, exp_mask, exp_post = oracle_vector(kb, evidence, thresholds)
        checked += 1
        if rep.overall_decision != exp_decision:
            mismatches += 1
        elif exp_decision != CONTRADICTION:
            if rep.veto_mask != exp_mask or any(
                frac(rep.hazard_posteriors[h]) != exp_post[h] for h in range(3)
            ):
                mismatches += 1

    ok = mismatches == 0
    print(f"  [{'PASS' if ok else 'FAIL'}] exhaustive {checked} hazard-driver configurations "
          f"({mismatches} mismatches)")
    if not ok:
        failures.append(f"exhaustive hazard sweep: {mismatches}/{checked} mismatches")


def main() -> int:
    failures = []
    lib = bind_vector_api(load_hiram_dll())
    kb = build_reference_kb()

    print("=" * 78)
    print("HIRAM MULTI-HAZARD VECTOR API VERIFICATION")
    print("=" * 78)
    print(f"Circuit nodes: {NODE_COUNT} (read from generated header)")
    print(f"Monitored hazards: {', '.join(MONITORED)}")

    print("\n-- 1. ABI agreement (C structs vs ctypes bindings) --")
    check_abi_agreement(failures)

    print("\n-- 2. Generated circuit structural invariants --")
    check_circuit_structure(failures)

    print("\n-- 3. Vector API differential vs Python oracle --")
    check_vector_differential(lib, kb, failures)

    print("\n-- 4. Exhaustive hazard-driver subspace --")
    check_exhaustive_hazard_subspace(lib, kb, failures)

    print("\n-- 5. Cross-API consistency (scalar vs vector) --")
    check_cross_api(lib, kb, failures)

    print("\n-- 6. Invalid-input contract --")
    check_invalid_inputs(lib, failures)

    print("\n-- 7. Context lifecycle --")
    check_context_reuse(lib, kb, failures)
    check_uninitialised_context(lib, kb, failures)

    print("\n" + "=" * 78)
    if failures:
        print(f"FAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL VECTOR API CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
