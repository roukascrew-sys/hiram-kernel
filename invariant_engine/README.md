# Invariant Engine — trusted verification kernel (Python)

Standard library only. No numpy, no solver bindings, no network calls.
Runs on Python 3.12.

## Why this lives in a C kernel's repository

`src/hiram_eval.c` is the embedded runtime: it evaluates a *pre-compiled*
arithmetic circuit on the target and answers one question fast and in bounded
stack. This directory is the layer above it, and it exists to answer a different
question — **can the thing that produced that circuit be trusted?**

Both implement the same architecture. An untrusted proposal engine (an LLM, a
learned search policy, an autoformaliser) sits outside the trusted computing
base and emits three kinds of output: candidate deduction traces, Bernoulli
priors for root variables, and rule dictionaries. Neither the C kernel nor this
one believes any of it. The C kernel re-evaluates the circuit itself; this one
re-checks the proposals that would produce such a circuit in the first place.

## Layout

| File | Role |
|---|---|
| `harness_generators.py` | Fixture generators. Mocks untrusted proposal output. |
| `invariant_engine.py` | The trusted kernel. Admits, verifies, and decides. |
| `test_harness_generators.py` | 570 tests over the generators. |
| `test_invariant_engine.py` | 243 tests over the kernel. |

The generators deliberately do not import the engine. A fixture generator that
imported the thing it tests could only ever produce fixtures that thing already
agrees with.

## What the kernel guarantees

- **Untrusted data cannot enter by construction.** `RuleBase` refuses direct
  construction; `admit_rules()` is the only door in, and it validates shape,
  vocabulary, and acyclicity first. Holding a `RuleBase` *is* the proof those
  checks ran.
- **A proposal can never raise a verdict.** Only a kernel-checked proof produces
  `VERIFIED`. A trace is a hint about where to look; if it is sound the kernel
  confirms it, and if it is not the kernel derives the goal itself or returns
  `UNKNOWN`. `check_goal()` re-verifies every `VERIFIED` before returning it.
- **Circular support is unprovable.** Every cited atom must be established by a
  *strictly earlier* step, base facts included. Forward chaining over definite
  clauses reaches a fixpoint on a cyclic rule set and will report the atom as
  "derived" — a fixpoint is not a proof, and the ordering requirement is the
  only thing separating them.
- **Absence of proof is not proof of absence.** `check_envelope()` returns
  `UNKNOWN`, not `SAFE`, when it cannot derive a violation. `SAFE` requires the
  caller to assert `closed_world=True`, and the assumption is then recorded in
  the report rather than left implicit.
- **No path enumeration.** Probabilistic queries use exact variable elimination.
  A 151-node chained diamond has 2**50 source-to-sink paths and is answered with
  a maximum factor width of 4.

## Running

```
python harness_generators.py          # generator self-test
python invariant_engine.py            # kernel self-test, 45 checks
python -m pytest .                    # 813 tests
```

Both modules exit non-zero on failure, so they work as smoke tests in a script.

## Caveats worth reading before trusting a number

- `NoisyOr.edge_activation` is an **engine assumption, not fixture data**. The
  generators supply one leak term per non-root node and nothing about edge
  strengths, so the CPD policy fills that gap. Marginals inherit its
  arbitrariness; read them as structural results, not calibrated probabilities.
- Every numeric threshold in `generate_aero_telemetry_fixture()` is
  **synthetic**. The speed nomenclature and limit-state names follow 14 CFR
  Part 1 / Part 25 terminology, but no value came from an AFM or type
  certificate, and each threshold carries `verified: False`.
- `circular_fraction` in `generate_horn_rules()` is an *injection* rate, not a
  guarantee about total cycle count or size — injected back-edges merge with the
  forward edges of the acyclic clauses into larger strongly connected
  components. Use `find_cycles_in_rules()` for what is actually there.
- `admit_rules()` cycle rejection is deliberately blunt: it drops any clause with
  both its consequent and an antecedent inside a cyclic component. Minimum
  feedback arc set is NP-hard, so this errs toward rejecting too much.
