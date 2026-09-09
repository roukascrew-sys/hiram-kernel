# Audit harnesses

Whole-system checks that run the built kernel against independent oracles. They
are separate from `tests/` on purpose: `tests/` verifies the kernel against the
Python model it was generated from, and these verify it against reasoning that
shares no code with either side.

All three take the repository root as their only argument:

```
python harnesses/independent_audit.py .
python harnesses/rigor_regression.py .
python harnesses/avionics_readiness.py .
```

They build their own shared library into a gitignored `tmp_test_*` directory and
do not modify tracked sources.

| Harness | What it establishes |
|---|---|
| `independent_audit.py` | Rebuilds the evidence mass for all 3**12 configurations from first principles — a product of the twelve priors, gated by the four rule implications — and compares it against `hiram_eval_evidence` and `hiram_audit_hazard`. Shares no code with the compiler or the evaluator, so agreement is real evidence rather than two implementations of the same mistake. |
| `rigor_regression.py` | 1,000 random CNF models checked against an independent oracle (41,588 checks); threshold-comparison sweeps; malformed-input probes; and a regeneration check that the committed circuit tables reproduce byte-for-byte from the model. |
| `avionics_readiness.py` | Exhaustive sweep of all four API entry points across every evidence configuration (2,125,764 calls); a strict `-Werror -Wconversion -pedantic-errors` build; a sanitizer-instrumented build where the platform supports one; robustness probes; and the generated circuit's structural invariants. |

## Provenance

These lived outside the repository until 2026-09-09, in the author's Downloads
folder, which meant CI could not run them and nothing recorded which commit they
were written against. Two were stale on arrival and needed repair:

- `independent_audit.py` predated the DTCM context migration, so every call into
  the kernel was missing its `HiramContext` argument. It also expected
  `compile_circuit` to silently accept malformed input, which the compiler now
  correctly rejects; those probes record the rejection instead of aborting the
  run. Its exhaustive oracle had never completed against the current kernel.
- Its build directory was a system temp path. Windows Smart App Control blocks
  execution from `%LOCALAPPDATA%\Temp` with `WinError 4551`, so it now builds
  into a workspace-relative `tmp_test_audit/`.

## A check that was removed

The regeneration check used to compare `hiram_eval.c` and `hiram_eval.h` against
copies of themselves that the exporter made for exactly that purpose, and report
a pass. It could not fail. Those two files are hand-written, not generated, so
reproducibility is now claimed only for `hiram_circuit_data.h`, which is the one
artifact the exporter actually produces. Verified by corrupting
`CIRCUIT_ROOT_ID` in the committed header and confirming the harness exits 1.

## Known platform limitation

On Windows, Smart App Control intermittently blocks the freshly compiled
executables these harnesses produce, surfacing as `WinError 4551` from
`CreateProcess`. It is not deterministic — the same harness passes and fails
across consecutive runs — and it is an environment restriction rather than a
defect in the kernel or the harness. CI runs them on Linux, where it does not
occur. If a local run fails this way, retry it.

Undefined-behaviour instrumentation is also Linux-only in practice: MinGW ships
no `libubsan`, so `avionics_readiness.py` reports
`sanitizer: skipped: win32` and `ub_detection_active: false` on the author's
machine. It reports the status explicitly rather than claiming a pass, because
an earlier version reported `instrumented_build: pass` while running the entire
sweep with no detection at all.
