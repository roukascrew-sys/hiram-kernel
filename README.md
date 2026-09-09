# HIRAM Flight Safety Kernel

[![CI](https://github.com/roukascrew-sys/hiram-kernel/actions/workflows/ci.yml/badge.svg)](https://github.com/roukascrew-sys/hiram-kernel/actions/workflows/ci.yml)

HIRAM (High-Integrity Rational Arithmetic Monitor) is an embedded,
zero-allocation flight safety kernel designed for hard real-time avionics
control loops. It evaluates P(hazard | evidence) as an exact rational number and
vetoes actions from an untrusted flight controller when that probability exceeds
a rational threshold.

## System invariants

| Property | Value | Basis |
|---|---|---|
| Dynamic memory | 0 bytes | No malloc/free. MISRA-C:2012 Rule 21.3. |
| Recursion | none | Flat topological array sweep. MISRA-C:2012 Rule 17.2. |
| Arithmetic | exact rationals | `int64_t` with `__int128` intermediates; a 64-bit-only fallback is selectable and separately tested. |
| Max denominator observed | 53 bits | Measured max \|den\| = 6.0e15 across all 3<sup>12</sup> evidence configurations. |
| Evaluation context | 1,304 bytes | Static, in Cortex-M7 DTCM at `0x20000000`. Not on the stack. |
| Worst-case stack | 376 bytes | Deepest chain: `hiram_audit_hazard_vector` → `hiram_eval_evidence` → outlined body, ARM `-O3`. Enforced in CI. |
| Flash / RAM | 10,216 B / 1,304 B | `arm-none-eabi-size` on the linked firmware. |

Latency has only ever been measured on a desktop host (~24.5 µs mean under
Windows `QueryPerformanceCounter`). That is **not** a WCET figure: WCET requires
analysis on the target hardware. A DWT cycle-counting harness exists in
`src/target_main.c`, but no on-target measurement has been recorded.

## Repository map

| Path | Contents |
|---|---|
| `src/`, `include/` | The C kernel and its generated circuit tables. |
| `tools/` | Python model compiler, and the exporter that generates `hiram_circuit_data.h`. |
| `tests/` | Python/C differential suites and a standalone exhaustive sanitizer driver. |
| `harnesses/` | Whole-system audits against oracles that share no code with the kernel. |
| `invariant_engine/` | Python trusted verification kernel for untrusted proposals. 813 tests. |
| `docs/verification/` | Verification records. |

## Build and test

```bash
# Standalone self-test
gcc -O3 -Wall -Iinclude -Iinclude/hiram -DHIRAM_STANDALONE_TEST src/hiram_eval.c -o hiram_test
./hiram_test

# The build the project holds to
gcc -std=c99 -Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wshadow \
    -Wcast-align -Wcast-qual -Wswitch-enum -Wundef -Wformat=2 -pedantic-errors \
    -Iinclude -Iinclude/hiram -c src/hiram_eval.c -o hiram_eval.o

# Regenerate the circuit tables from the model (must be byte-identical)
PYTHONPATH=tools python tools/hiram_c_exporter.py

# Differential and whole-system verification
python tests/test_c_binding.py
python tests/test_vector_binding.py
python harnesses/independent_audit.py .
python harnesses/rigor_regression.py .
python harnesses/avionics_readiness.py .
python -m pytest invariant_engine -q
```

On Windows, Smart App Control intermittently blocks the executables the
harnesses compile (`WinError 4551`). It is non-deterministic; retry, or rely on
CI, which runs them on Linux. Undefined-behaviour instrumentation is also
Linux-only in practice, since MinGW ships no `libubsan`.

## Certification status

`FORMAL_CERTIFICATION_REPORT.md` is a self-issued document. It states a DO-178C
DAL-A baseline, which no certification authority has reviewed or granted, and
asserts MISRA-C:2012 compliance on the basis of three rules out of roughly 170.
Read it as a verification evidence summary, not a certificate.
