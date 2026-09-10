# CLAUDE.md — HIRAM Safety Kernel Guidelines for Claude Code

## Project Identity & High-Level Invariants
HIRAM is an embedded, exact-rational Bayesian safety monitor operating on ARM Cortex-M7 (STM32H723ZG) and host x86_64.
- Target constraints: Zero dynamic heap allocation (`malloc`/`free`), zero recursion, bounded call graphs.
- Memory: Evaluator scratchpad must reside inside `HiramContext` mapped to DTCM (`0x20000000`). Stack frame must remain <= 48 bytes.
- Mathematical precision: Exact integer rational arithmetic (`int64_t` pairs) using continued-fraction Euclidean descent (`rational_cmp`). No floating-point operations in the C kernel.

## Build, Test & Lint Commands
- Strict C99 Compilation:
  `gcc -std=c99 -Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wshadow -Wcast-align -Wcast-qual -Wswitch-enum -Wundef -Wformat=2 -pedantic-errors -Iinclude/hiram -c src/hiram_eval.c -o hiram_eval.o`
- Stack Usage Audit:
  `gcc -O3 -fstack-usage -DHIRAM_FORCE_64BIT_ACC -Iinclude/hiram -c src/hiram_eval.c -o stack_audit.o`
- Differential Python/C Test Suite:
  `python tests/test_c_binding.py`
- Target Cortex-M7 Cross-Compilation:
  `arm-none-eabi-gcc -mcpu=cortex-m7 -mthumb -O3 -Iinclude -Iinclude/hiram -c src/hiram_eval.c -o hiram_eval_m7.o`
- Link Bare-Metal ELF:
  `arm-none-eabi-gcc -T stm32h723zg.ld -nostartfiles -mcpu=cortex-m7 -mthumb -Wl,--gc-sections hiram_eval_m7.o target_main_m7.o -lgcc -o hiram_firmware.elf`

## Operating Rules for Claude Code
1. Never edit `include/hiram/hiram_circuit_data.h` manually. It must be generated exclusively via `tools/hiram_c_exporter.py`.
2. Do not introduce third-party C libraries, external packages, or dependencies. Use standard C99 and Python 3.12+ standard library (`fractions`, `ctypes`, `math`, `random`, `pathlib`, `json`).
3. Scratchpad & Output Paths:
   - On Windows 11, never write or execute compiled `.dll`/`.so` binaries inside `%LOCALAPPDATA%\Temp` or dot-prefixed folders (`.test_*`).
   - Use workspace-relative paths prefixed with `tmp_test_` to avoid Windows Smart App Control `WinError 4551`.
4. Stop & Ask Conditions:
   - Stop immediately if any test outputs a numerical discrepancy between Python and C.
   - Stop if `arm-none-eabi-size` shows non-zero `.bss` or `.data` allocations in the standalone evaluator.
   - Stop if `-fstack-usage` reports a frame size > 48 bytes for `hiram_eval_evidence`.