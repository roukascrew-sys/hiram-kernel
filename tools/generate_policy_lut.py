"""
tools/generate_policy_lut.py

Evaluates the sealed Bayesian Inference Engine / C Kernel across all 64 discrete
tri-state evidence combinations:
    LidarObs     in {-1, 0, 1, 2}
    TofObs       in {-1, 0, 1, 2}
    WheelSlipObs in {-1, 0, 1, 2}

Compiles the optimal Bayesian action decisions into:
  1. include/hiram_policy_lut.h  (ABI, prototypes, section-placement macros)
  2. src/hiram_policy_lut.c      (Table + evaluator + CRC32 integrity check)
  3. build/policy_lut_data.json  (Bit-exact mapping verification table)

The header no longer carries the table or the evaluator body: Task 3.3 places
both in explicit STM32H723 link sections (.dtcm_lut / .itcm_text) so each has
exactly one resident copy in on-chip TCM, which a `static inline` in a header
cannot guarantee. HIRAM_POLICY_LUT_CANONICAL_CRC32 is computed here, over the
same 64 bytes written below, so the on-target startup check
(hiram_policy_lut_verify_integrity) always has a digest that matches what was
actually compiled -- never a hand-typed constant that can drift from the table.
"""

from __future__ import annotations

import json
import pathlib
import sys
import zlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hiram_c_binding import HIRAM_ACTION_NAMES, step_c_kernel

VALID_DOMAIN = (-1, 0, 1, 2)


def generate_lut():
    lut_entries = []
    c_table = []

    print(f"Generating 64-state Bayesian policy lookup table...")

    for lidar in VALID_DOMAIN:
        for tof in VALID_DOMAIN:
            for slip in VALID_DOMAIN:
                # Key indexing formula: (val + 1) -> maps {-1, 0, 1, 2} to {0, 1, 2, 3}
                idx = ((lidar + 1) << 4) | ((tof + 1) << 2) | (slip + 1)

                step_res = step_c_kernel(lidar=lidar, tof=tof, slip=slip)
                action_idx = step_res["selected_action_index"]
                action_name = step_res["selected_action"]

                lut_entries.append({
                    "lut_index": idx,
                    "lidar": lidar,
                    "tof": tof,
                    "slip": slip,
                    "action_index": action_idx,
                    "action_name": action_name,
                    "expected_losses": step_res["expected_losses"],
                })

    # Sort strictly by LUT index 0..63
    lut_entries.sort(key=lambda x: x["lut_index"])

    table_bytes = bytes(e["action_index"] for e in lut_entries)
    assert len(table_bytes) == 64, f"expected 64 table entries, got {len(table_bytes)}"
    canonical_crc32 = zlib.crc32(table_bytes) & 0xFFFFFFFF

    header_path = REPO_ROOT / "include" / "hiram_policy_lut.h"
    source_path = REPO_ROOT / "src" / "hiram_policy_lut.c"
    json_path = REPO_ROOT / "build" / "policy_lut_data.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"crc32": f"0x{canonical_crc32:08X}", "entries": lut_entries}, f, indent=2)

    header_lines = [
        "/**",
        " * hiram_policy_lut.h - Compiled 64-State Policy-to-LUT Evaluator ABI",
        " *",
        " * Automatically synthesized from Bayesian C-Kernel decision policy.",
        " * Exact action selection over all 64 tri-state permutations.",
        " * Table and evaluator live in src/hiram_policy_lut.c (generated together",
        " * with this header) so each has exactly one link-time placement.",
        " */",
        "",
        "#ifndef HIRAM_POLICY_LUT_H",
        "#define HIRAM_POLICY_LUT_H",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        "/* Action Definitions matching include/hiram_tables.h */",
        "#define HIRAM_LUT_ACTION_ACCEL           0",
        "#define HIRAM_LUT_ACTION_COAST           1",
        "#define HIRAM_LUT_ACTION_EMERGENCY_BRAKE 2",
        "",
        "/* Fault bitmask reported alongside the selected action. */",
        "#define HIRAM_LUT_FAULT_NONE             0x00u",
        "#define HIRAM_LUT_FAULT_INPUT_OOB        0x01u /* an input fell outside [-1,2]; action forced to EMERGENCY_BRAKE */",
        "#define HIRAM_LUT_FAULT_INTEGRITY        0x02u /* CRC32 mismatch against HIRAM_POLICY_LUT_CANONICAL_CRC32 */",
        "",
        "/* Fixed-width result ABI: stable 4-byte layout for cross-TU/cross-language callers. */",
        "typedef struct {",
        "    uint8_t  action;      /* HIRAM_LUT_ACTION_*            */",
        "    uint8_t  fault_flags; /* bitmask of HIRAM_LUT_FAULT_*  */",
        "    uint16_t reserved;    /* padding; always zero          */",
        "} hiram_lut_result_t;",
        "",
        "/* CRC32 (IEEE 802.3, poly 0xEDB88320) of HIRAM_POLICY_LUT, computed here at",
        " * generation time over the exact 64 bytes written to hiram_policy_lut.c.",
        f" * A regenerate always keeps this in sync with the table -- it is never",
        " * hand-typed. hiram_policy_lut_verify_integrity() recomputes it on the",
        " * target at startup. */",
        f"#define HIRAM_POLICY_LUT_CANONICAL_CRC32 0x{canonical_crc32:08X}UL",
        "",
        "/* Section placement for the STM32H723 (ARM Cortex-M7) memory map.",
        " * HIRAM_ITCM_TEXT places the evaluator in zero-wait-state Instruction",
        " * TCM (.itcm_text, copied from Flash by Reset_Handler -- see",
        " * stm32h723zg.ld). HIRAM_DTCM_DATA places the table in zero-wait-state",
        " * Data TCM (.dtcm_lut), likewise copied from Flash. Both expand to",
        " * nothing without -DHIRAM_TARGET_STM32H723, so the host build (Python",
        " * ctypes bridge, benchmark_lut_vs_dynamic, host tests) is unaffected. */",
        "#if defined(HIRAM_TARGET_STM32H723)",
        "#define HIRAM_ITCM_TEXT __attribute__((section(\".itcm_text\"), noinline, used))",
        "#define HIRAM_DTCM_DATA __attribute__((section(\".dtcm_lut\"), aligned(4)))",
        "#else",
        "#define HIRAM_ITCM_TEXT",
        "#define HIRAM_DTCM_DATA",
        "#endif",
        "",
        "extern const uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA;",
        "",
        "/**",
        " * Evaluates the 64-state policy LUT.",
        " * Out-of-bounds inputs set HIRAM_LUT_FAULT_INPUT_OOB and force",
        " * HIRAM_LUT_ACTION_EMERGENCY_BRAKE (fail-safe) rather than indexing the",
        " * table with an unchecked value.",
        " */",
        "HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip);",
        "",
        "/**",
        " * Recomputes CRC32 over HIRAM_POLICY_LUT and compares it against",
        " * HIRAM_POLICY_LUT_CANONICAL_CRC32. Call once at startup, before the",
        " * first hiram_policy_lut_step(): a false return means the table was",
        " * corrupted or mis-copied from Flash and the caller must fail safe",
        " * (e.g. force EMERGENCY_BRAKE and halt) rather than trust the LUT.",
        " */",
        "bool hiram_policy_lut_verify_integrity(void);",
        "",
        "#endif /* HIRAM_POLICY_LUT_H */",
        "",
    ]

    source_lines = [
        "/**",
        " * hiram_policy_lut.c - Compiled 64-State Policy-to-LUT Evaluator",
        " *",
        " * Automatically synthesized by tools/generate_policy_lut.py. Do not hand-edit:",
        " * regenerate from the sealed Bayesian C-Kernel decision policy instead.",
        " *",
        " * The table and the evaluator are given exactly one definition each (rather",
        " * than living as `static` header content) so HIRAM_DTCM_DATA/HIRAM_ITCM_TEXT",
        " * place a single resident copy of each in on-chip TCM -- inlining or",
        " * per-translation-unit static copies would defeat that placement.",
        " */",
        "",
        "#include \"hiram_policy_lut.h\"",
        "",
        "/* 64-byte pre-computed optimal Bayesian decision policy. */",
        "const uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA = {",
    ]

    for i in range(0, 64, 4):
        chunk = lut_entries[i:i+4]
        chunk_str = ", ".join(f"{c['action_index']}" for c in chunk)
        comment_items = [
            f"[{c['lut_index']:02d}]: L={c['lidar']:+d},T={c['tof']:+d},S={c['slip']:+d}->{c['action_name']}"
            for c in chunk
        ]
        source_lines.append(f"    {chunk_str}, /* {'; '.join(comment_items)} */")

    source_lines.extend([
        "};",
        "",
        "HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip) {",
        "    hiram_lut_result_t result;",
        "    result.reserved = 0;",
        "",
        "    if ((lidar < -1 || lidar > 2) || (tof < -1 || tof > 2) || (slip < -1 || slip > 2)) {",
        "        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE; /* Defensive boundary fallback */",
        "        result.fault_flags = HIRAM_LUT_FAULT_INPUT_OOB;",
        "        return result;",
        "    }",
        "",
        "    uint32_t idx = (uint32_t)(((lidar + 1) << 4) | ((tof + 1) << 2) | (slip + 1));",
        "    result.action = HIRAM_POLICY_LUT[idx & 0x3FU];",
        "    result.fault_flags = HIRAM_LUT_FAULT_NONE;",
        "    return result;",
        "}",
        "",
        "/* IEEE 802.3 CRC32, reflected, poly 0xEDB88320, init/final XOR 0xFFFFFFFF --",
        "   the same construction zlib.crc32() uses, so this matches the digest",
        "   tools/generate_policy_lut.py computed for HIRAM_POLICY_LUT_CANONICAL_CRC32.",
        "   Implemented bit-by-bit (no lookup table) so it costs no extra Flash/DTCM",
        "   placement decisions of its own; it runs once, at startup. */",
        "static uint32_t hiram_crc32(const uint8_t *data, uint32_t len) {",
        "    uint32_t crc = 0xFFFFFFFFu;",
        "    for (uint32_t i = 0; i < len; i++) {",
        "        crc ^= data[i];",
        "        for (int bit = 0; bit < 8; bit++) {",
        "            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));",
        "            crc = (crc >> 1) ^ (0xEDB88320u & mask);",
        "        }",
        "    }",
        "    return crc ^ 0xFFFFFFFFu;",
        "}",
        "",
        "bool hiram_policy_lut_verify_integrity(void) {",
        "    return hiram_crc32(HIRAM_POLICY_LUT, sizeof(HIRAM_POLICY_LUT)) == HIRAM_POLICY_LUT_CANONICAL_CRC32;",
        "}",
        "",
    ])

    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines))
    with open(source_path, "w", encoding="utf-8") as f:
        f.write("\n".join(source_lines))

    print(f"[OK] Successfully synthesized {header_path}")
    print(f"[OK] Successfully synthesized {source_path}")
    print(f"[OK] Canonical CRC32: 0x{canonical_crc32:08X}")
    print(f"[OK] Wrote verification dataset to {json_path}")


if __name__ == "__main__":
    generate_lut()