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

B1 (dual-rail encoding). A startup-only CRC32 cannot see a bit-flip that lands
in DTCM *after* the check has run, so the pre-audit review was able to mutate
EMERGENCY_BRAKE (2) into ACCEL (0) in a live table and have every subsequent
step report fault_flags == 0. Each cell is therefore stored as its own
complement pair rather than as a bare action index:

    cell = (action & 0x03) | ((~action & 0x03) << 2)

    ACCEL (0) -> 0x0C     COAST (1) -> 0x09     EMERGENCY_BRAKE (2) -> 0x06

Exactly 3 of the 256 byte values are legal, and the three legal ones are
mutually distinguishable under any single-bit mutation: a flip in bits [3:0]
breaks the rail-vs-complement relation, and a flip in bits [7:4] breaks the
"upper nibble is zero" invariant. hiram_policy_lut_step() re-decodes and
re-checks that relation on the cell it is about to return, every call -- so
detection no longer depends on when the corruption arrived relative to
startup. A violation latches the table out of service permanently (only a
reset clears it), because a table that has been observed to be corrupt cannot
be argued back into trustworthiness by a later re-read.
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

#: Number of distinct actions the policy can select (ACCEL/COAST/EMERGENCY_BRAKE).
N_ACTIONS = 3


def encode_cell(action: int) -> int:
    """Dual-rail cell encoding -- the single source of truth for the byte
    values written into src/hiram_policy_lut.c.

    Mirrors hiram_lut_cell_is_valid()'s decode side in the generated header
    exactly: the low rail carries the action, the high rail its 2-bit
    complement, and bits [7:4] are always zero.
    """
    if action not in range(N_ACTIONS):
        raise ValueError(f"action {action!r} outside [0, {N_ACTIONS})")
    return (action & 0x03) | ((~action & 0x03) << 2)


def cell_is_valid(cell: int) -> bool:
    """Python mirror of the generated C hiram_lut_cell_is_valid()."""
    if cell & 0xF0:
        return False
    if ((cell ^ (cell >> 2)) & 0x03) != 0x03:
        return False
    return (cell & 0x03) < N_ACTIONS


def _assert_single_bit_flip_coverage() -> None:
    """Proves, at generation time, the property the encoding exists for: every
    single-bit mutation of every legal cell value is rejected by the decode
    check. A generator that emitted an encoding without this property would
    produce a table whose runtime parity check silently passes corruption --
    exactly the B1 failure mode, reintroduced one layer down."""
    legal = {encode_cell(a) for a in range(N_ACTIONS)}
    assert len(legal) == N_ACTIONS, f"encoding is not injective: {legal}"
    for cell in legal:
        assert cell_is_valid(cell), f"legal cell 0x{cell:02X} rejected by its own decode check"
        for bit in range(8):
            mutated = cell ^ (1 << bit)
            assert not cell_is_valid(mutated), (
                f"single-bit flip of bit {bit} in 0x{cell:02X} -> 0x{mutated:02X} "
                "survives the dual-rail check"
            )
    accepted = [c for c in range(256) if cell_is_valid(c)]
    assert set(accepted) == legal, f"decode accepts illegal values: {sorted(set(accepted) - legal)}"


def generate_lut():
    lut_entries = []

    _assert_single_bit_flip_coverage()
    print("Generating 64-state Bayesian policy lookup table...")

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
                    "encoded_cell": encode_cell(action_idx),
                    "expected_losses": step_res["expected_losses"],
                })

    # Sort strictly by LUT index 0..63
    lut_entries.sort(key=lambda x: x["lut_index"])

    # The CRC32 covers the *encoded* bytes, because those are the 64 bytes that
    # are actually written to src/hiram_policy_lut.c, linked into .dtcm_lut and
    # re-read by hiram_policy_lut_verify_integrity() on the target.
    table_bytes = bytes(e["encoded_cell"] for e in lut_entries)
    assert len(table_bytes) == 64, f"expected 64 table entries, got {len(table_bytes)}"
    assert all(cell_is_valid(b) for b in table_bytes), "emitted a cell that fails its own decode check"
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
        "#define HIRAM_LUT_FAULT_INTEGRITY        0x02u /* table not in service: init not run, init failed, or a dual-rail parity violation latched */",
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
        "/* Dual-rail cell encoding (B1). Each byte carries the action in bits",
        " * [1:0] and its 2-bit complement in bits [3:2]; bits [7:4] are zero.",
        " * Only 3 of the 256 byte values are legal, and every single-bit",
        " * mutation of a legal value is illegal -- so a post-startup SRAM flip",
        " * cannot turn one valid action into another valid action, which is",
        " * precisely what a bare action index allowed (EMERGENCY_BRAKE 0b10 ->",
        " * ACCEL 0b00 is a single bit). tools/generate_policy_lut.py proves that",
        " * property exhaustively at generation time before emitting this file. */",
        "#define HIRAM_LUT_ENCODE_CELL(action) \\",
        "    ((uint8_t)(((uint32_t)(action) & 0x03u) | ((~(uint32_t)(action) & 0x03u) << 2u)))",
        "",
        "/* Decode-side check. `static inline` (not a macro) so it evaluates its",
        " * argument once and type-checks, and so hiram_policy_lut_step() can use",
        " * the identical predicate the exported hiram_policy_lut_cell_is_valid()",
        " * exposes for exhaustive verification -- one definition, no drift. */",
        "static inline bool hiram_lut_cell_is_valid(uint8_t cell) {",
        "    /* Upper nibble must be clear: catches a flip in bits [7:4], which",
        "       the rail comparison below cannot see. */",
        "    if ((cell & 0xF0u) != 0x00u) {",
        "        return false;",
        "    }",
        "    /* Rail and complement must differ in both bits: catches any single",
        "       flip in bits [3:0]. */",
        "    if ((uint8_t)((cell ^ (uint8_t)(cell >> 2u)) & 0x03u) != 0x03u) {",
        "        return false;",
        "    }",
        "    /* 0b11 passes the rail test but is not an action this policy emits. */",
        "    return (uint8_t)(cell & 0x03u) <= (uint8_t)HIRAM_LUT_ACTION_EMERGENCY_BRAKE;",
        "}",
        "",
        "/* The table is deliberately NOT const-qualified. On the STM32H723 it is",
        " * RAM: stm32h723zg.ld places it in .dtcm_lut and Reset_Handler copies it",
        " * there from Flash at boot, so `const` would have described the storage",
        " * inaccurately and implied an immutability the hardware does not provide.",
        " * Write protection is enforced where it can actually be enforced -- MPU",
        " * region 1, Read-Only/Execute-Never, armed by mpu_lock_regions_post_init()",
        " * in src/target_main.c once the boot copy and integrity check are done.",
        " * Keeping it writable is also what lets tests/test_sil_fault_injection.c",
        " * inject a real post-startup bit-flip into the live table rather than",
        " * into a copy the evaluator never reads. */",
        "extern uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA;",
        "",
        "/**",
        " * Brings the LUT into service. Verifies HIRAM_POLICY_LUT against",
        " * HIRAM_POLICY_LUT_CANONICAL_CRC32 *and* checks that all 64 cells decode",
        " * as well-formed dual-rail pairs, then latches the table operational.",
        " * Must be called once at startup, after the Flash->DTCM copy: until it",
        " * returns true, every hiram_policy_lut_step() call reports",
        " * HIRAM_LUT_FAULT_INTEGRITY and HIRAM_LUT_ACTION_EMERGENCY_BRAKE.",
        " *",
        " * An integrity failure -- here or later, in hiram_policy_lut_step() --",
        " * latches permanently: this function will not bring a table that has",
        " * once been observed corrupt back into service, and returns false on",
        " * every subsequent call. Only a reset clears that state.",
        " */",
        "bool hiram_policy_lut_init(void);",
        "",
        "/**",
        " * True iff the LUT is currently in service (init succeeded and no",
        " * dual-rail parity violation has been observed since).",
        " */",
        "bool hiram_policy_lut_is_operational(void);",
        "",
        "/**",
        " * The exact decode predicate hiram_policy_lut_step() applies to the cell",
        " * it is about to return, exported so the SIL suite can sweep it over all",
        " * 256 byte values and all 512 single-bit mutations of the live table",
        " * without having to restate the check (and risk testing a copy of the",
        " * logic rather than the logic itself).",
        " */",
        "bool hiram_policy_lut_cell_is_valid(uint8_t cell);",
        "",
        "/**",
        " * Evaluates the 64-state policy LUT.",
        " * Fail-safe on three independent conditions, each forcing",
        " * HIRAM_LUT_ACTION_EMERGENCY_BRAKE rather than a trusted action:",
        " *   - table not in service        -> HIRAM_LUT_FAULT_INTEGRITY",
        " *   - an input outside [-1, 2]    -> HIRAM_LUT_FAULT_INPUT_OOB",
        " *   - selected cell fails decode  -> HIRAM_LUT_FAULT_INTEGRITY (latches)",
        " */",
        "HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip);",
        "",
        "/**",
        " * Recomputes CRC32 over HIRAM_POLICY_LUT and compares it against",
        " * HIRAM_POLICY_LUT_CANONICAL_CRC32. Pure: it neither consults nor",
        " * changes the operational latch, so a caller can use it as a periodic",
        " * background scrub. hiram_policy_lut_init() is what actually gates the",
        " * table into service; this is the digest half of that decision.",
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
        "/* 64-cell pre-computed optimal Bayesian decision policy, dual-rail",
        "   encoded (see HIRAM_LUT_ENCODE_CELL in the header): 0x0C = ACCEL,",
        "   0x09 = COAST, 0x06 = EMERGENCY_BRAKE. Not const -- this is .dtcm_lut",
        "   RAM on the target; the MPU, not the type system, write-protects it. */",
        "uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA = {",
    ]

    for i in range(0, 64, 4):
        chunk = lut_entries[i:i+4]
        chunk_str = ", ".join(f"0x{c['encoded_cell']:02X}" for c in chunk)
        comment_items = [
            f"[{c['lut_index']:02d}]: L={c['lidar']:+d},T={c['tof']:+d},S={c['slip']:+d}->{c['action_name']}"
            for c in chunk
        ]
        source_lines.append(f"    {chunk_str}, /* {'; '.join(comment_items)} */")

    source_lines.extend([
        "};",
        "",
        "/* Operational latch (B1).",
        "",
        "   g_lut_operational gates every step; it starts false, so a caller that",
        "   forgets hiram_policy_lut_init() gets EMERGENCY_BRAKE rather than",
        "   whatever bytes happen to be sitting in DTCM before the boot copy.",
        "",
        "   g_lut_integrity_latched is the one-way door: once any check has seen",
        "   this table corrupt, no later re-read can put it back in service. A",
        "   transient that clears on the next read is not evidence the table is",
        "   sound -- it is evidence the memory is unreliable, which is the same",
        "   conclusion. Only a reset clears it.",
        "",
        "   Both are volatile: they are written from the step path and read by",
        "   fault handlers and the startup banner, and must not be cached across",
        "   those boundaries. */",
        "static volatile bool g_lut_operational = false;",
        "static volatile bool g_lut_integrity_latched = false;",
        "",
        "bool hiram_policy_lut_cell_is_valid(uint8_t cell) {",
        "    return hiram_lut_cell_is_valid(cell);",
        "}",
        "",
        "bool hiram_policy_lut_is_operational(void) {",
        "    return g_lut_operational;",
        "}",
        "",
        "HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip) {",
        "    hiram_lut_result_t result;",
        "    result.reserved = 0u;",
        "",
        "    /* Checked first: an un-initialised or latched-out table must not be",
        "       indexed at all, whatever the inputs look like. */",
        "    if (!g_lut_operational) {",
        "        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE;",
        "        result.fault_flags = HIRAM_LUT_FAULT_INTEGRITY;",
        "        return result;",
        "    }",
        "",
        "    if ((lidar < -1) || (lidar > 2) ||",
        "        (tof   < -1) || (tof   > 2) ||",
        "        (slip  < -1) || (slip  > 2)) {",
        "        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE; /* Defensive boundary fallback */",
        "        result.fault_flags = HIRAM_LUT_FAULT_INPUT_OOB;",
        "        return result;",
        "    }",
        "",
        "    /* M1: MISRA C:2012 Rule 10.1 -- shift operands must be unsigned. The",
        "       range check above has already established each value is in [-1, 2],",
        "       so each (v + 1) is in [0, 3] and the casts are value-preserving.",
        "       The trailing mask is redundant given that, and kept as a defence",
        "       against a future edit to the range check silently widening idx. */",
        "    const uint32_t l = (uint32_t)(lidar + 1);",
        "    const uint32_t t = (uint32_t)(tof   + 1);",
        "    const uint32_t s = (uint32_t)(slip  + 1);",
        "    const uint32_t idx = ((l << 4u) | (t << 2u) | s) & 0x3Fu;",
        "",
        "    /* Re-decode on every call, not once at startup: this is the check",
        "       that catches a bit-flip which arrived after the CRC32 ran. */",
        "    const uint8_t cell = HIRAM_POLICY_LUT[idx];",
        "    if (!hiram_lut_cell_is_valid(cell)) {",
        "        g_lut_integrity_latched = true;",
        "        g_lut_operational = false;",
        "        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE;",
        "        result.fault_flags = HIRAM_LUT_FAULT_INTEGRITY;",
        "        return result;",
        "    }",
        "",
        "    result.action = (uint8_t)(cell & 0x03u);",
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
        "    return hiram_crc32(HIRAM_POLICY_LUT, (uint32_t)sizeof(HIRAM_POLICY_LUT)) == HIRAM_POLICY_LUT_CANONICAL_CRC32;",
        "}",
        "",
        "bool hiram_policy_lut_init(void) {",
        "    uint32_t i;",
        "",
        "    /* One-way door: a table that has already been seen corrupt stays out",
        "       of service for the life of this power cycle. */",
        "    if (g_lut_integrity_latched) {",
        "        g_lut_operational = false;",
        "        return false;",
        "    }",
        "",
        "    if (!hiram_policy_lut_verify_integrity()) {",
        "        g_lut_integrity_latched = true;",
        "        g_lut_operational = false;",
        "        return false;",
        "    }",
        "",
        "    /* Defence in depth against a CRC32 collision: a 32-bit digest over 64",
        "       bytes is not injective, and the dual-rail check is a structural",
        "       property no colliding table can satisfy by accident. */",
        "    for (i = 0u; i < (uint32_t)sizeof(HIRAM_POLICY_LUT); i++) {",
        "        if (!hiram_lut_cell_is_valid(HIRAM_POLICY_LUT[i])) {",
        "            g_lut_integrity_latched = true;",
        "            g_lut_operational = false;",
        "            return false;",
        "        }",
        "    }",
        "",
        "    g_lut_operational = true;",
        "    return true;",
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