/**
 * hiram_policy_lut.h - Compiled 64-State Policy-to-LUT Evaluator ABI
 *
 * Automatically synthesized from Bayesian C-Kernel decision policy.
 * Exact action selection over all 64 tri-state permutations.
 * Table and evaluator live in src/hiram_policy_lut.c (generated together
 * with this header) so each has exactly one link-time placement.
 */

#ifndef HIRAM_POLICY_LUT_H
#define HIRAM_POLICY_LUT_H

#include <stdint.h>
#include <stdbool.h>

/* Action Definitions matching include/hiram_tables.h */
#define HIRAM_LUT_ACTION_ACCEL           0
#define HIRAM_LUT_ACTION_COAST           1
#define HIRAM_LUT_ACTION_EMERGENCY_BRAKE 2

/* Fault bitmask reported alongside the selected action. */
#define HIRAM_LUT_FAULT_NONE             0x00u
#define HIRAM_LUT_FAULT_INPUT_OOB        0x01u /* an input fell outside [-1,2]; action forced to EMERGENCY_BRAKE */
#define HIRAM_LUT_FAULT_INTEGRITY        0x02u /* table not in service: init not run, init failed, or a dual-rail parity violation latched */

/* Fixed-width result ABI: stable 4-byte layout for cross-TU/cross-language callers. */
typedef struct {
    uint8_t  action;      /* HIRAM_LUT_ACTION_*            */
    uint8_t  fault_flags; /* bitmask of HIRAM_LUT_FAULT_*  */
    uint16_t reserved;    /* padding; always zero          */
} hiram_lut_result_t;

/* CRC32 (IEEE 802.3, poly 0xEDB88320) of HIRAM_POLICY_LUT, computed here at
 * generation time over the exact 64 bytes written to hiram_policy_lut.c.
 * A regenerate always keeps this in sync with the table -- it is never
 * hand-typed. hiram_policy_lut_verify_integrity() recomputes it on the
 * target at startup. */
#define HIRAM_POLICY_LUT_CANONICAL_CRC32 0xDBE2F4BAUL

/* Section placement for the STM32H723 (ARM Cortex-M7) memory map.
 * HIRAM_ITCM_TEXT places the evaluator in zero-wait-state Instruction
 * TCM (.itcm_text, copied from Flash by Reset_Handler -- see
 * stm32h723zg.ld). HIRAM_DTCM_DATA places the table in zero-wait-state
 * Data TCM (.dtcm_lut), likewise copied from Flash. Both expand to
 * nothing without -DHIRAM_TARGET_STM32H723, so the host build (Python
 * ctypes bridge, benchmark_lut_vs_dynamic, host tests) is unaffected. */
#if defined(HIRAM_TARGET_STM32H723)
#define HIRAM_ITCM_TEXT __attribute__((section(".itcm_text"), noinline, used))
#define HIRAM_DTCM_DATA __attribute__((section(".dtcm_lut"), aligned(4)))
#else
#define HIRAM_ITCM_TEXT
#define HIRAM_DTCM_DATA
#endif

/* Dual-rail cell encoding (B1). Each byte carries the action in bits
 * [1:0] and its 2-bit complement in bits [3:2]; bits [7:4] are zero.
 * Only 3 of the 256 byte values are legal, and every single-bit
 * mutation of a legal value is illegal -- so a post-startup SRAM flip
 * cannot turn one valid action into another valid action, which is
 * precisely what a bare action index allowed (EMERGENCY_BRAKE 0b10 ->
 * ACCEL 0b00 is a single bit). tools/generate_policy_lut.py proves that
 * property exhaustively at generation time before emitting this file. */
#define HIRAM_LUT_ENCODE_CELL(action) \
    ((uint8_t)(((uint32_t)(action) & 0x03u) | ((~(uint32_t)(action) & 0x03u) << 2u)))

/* Decode-side check. `static inline` (not a macro) so it evaluates its
 * argument once and type-checks, and so hiram_policy_lut_step() can use
 * the identical predicate the exported hiram_policy_lut_cell_is_valid()
 * exposes for exhaustive verification -- one definition, no drift. */
static inline bool hiram_lut_cell_is_valid(uint8_t cell) {
    /* Upper nibble must be clear: catches a flip in bits [7:4], which
       the rail comparison below cannot see. */
    if ((cell & 0xF0u) != 0x00u) {
        return false;
    }
    /* Rail and complement must differ in both bits: catches any single
       flip in bits [3:0]. */
    if ((uint8_t)((cell ^ (uint8_t)(cell >> 2u)) & 0x03u) != 0x03u) {
        return false;
    }
    /* 0b11 passes the rail test but is not an action this policy emits. */
    return (uint8_t)(cell & 0x03u) <= (uint8_t)HIRAM_LUT_ACTION_EMERGENCY_BRAKE;
}

/* The table is deliberately NOT const-qualified. On the STM32H723 it is
 * RAM: stm32h723zg.ld places it in .dtcm_lut and Reset_Handler copies it
 * there from Flash at boot, so `const` would have described the storage
 * inaccurately and implied an immutability the hardware does not provide.
 * Write protection is enforced where it can actually be enforced -- MPU
 * region 1, Read-Only/Execute-Never, armed by mpu_lock_regions_post_init()
 * in src/target_main.c once the boot copy and integrity check are done.
 * Keeping it writable is also what lets tests/test_sil_fault_injection.c
 * inject a real post-startup bit-flip into the live table rather than
 * into a copy the evaluator never reads. */
extern uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA;

/**
 * Brings the LUT into service. Verifies HIRAM_POLICY_LUT against
 * HIRAM_POLICY_LUT_CANONICAL_CRC32 *and* checks that all 64 cells decode
 * as well-formed dual-rail pairs, then latches the table operational.
 * Must be called once at startup, after the Flash->DTCM copy: until it
 * returns true, every hiram_policy_lut_step() call reports
 * HIRAM_LUT_FAULT_INTEGRITY and HIRAM_LUT_ACTION_EMERGENCY_BRAKE.
 *
 * An integrity failure -- here or later, in hiram_policy_lut_step() --
 * latches permanently: this function will not bring a table that has
 * once been observed corrupt back into service, and returns false on
 * every subsequent call. Only a reset clears that state.
 */
bool hiram_policy_lut_init(void);

/**
 * True iff the LUT is currently in service (init succeeded and no
 * dual-rail parity violation has been observed since).
 */
bool hiram_policy_lut_is_operational(void);

/**
 * The exact decode predicate hiram_policy_lut_step() applies to the cell
 * it is about to return, exported so the SIL suite can sweep it over all
 * 256 byte values and all 512 single-bit mutations of the live table
 * without having to restate the check (and risk testing a copy of the
 * logic rather than the logic itself).
 */
bool hiram_policy_lut_cell_is_valid(uint8_t cell);

/**
 * Evaluates the 64-state policy LUT.
 * Fail-safe on three independent conditions, each forcing
 * HIRAM_LUT_ACTION_EMERGENCY_BRAKE rather than a trusted action:
 *   - table not in service        -> HIRAM_LUT_FAULT_INTEGRITY
 *   - an input outside [-1, 2]    -> HIRAM_LUT_FAULT_INPUT_OOB
 *   - selected cell fails decode  -> HIRAM_LUT_FAULT_INTEGRITY (latches)
 */
HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip);

/**
 * Recomputes CRC32 over HIRAM_POLICY_LUT and compares it against
 * HIRAM_POLICY_LUT_CANONICAL_CRC32. Pure: it neither consults nor
 * changes the operational latch, so a caller can use it as a periodic
 * background scrub. hiram_policy_lut_init() is what actually gates the
 * table into service; this is the digest half of that decision.
 */
bool hiram_policy_lut_verify_integrity(void);

#endif /* HIRAM_POLICY_LUT_H */
