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
#define HIRAM_LUT_FAULT_INTEGRITY        0x02u /* CRC32 mismatch against HIRAM_POLICY_LUT_CANONICAL_CRC32 */

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
#define HIRAM_POLICY_LUT_CANONICAL_CRC32 0x255BACBCUL

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

extern const uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA;

/**
 * Evaluates the 64-state policy LUT.
 * Out-of-bounds inputs set HIRAM_LUT_FAULT_INPUT_OOB and force
 * HIRAM_LUT_ACTION_EMERGENCY_BRAKE (fail-safe) rather than indexing the
 * table with an unchecked value.
 */
HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip);

/**
 * Recomputes CRC32 over HIRAM_POLICY_LUT and compares it against
 * HIRAM_POLICY_LUT_CANONICAL_CRC32. Call once at startup, before the
 * first hiram_policy_lut_step(): a false return means the table was
 * corrupted or mis-copied from Flash and the caller must fail safe
 * (e.g. force EMERGENCY_BRAKE and halt) rather than trust the LUT.
 */
bool hiram_policy_lut_verify_integrity(void);

#endif /* HIRAM_POLICY_LUT_H */
