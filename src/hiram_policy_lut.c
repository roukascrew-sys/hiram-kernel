/**
 * hiram_policy_lut.c - Compiled 64-State Policy-to-LUT Evaluator
 *
 * Automatically synthesized by tools/generate_policy_lut.py. Do not hand-edit:
 * regenerate from the sealed Bayesian C-Kernel decision policy instead.
 *
 * The table and the evaluator are given exactly one definition each (rather
 * than living as `static` header content) so HIRAM_DTCM_DATA/HIRAM_ITCM_TEXT
 * place a single resident copy of each in on-chip TCM -- inlining or
 * per-translation-unit static copies would defeat that placement.
 */

#include "hiram_policy_lut.h"

/* 64-cell pre-computed optimal Bayesian decision policy, dual-rail
   encoded (see HIRAM_LUT_ENCODE_CELL in the header): 0x0C = ACCEL,
   0x09 = COAST, 0x06 = EMERGENCY_BRAKE. Not const -- this is .dtcm_lut
   RAM on the target; the MPU, not the type system, write-protects it. */
uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA = {
    0x09, 0x09, 0x06, 0x09, /* [00]: L=-1,T=-1,S=-1->COAST; [01]: L=-1,T=-1,S=+0->COAST; [02]: L=-1,T=-1,S=+1->EMERGENCY_BRAKE; [03]: L=-1,T=-1,S=+2->COAST */
    0x0C, 0x0C, 0x09, 0x0C, /* [04]: L=-1,T=+0,S=-1->ACCEL; [05]: L=-1,T=+0,S=+0->ACCEL; [06]: L=-1,T=+0,S=+1->COAST; [07]: L=-1,T=+0,S=+2->ACCEL */
    0x06, 0x06, 0x06, 0x06, /* [08]: L=-1,T=+1,S=-1->EMERGENCY_BRAKE; [09]: L=-1,T=+1,S=+0->EMERGENCY_BRAKE; [10]: L=-1,T=+1,S=+1->EMERGENCY_BRAKE; [11]: L=-1,T=+1,S=+2->EMERGENCY_BRAKE */
    0x09, 0x09, 0x06, 0x09, /* [12]: L=-1,T=+2,S=-1->COAST; [13]: L=-1,T=+2,S=+0->COAST; [14]: L=-1,T=+2,S=+1->EMERGENCY_BRAKE; [15]: L=-1,T=+2,S=+2->COAST */
    0x0C, 0x0C, 0x09, 0x0C, /* [16]: L=+0,T=-1,S=-1->ACCEL; [17]: L=+0,T=-1,S=+0->ACCEL; [18]: L=+0,T=-1,S=+1->COAST; [19]: L=+0,T=-1,S=+2->ACCEL */
    0x0C, 0x0C, 0x0C, 0x0C, /* [20]: L=+0,T=+0,S=-1->ACCEL; [21]: L=+0,T=+0,S=+0->ACCEL; [22]: L=+0,T=+0,S=+1->ACCEL; [23]: L=+0,T=+0,S=+2->ACCEL */
    0x09, 0x09, 0x06, 0x09, /* [24]: L=+0,T=+1,S=-1->COAST; [25]: L=+0,T=+1,S=+0->COAST; [26]: L=+0,T=+1,S=+1->EMERGENCY_BRAKE; [27]: L=+0,T=+1,S=+2->COAST */
    0x0C, 0x0C, 0x09, 0x0C, /* [28]: L=+0,T=+2,S=-1->ACCEL; [29]: L=+0,T=+2,S=+0->ACCEL; [30]: L=+0,T=+2,S=+1->COAST; [31]: L=+0,T=+2,S=+2->ACCEL */
    0x06, 0x06, 0x06, 0x06, /* [32]: L=+1,T=-1,S=-1->EMERGENCY_BRAKE; [33]: L=+1,T=-1,S=+0->EMERGENCY_BRAKE; [34]: L=+1,T=-1,S=+1->EMERGENCY_BRAKE; [35]: L=+1,T=-1,S=+2->EMERGENCY_BRAKE */
    0x06, 0x06, 0x06, 0x06, /* [36]: L=+1,T=+0,S=-1->EMERGENCY_BRAKE; [37]: L=+1,T=+0,S=+0->EMERGENCY_BRAKE; [38]: L=+1,T=+0,S=+1->EMERGENCY_BRAKE; [39]: L=+1,T=+0,S=+2->EMERGENCY_BRAKE */
    0x06, 0x06, 0x06, 0x06, /* [40]: L=+1,T=+1,S=-1->EMERGENCY_BRAKE; [41]: L=+1,T=+1,S=+0->EMERGENCY_BRAKE; [42]: L=+1,T=+1,S=+1->EMERGENCY_BRAKE; [43]: L=+1,T=+1,S=+2->EMERGENCY_BRAKE */
    0x06, 0x06, 0x06, 0x06, /* [44]: L=+1,T=+2,S=-1->EMERGENCY_BRAKE; [45]: L=+1,T=+2,S=+0->EMERGENCY_BRAKE; [46]: L=+1,T=+2,S=+1->EMERGENCY_BRAKE; [47]: L=+1,T=+2,S=+2->EMERGENCY_BRAKE */
    0x09, 0x09, 0x06, 0x09, /* [48]: L=+2,T=-1,S=-1->COAST; [49]: L=+2,T=-1,S=+0->COAST; [50]: L=+2,T=-1,S=+1->EMERGENCY_BRAKE; [51]: L=+2,T=-1,S=+2->COAST */
    0x0C, 0x0C, 0x09, 0x0C, /* [52]: L=+2,T=+0,S=-1->ACCEL; [53]: L=+2,T=+0,S=+0->ACCEL; [54]: L=+2,T=+0,S=+1->COAST; [55]: L=+2,T=+0,S=+2->ACCEL */
    0x06, 0x06, 0x06, 0x06, /* [56]: L=+2,T=+1,S=-1->EMERGENCY_BRAKE; [57]: L=+2,T=+1,S=+0->EMERGENCY_BRAKE; [58]: L=+2,T=+1,S=+1->EMERGENCY_BRAKE; [59]: L=+2,T=+1,S=+2->EMERGENCY_BRAKE */
    0x09, 0x09, 0x06, 0x09, /* [60]: L=+2,T=+2,S=-1->COAST; [61]: L=+2,T=+2,S=+0->COAST; [62]: L=+2,T=+2,S=+1->EMERGENCY_BRAKE; [63]: L=+2,T=+2,S=+2->COAST */
};

/* Operational latch (B1).

   g_lut_operational gates every step; it starts false, so a caller that
   forgets hiram_policy_lut_init() gets EMERGENCY_BRAKE rather than
   whatever bytes happen to be sitting in DTCM before the boot copy.

   g_lut_integrity_latched is the one-way door: once any check has seen
   this table corrupt, no later re-read can put it back in service. A
   transient that clears on the next read is not evidence the table is
   sound -- it is evidence the memory is unreliable, which is the same
   conclusion. Only a reset clears it.

   Both are volatile: they are written from the step path and read by
   fault handlers and the startup banner, and must not be cached across
   those boundaries. */
static volatile bool g_lut_operational = false;
static volatile bool g_lut_integrity_latched = false;

bool hiram_policy_lut_cell_is_valid(uint8_t cell) {
    return hiram_lut_cell_is_valid(cell);
}

bool hiram_policy_lut_is_operational(void) {
    return g_lut_operational;
}

HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip) {
    hiram_lut_result_t result;
    result.reserved = 0u;

    /* Checked first: an un-initialised or latched-out table must not be
       indexed at all, whatever the inputs look like. */
    if (!g_lut_operational) {
        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE;
        result.fault_flags = HIRAM_LUT_FAULT_INTEGRITY;
        return result;
    }

    if ((lidar < -1) || (lidar > 2) ||
        (tof   < -1) || (tof   > 2) ||
        (slip  < -1) || (slip  > 2)) {
        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE; /* Defensive boundary fallback */
        result.fault_flags = HIRAM_LUT_FAULT_INPUT_OOB;
        return result;
    }

    /* M1: MISRA C:2012 Rule 10.1 -- shift operands must be unsigned. The
       range check above has already established each value is in [-1, 2],
       so each (v + 1) is in [0, 3] and the casts are value-preserving.
       The trailing mask is redundant given that, and kept as a defence
       against a future edit to the range check silently widening idx. */
    const uint32_t l = (uint32_t)(lidar + 1);
    const uint32_t t = (uint32_t)(tof   + 1);
    const uint32_t s = (uint32_t)(slip  + 1);
    const uint32_t idx = ((l << 4u) | (t << 2u) | s) & 0x3Fu;

    /* Re-decode on every call, not once at startup: this is the check
       that catches a bit-flip which arrived after the CRC32 ran. */
    const uint8_t cell = HIRAM_POLICY_LUT[idx];
    if (!hiram_lut_cell_is_valid(cell)) {
        g_lut_integrity_latched = true;
        g_lut_operational = false;
        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE;
        result.fault_flags = HIRAM_LUT_FAULT_INTEGRITY;
        return result;
    }

    result.action = (uint8_t)(cell & 0x03u);
    result.fault_flags = HIRAM_LUT_FAULT_NONE;
    return result;
}

/* IEEE 802.3 CRC32, reflected, poly 0xEDB88320, init/final XOR 0xFFFFFFFF --
   the same construction zlib.crc32() uses, so this matches the digest
   tools/generate_policy_lut.py computed for HIRAM_POLICY_LUT_CANONICAL_CRC32.
   Implemented bit-by-bit (no lookup table) so it costs no extra Flash/DTCM
   placement decisions of its own; it runs once, at startup. */
static uint32_t hiram_crc32(const uint8_t *data, uint32_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (uint32_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; bit++) {
            uint32_t mask = (uint32_t)(-(int32_t)(crc & 1u));
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return crc ^ 0xFFFFFFFFu;
}

bool hiram_policy_lut_verify_integrity(void) {
    return hiram_crc32(HIRAM_POLICY_LUT, (uint32_t)sizeof(HIRAM_POLICY_LUT)) == HIRAM_POLICY_LUT_CANONICAL_CRC32;
}

bool hiram_policy_lut_init(void) {
    uint32_t i;

    /* One-way door: a table that has already been seen corrupt stays out
       of service for the life of this power cycle. */
    if (g_lut_integrity_latched) {
        g_lut_operational = false;
        return false;
    }

    if (!hiram_policy_lut_verify_integrity()) {
        g_lut_integrity_latched = true;
        g_lut_operational = false;
        return false;
    }

    /* Defence in depth against a CRC32 collision: a 32-bit digest over 64
       bytes is not injective, and the dual-rail check is a structural
       property no colliding table can satisfy by accident. */
    for (i = 0u; i < (uint32_t)sizeof(HIRAM_POLICY_LUT); i++) {
        if (!hiram_lut_cell_is_valid(HIRAM_POLICY_LUT[i])) {
            g_lut_integrity_latched = true;
            g_lut_operational = false;
            return false;
        }
    }

    g_lut_operational = true;
    return true;
}
