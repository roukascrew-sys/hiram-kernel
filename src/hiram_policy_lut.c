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

/* 64-byte pre-computed optimal Bayesian decision policy. */
const uint8_t HIRAM_POLICY_LUT[64] HIRAM_DTCM_DATA = {
    1, 1, 2, 1, /* [00]: L=-1,T=-1,S=-1->COAST; [01]: L=-1,T=-1,S=+0->COAST; [02]: L=-1,T=-1,S=+1->EMERGENCY_BRAKE; [03]: L=-1,T=-1,S=+2->COAST */
    0, 0, 1, 0, /* [04]: L=-1,T=+0,S=-1->ACCEL; [05]: L=-1,T=+0,S=+0->ACCEL; [06]: L=-1,T=+0,S=+1->COAST; [07]: L=-1,T=+0,S=+2->ACCEL */
    2, 2, 2, 2, /* [08]: L=-1,T=+1,S=-1->EMERGENCY_BRAKE; [09]: L=-1,T=+1,S=+0->EMERGENCY_BRAKE; [10]: L=-1,T=+1,S=+1->EMERGENCY_BRAKE; [11]: L=-1,T=+1,S=+2->EMERGENCY_BRAKE */
    1, 1, 2, 1, /* [12]: L=-1,T=+2,S=-1->COAST; [13]: L=-1,T=+2,S=+0->COAST; [14]: L=-1,T=+2,S=+1->EMERGENCY_BRAKE; [15]: L=-1,T=+2,S=+2->COAST */
    0, 0, 1, 0, /* [16]: L=+0,T=-1,S=-1->ACCEL; [17]: L=+0,T=-1,S=+0->ACCEL; [18]: L=+0,T=-1,S=+1->COAST; [19]: L=+0,T=-1,S=+2->ACCEL */
    0, 0, 0, 0, /* [20]: L=+0,T=+0,S=-1->ACCEL; [21]: L=+0,T=+0,S=+0->ACCEL; [22]: L=+0,T=+0,S=+1->ACCEL; [23]: L=+0,T=+0,S=+2->ACCEL */
    1, 1, 2, 1, /* [24]: L=+0,T=+1,S=-1->COAST; [25]: L=+0,T=+1,S=+0->COAST; [26]: L=+0,T=+1,S=+1->EMERGENCY_BRAKE; [27]: L=+0,T=+1,S=+2->COAST */
    0, 0, 1, 0, /* [28]: L=+0,T=+2,S=-1->ACCEL; [29]: L=+0,T=+2,S=+0->ACCEL; [30]: L=+0,T=+2,S=+1->COAST; [31]: L=+0,T=+2,S=+2->ACCEL */
    2, 2, 2, 2, /* [32]: L=+1,T=-1,S=-1->EMERGENCY_BRAKE; [33]: L=+1,T=-1,S=+0->EMERGENCY_BRAKE; [34]: L=+1,T=-1,S=+1->EMERGENCY_BRAKE; [35]: L=+1,T=-1,S=+2->EMERGENCY_BRAKE */
    2, 2, 2, 2, /* [36]: L=+1,T=+0,S=-1->EMERGENCY_BRAKE; [37]: L=+1,T=+0,S=+0->EMERGENCY_BRAKE; [38]: L=+1,T=+0,S=+1->EMERGENCY_BRAKE; [39]: L=+1,T=+0,S=+2->EMERGENCY_BRAKE */
    2, 2, 2, 2, /* [40]: L=+1,T=+1,S=-1->EMERGENCY_BRAKE; [41]: L=+1,T=+1,S=+0->EMERGENCY_BRAKE; [42]: L=+1,T=+1,S=+1->EMERGENCY_BRAKE; [43]: L=+1,T=+1,S=+2->EMERGENCY_BRAKE */
    2, 2, 2, 2, /* [44]: L=+1,T=+2,S=-1->EMERGENCY_BRAKE; [45]: L=+1,T=+2,S=+0->EMERGENCY_BRAKE; [46]: L=+1,T=+2,S=+1->EMERGENCY_BRAKE; [47]: L=+1,T=+2,S=+2->EMERGENCY_BRAKE */
    1, 1, 2, 1, /* [48]: L=+2,T=-1,S=-1->COAST; [49]: L=+2,T=-1,S=+0->COAST; [50]: L=+2,T=-1,S=+1->EMERGENCY_BRAKE; [51]: L=+2,T=-1,S=+2->COAST */
    0, 0, 1, 0, /* [52]: L=+2,T=+0,S=-1->ACCEL; [53]: L=+2,T=+0,S=+0->ACCEL; [54]: L=+2,T=+0,S=+1->COAST; [55]: L=+2,T=+0,S=+2->ACCEL */
    2, 2, 2, 2, /* [56]: L=+2,T=+1,S=-1->EMERGENCY_BRAKE; [57]: L=+2,T=+1,S=+0->EMERGENCY_BRAKE; [58]: L=+2,T=+1,S=+1->EMERGENCY_BRAKE; [59]: L=+2,T=+1,S=+2->EMERGENCY_BRAKE */
    1, 1, 2, 1, /* [60]: L=+2,T=+2,S=-1->COAST; [61]: L=+2,T=+2,S=+0->COAST; [62]: L=+2,T=+2,S=+1->EMERGENCY_BRAKE; [63]: L=+2,T=+2,S=+2->COAST */
};

HIRAM_ITCM_TEXT hiram_lut_result_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip) {
    hiram_lut_result_t result;
    result.reserved = 0;

    if ((lidar < -1 || lidar > 2) || (tof < -1 || tof > 2) || (slip < -1 || slip > 2)) {
        result.action = HIRAM_LUT_ACTION_EMERGENCY_BRAKE; /* Defensive boundary fallback */
        result.fault_flags = HIRAM_LUT_FAULT_INPUT_OOB;
        return result;
    }

    uint32_t idx = (uint32_t)(((lidar + 1) << 4) | ((tof + 1) << 2) | (slip + 1));
    result.action = HIRAM_POLICY_LUT[idx & 0x3FU];
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
    return hiram_crc32(HIRAM_POLICY_LUT, sizeof(HIRAM_POLICY_LUT)) == HIRAM_POLICY_LUT_CANONICAL_CRC32;
}
