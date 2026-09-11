/**
 * hiram_policy_lut.h - Compiled 64-State Policy-to-LUT Evaluator
 *
 * Automatically synthesized from Bayesian C-Kernel decision policy.
 * Exact action selection over all 64 tri-state permutations.
 * Execution cost: 1 memory access / array lookup (~2 to 4 cycles on ARM Cortex-M7).
 */

#ifndef HIRAM_POLICY_LUT_H
#define HIRAM_POLICY_LUT_H

#include <stdint.h>
#include <stdbool.h>

/* Action Definitions matching include/hiram_tables.h */
#define HIRAM_LUT_ACTION_ACCEL           0
#define HIRAM_LUT_ACTION_COAST           1
#define HIRAM_LUT_ACTION_EMERGENCY_BRAKE 2

/* 64-byte pre-computed optimal Bayesian decision policy */
static const uint8_t HIRAM_POLICY_LUT[64] = {
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

/**
 * Evaluates the 64-state policy LUT.
 * Returns action index: 0=ACCEL, 1=COAST, 2=EMERGENCY_BRAKE.
 * Out-of-bounds inputs return HIRAM_LUT_ACTION_EMERGENCY_BRAKE (fail-safe).
 */
static inline uint8_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip) {
    if ((lidar < -1 || lidar > 2) || (tof < -1 || tof > 2) || (slip < -1 || slip > 2)) {
        return HIRAM_LUT_ACTION_EMERGENCY_BRAKE; /* Defensive boundary fallback */
    }
    uint32_t idx = (uint32_t)(((lidar + 1) << 4) | ((tof + 1) << 2) | (slip + 1));
    return HIRAM_POLICY_LUT[idx & 0x3FU];
}

#endif /* HIRAM_POLICY_LUT_H */
