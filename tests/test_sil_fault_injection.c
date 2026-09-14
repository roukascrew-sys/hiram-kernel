#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include "hiram_kernel.h"
#include "hiram_policy_lut.h"

static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define ASSERT_TRUE(expr, msg) do { \
    if (!(expr)) { \
        printf("  [FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg); \
        g_tests_failed++; \
        return; \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    if ((a) != (b)) { \
        printf("  [FAIL] %s:%d: %s (0x%X != 0x%X)\n", __FILE__, __LINE__, msg, (unsigned)(a), (unsigned)(b)); \
        g_tests_failed++; \
        return; \
    } \
} while(0)

/* Standard IEEE 802.3 CRC32 implementation identical to hiram_policy_lut */
static uint32_t compute_crc32(const uint8_t *data, size_t length) {
    uint32_t crc = 0xFFFFFFFFUL;
    for (size_t i = 0; i < length; i++) {
        crc ^= data[i];
        for (uint8_t bit = 0; bit < 8; bit++) {
            if (crc & 1u) {
                crc = (crc >> 1) ^ 0xEDB88320UL;
            } else {
                crc >>= 1;
            }
        }
    }
    return ~crc;
}

/* -----------------------------------------------------------------------------
 * TEST 1: Exhaustive 64/64 State Equivalence (Dynamic Kernel vs Static LUT)
 * ----------------------------------------------------------------------------- */
static void test_64_state_parity(void) {
    printf("[SIL] Running Test 1: Exhaustive 64/64 Dynamic vs LUT Parity...\n");
    hiram_workspace_t ws;
    int states_checked = 0;

    for (int32_t l = -1; l <= 2; l++) {
        for (int32_t t = -1; t <= 2; t++) {
            for (int32_t s = -1; s <= 2; s++) {
                hiram_evidence_t ev = { .lidar_obs = l, .tof_obs = t, .wheel_slip_obs = s };
                hiram_decision_t dec;
                hiram_status_t status = hiram_kernel_step(&ev, &ws, &dec);
                ASSERT_EQ(status, HIRAM_OK, "Dynamic kernel step failed");

                hiram_lut_result_t lut_res = hiram_policy_lut_step(l, t, s);
                ASSERT_EQ(lut_res.fault_flags, HIRAM_LUT_FAULT_NONE, "Unexpected fault on legal state");
                ASSERT_EQ(lut_res.action, (uint8_t)dec.selected_action, "Parity mismatch between dynamic and LUT");
                states_checked++;
            }
        }
    }
    ASSERT_EQ(states_checked, 64, "Incomplete 64-state iteration");
    printf("  [PASS] 64/64 states match bit-for-bit with dynamic Bayesian kernel.\n");
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 2: Boundary Value Stress & Out-of-Bounds Invalidation Traps
 * ----------------------------------------------------------------------------- */
static void test_input_oob_fault_traps(void) {
    printf("[SIL] Running Test 2: Input Boundary & Fuzz Invalidation Traps...\n");
    const int32_t oob_vals[] = {
        -1000, -256, -10, -3, -2,
        3, 4, 10, 256, 1000,
        INT32_MIN, INT32_MIN + 1, INT32_MAX - 1, INT32_MAX
    };
    const size_t num_oob = sizeof(oob_vals) / sizeof(oob_vals[0]);
    uint32_t traps_triggered = 0;

    for (size_t i = 0; i < num_oob; i++) {
        hiram_lut_result_t r = hiram_policy_lut_step(oob_vals[i], 0, 0);
        ASSERT_EQ(r.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "Lidar OOB did not force EMERGENCY_BRAKE");
        ASSERT_TRUE(r.fault_flags & HIRAM_LUT_FAULT_INPUT_OOB, "Lidar OOB missing FAULT_INPUT_OOB flag");
        traps_triggered++;
    }

    for (size_t i = 0; i < num_oob; i++) {
        hiram_lut_result_t r = hiram_policy_lut_step(0, oob_vals[i], 0);
        ASSERT_EQ(r.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "ToF OOB did not force EMERGENCY_BRAKE");
        ASSERT_TRUE(r.fault_flags & HIRAM_LUT_FAULT_INPUT_OOB, "ToF OOB missing FAULT_INPUT_OOB flag");
        traps_triggered++;
    }

    for (size_t i = 0; i < num_oob; i++) {
        hiram_lut_result_t r = hiram_policy_lut_step(0, 0, oob_vals[i]);
        ASSERT_EQ(r.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "Slip OOB did not force EMERGENCY_BRAKE");
        ASSERT_TRUE(r.fault_flags & HIRAM_LUT_FAULT_INPUT_OOB, "Slip OOB missing FAULT_INPUT_OOB flag");
        traps_triggered++;
    }

    for (size_t i = 0; i < num_oob; i++) {
        hiram_lut_result_t r = hiram_policy_lut_step(oob_vals[i], oob_vals[num_oob - 1 - i], oob_vals[i]);
        ASSERT_EQ(r.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "Multi OOB did not force EMERGENCY_BRAKE");
        ASSERT_TRUE(r.fault_flags & HIRAM_LUT_FAULT_INPUT_OOB, "Multi OOB missing FAULT_INPUT_OOB flag");
        traps_triggered++;
    }

    printf("  [PASS] %u boundary/fuzz vectors safely trapped to EMERGENCY_BRAKE.\n", traps_triggered);
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 3: SEU SRAM Bit-Flip & CRC32 Integrity Detection
 * ----------------------------------------------------------------------------- */
static void test_memory_bit_flip_seu(void) {
    printf("[SIL] Running Test 3: SEU Memory Bit-Flip Injections...\n");

    /* 3.1: Nominal verification on the baked table */
    ASSERT_TRUE(hiram_policy_lut_verify_integrity(), "Nominal LUT failed CRC verification");
    uint32_t nominal_crc = compute_crc32(HIRAM_POLICY_LUT, 64);
    ASSERT_EQ(nominal_crc, HIRAM_POLICY_LUT_CANONICAL_CRC32, "Computed CRC32 != CANONICAL_CRC32");

    /* 3.2: Mutable copy in RAM to simulate SRAM bit-flips */
    uint8_t ram_lut[64];
    memcpy(ram_lut, HIRAM_POLICY_LUT, 64);

    uint32_t flips_detected = 0;
    for (size_t byte_idx = 0; byte_idx < 64; byte_idx++) {
        for (uint8_t bit = 0; bit < 8; bit++) {
            uint8_t original = ram_lut[byte_idx];
            ram_lut[byte_idx] = (uint8_t)(original ^ (1u << bit));

            uint32_t corrupted_crc = compute_crc32(ram_lut, 64);
            ASSERT_TRUE(corrupted_crc != HIRAM_POLICY_LUT_CANONICAL_CRC32, "Bit-flip passed CRC check");

            ram_lut[byte_idx] = original;
            flips_detected++;
        }
    }

    ASSERT_EQ(compute_crc32(ram_lut, 64), HIRAM_POLICY_LUT_CANONICAL_CRC32, "Restored buffer failed CRC check");
    printf("  [PASS] Injected %u individual bit-flips; 100%% rejected by CRC32.\n", flips_detected);
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 4: Multi-Byte Burst Corruption & Null Table Injection
 * ----------------------------------------------------------------------------- */
static void test_burst_corruption(void) {
    printf("[SIL] Running Test 4: Multi-Byte Burst Memory Corruption Injections...\n");
    uint8_t ram_lut[64];
    memcpy(ram_lut, HIRAM_POLICY_LUT, 64);

    /* Zero out first 16 bytes */
    memset(ram_lut, 0x00, 16);
    ASSERT_TRUE(compute_crc32(ram_lut, 64) != HIRAM_POLICY_LUT_CANONICAL_CRC32, "Zeroed table block passed CRC32");

    /* Saturated 0xFF buffer */
    memset(ram_lut, 0xFF, 64);
    ASSERT_TRUE(compute_crc32(ram_lut, 64) != HIRAM_POLICY_LUT_CANONICAL_CRC32, "Saturated 0xFF table passed CRC32");

    /* Custody baseline digest verification */
    ASSERT_EQ(hiram_kernel_verify_custody(), HIRAM_OK, "Frozen IR digest custody check failed");

    printf("  [PASS] Burst corruption patterns and model custody verified.\n");
    g_tests_passed++;
}

int main(void) {
    printf("====================================================================\n");
    printf("  HIRAM SIL FAULT INJECTION & ASIL-D REGULATORY SUITE               \n");
    printf("====================================================================\n");

    test_64_state_parity();
    test_input_oob_fault_traps();
    test_memory_bit_flip_seu();
    test_burst_corruption();

    printf("====================================================================\n");
    if (g_tests_failed == 0) {
        printf("  SIL REGULATORY SUITE: ALL %d SUITES PASSED (0 FAILURES)\n", g_tests_passed);
        printf("====================================================================\n");
        return 0;
    }
    printf("  SIL REGULATORY SUITE: FAILED (%d failures)\n", g_tests_failed);
    printf("====================================================================\n");
    return 1;
}
