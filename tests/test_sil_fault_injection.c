/*
 * test_sil_fault_injection.c - HIRAM SIL fault injection & ASIL-D regulatory suite.
 *
 * The pre-audit version of this file proved that CRC32 detects corruption in a
 * *copy* of the policy LUT. It never corrupted the live table and then asked
 * the evaluator what it returned -- which is why a post-startup bit-flip that
 * turned EMERGENCY_BRAKE (2) into ACCEL (0) passed the whole suite with
 * fault_flags == 0. The tests below drive the production predicates directly
 * and, where the failure mode is a live-memory upset, mutate the live object.
 *
 * ORDERING IS PART OF THE TEST. Suite 1 runs before hiram_policy_lut_init() so
 * it can observe the pre-init guard, and suite 10 is last because a dual-rail
 * violation latches the LUT out of service permanently and by design -- there
 * is no un-latch entry point to call, and adding one would be the mock this
 * suite exists to avoid.
 */
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
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

/* Flips the IEEE-754 sign bit of `v` without type-punning through a pointer.
   This is the exact mutation the pre-audit review applied to
   HIRAM_LOSS_MATRIX[3]: +500.0 -> -500.0, a single bit. */
static double flip_sign_bit(double v) {
    uint64_t bits;
    double out;
    memcpy(&bits, &v, sizeof(bits));
    bits ^= 0x8000000000000000ULL;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

#define LOSS_MATRIX_LEN (HIRAM_N_ACTIONS * HIRAM_LOSS_ACTION_STRIDE)

/* -----------------------------------------------------------------------------
 * TEST 1: Pre-init guard -- the LUT must not be trusted before it is in service
 *
 * Runs before hiram_policy_lut_init(). At this point the table's contents are
 * whatever the loader left there, which on the target is whatever DTCM held
 * before Reset_Handler's copy. Returning an action from it would be a decision
 * made on unverified memory.
 * ----------------------------------------------------------------------------- */
static void test_step_before_init_is_fail_safe(void) {
    printf("[SIL] Running Test 1: Step-before-init guard...\n");

    ASSERT_TRUE(!hiram_policy_lut_is_operational(), "LUT reported operational before init");

    for (int32_t l = -1; l <= 2; l++) {
        for (int32_t t = -1; t <= 2; t++) {
            for (int32_t s = -1; s <= 2; s++) {
                hiram_lut_result_t r = hiram_policy_lut_step(l, t, s);
                ASSERT_EQ(r.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE,
                          "pre-init step did not force EMERGENCY_BRAKE");
                ASSERT_EQ(r.fault_flags, HIRAM_LUT_FAULT_INTEGRITY,
                          "pre-init step did not report FAULT_INTEGRITY");
            }
        }
    }

    printf("  [PASS] All 64 pre-init states fail safe to EMERGENCY_BRAKE + FAULT_INTEGRITY.\n");
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 2: Exhaustive 64/64 State Equivalence (Dynamic Kernel vs Static LUT)
 * ----------------------------------------------------------------------------- */
static void test_64_state_parity(void) {
    printf("[SIL] Running Test 2: Exhaustive 64/64 Dynamic vs LUT Parity...\n");
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
 * TEST 3: Boundary Value Stress & Out-of-Bounds Invalidation Traps
 * ----------------------------------------------------------------------------- */
static void test_input_oob_fault_traps(void) {
    printf("[SIL] Running Test 3: Input Boundary & Fuzz Invalidation Traps...\n");
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
 * TEST 4: SEU SRAM Bit-Flip & CRC32 Integrity Detection (startup check)
 * ----------------------------------------------------------------------------- */
static void test_memory_bit_flip_seu(void) {
    printf("[SIL] Running Test 4: SEU Memory Bit-Flip Injections (CRC32 layer)...\n");

    ASSERT_TRUE(hiram_policy_lut_verify_integrity(), "Nominal LUT failed CRC verification");
    uint32_t nominal_crc = compute_crc32(HIRAM_POLICY_LUT, 64);
    ASSERT_EQ(nominal_crc, HIRAM_POLICY_LUT_CANONICAL_CRC32, "Computed CRC32 != CANONICAL_CRC32");

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
 * TEST 5: Dual-rail decode -- exhaustive coverage of the per-call check (B1)
 *
 * CRC32 only runs at startup, so it is not the defence against a flip that
 * arrives later. The decode check in hiram_policy_lut_step() is, and this
 * sweeps the production predicate over its entire 256-value input domain and
 * over every single-bit mutation of every cell actually in the table.
 * ----------------------------------------------------------------------------- */
static void test_dual_rail_decode_coverage(void) {
    printf("[SIL] Running Test 5: Dual-Rail Decode Coverage (per-call layer)...\n");

    /* 5.1: exactly three of the 256 byte values may decode. */
    unsigned accepted = 0;
    for (unsigned v = 0; v < 256u; v++) {
        if (hiram_policy_lut_cell_is_valid((uint8_t)v)) {
            accepted++;
        }
    }
    ASSERT_EQ(accepted, 3u, "dual-rail decode accepts a number of values other than 3");

    for (unsigned a = 0; a < 3u; a++) {
        const uint8_t cell = HIRAM_LUT_ENCODE_CELL(a);
        ASSERT_TRUE(hiram_policy_lut_cell_is_valid(cell), "legal encoded cell rejected by its own decode");
        ASSERT_EQ((unsigned)(cell & 0x03u), a, "encoded cell does not decode back to its action");
    }

    /* 5.2: every single-bit mutation of every live cell must be rejected --
       including the EMERGENCY_BRAKE -> ACCEL mutation that motivated B1. */
    uint32_t mutations = 0;
    for (size_t i = 0; i < 64; i++) {
        const uint8_t cell = HIRAM_POLICY_LUT[i];
        ASSERT_TRUE(hiram_policy_lut_cell_is_valid(cell), "live table holds a cell that fails decode");
        for (uint8_t bit = 0; bit < 8; bit++) {
            const uint8_t mutated = (uint8_t)(cell ^ (1u << bit));
            ASSERT_TRUE(!hiram_policy_lut_cell_is_valid(mutated), "single-bit mutation survived dual-rail decode");
            mutations++;
        }
    }
    ASSERT_EQ(mutations, 512u, "incomplete single-bit mutation sweep");

    /* 5.3: the specific pre-audit mutation, spelled out. A bare action index
       put EMERGENCY_BRAKE (0b10) and ACCEL (0b00) one bit apart. The encoded
       forms are 0x06 and 0x0C, Hamming distance 2 -- and 2 is the minimum
       distance across all three codes, which is exactly the condition for
       every single-bit error to be detectable. */
    const uint8_t brake = HIRAM_LUT_ENCODE_CELL(HIRAM_LUT_ACTION_EMERGENCY_BRAKE);
    const uint8_t accel = HIRAM_LUT_ENCODE_CELL(HIRAM_LUT_ACTION_ACCEL);
    unsigned hamming = 0;
    for (uint8_t bit = 0; bit < 8; bit++) {
        if (((brake ^ accel) >> bit) & 1u) {
            hamming++;
        }
    }
    ASSERT_TRUE(hamming >= 2u, "EMERGENCY_BRAKE and ACCEL are within one bit-flip of each other");

    printf("  [PASS] 256/256 domain values classified; %u single-bit mutations rejected;\n", mutations);
    printf("         EMERGENCY_BRAKE<->ACCEL Hamming distance = %u bits.\n", hamming);
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 6: Multi-Byte Burst Corruption & Custody
 * ----------------------------------------------------------------------------- */
static void test_burst_corruption(void) {
    printf("[SIL] Running Test 6: Multi-Byte Burst Memory Corruption Injections...\n");
    uint8_t ram_lut[64];
    memcpy(ram_lut, HIRAM_POLICY_LUT, 64);

    memset(ram_lut, 0x00, 16);
    ASSERT_TRUE(compute_crc32(ram_lut, 64) != HIRAM_POLICY_LUT_CANONICAL_CRC32, "Zeroed table block passed CRC32");
    ASSERT_TRUE(!hiram_policy_lut_cell_is_valid(ram_lut[0]), "zeroed cell passed dual-rail decode");

    memset(ram_lut, 0xFF, 64);
    ASSERT_TRUE(compute_crc32(ram_lut, 64) != HIRAM_POLICY_LUT_CANONICAL_CRC32, "Saturated 0xFF table passed CRC32");
    ASSERT_TRUE(!hiram_policy_lut_cell_is_valid(ram_lut[0]), "saturated cell passed dual-rail decode");

    ASSERT_EQ(hiram_kernel_verify_custody(), HIRAM_OK, "Frozen IR digest custody check failed");

    printf("  [PASS] Burst corruption rejected by both layers; model custody verified.\n");
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 7: Loss matrix corruption (B2)
 *
 * Drives hiram_kernel_validate_loss_matrix() -- the same function
 * hiram_kernel_step() and hiram_kernel_decide() call before every decision --
 * over the sign flip the pre-audit review used, then over every entry and
 * every non-finite value.
 * ----------------------------------------------------------------------------- */
static void test_loss_matrix_sign_mutation(void) {
    printf("[SIL] Running Test 7: HIRAM_LOSS_MATRIX Corruption Injections...\n");

    double m[LOSS_MATRIX_LEN];
    memcpy(m, HIRAM_LOSS_MATRIX, sizeof(m));

    /* 7.0: the uncorrupted model must pass, or nothing below means anything. */
    ASSERT_EQ(hiram_kernel_validate_loss_matrix(HIRAM_LOSS_MATRIX), HIRAM_OK,
              "sealed HIRAM_LOSS_MATRIX rejected by its own validator");
    ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_OK, "faithful copy rejected");

    /* 7.1: the exact pre-audit mutation. HIRAM_LOSS_MATRIX[3] is the cost of
       ACCEL with an obstacle present and full decel capability: +500 -> -500
       turned the worst outcome into the cheapest one, and the kernel returned
       ACCEL with HIRAM_OK. */
    ASSERT_TRUE(HIRAM_LOSS_MATRIX[3] > 0.0, "fixture drift: HIRAM_LOSS_MATRIX[3] is not positive");
    m[3] = flip_sign_bit(HIRAM_LOSS_MATRIX[3]);
    ASSERT_TRUE(m[3] < 0.0, "sign-bit flip did not produce a negative loss");
    ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_ERR_MODEL_CORRUPTED,
              "sign-flipped HIRAM_LOSS_MATRIX[3] was not reported as MODEL_CORRUPTED");
    m[3] = HIRAM_LOSS_MATRIX[3];

    /* 7.2: every entry, every corruption class. */
    uint32_t injections = 0;
    for (int i = 0; i < LOSS_MATRIX_LEN; i++) {
        const double original = m[i];

        m[i] = flip_sign_bit(original);
        if (original != 0.0) {
            /* -0.0 is not < 0.0, so a sign flip of a zero entry is genuinely
               not a corruption of the cost ordering and must not be flagged. */
            ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_ERR_MODEL_CORRUPTED,
                      "sign-flipped entry not reported as MODEL_CORRUPTED");
            injections++;
        }

        m[i] = -1.0;
        ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_ERR_MODEL_CORRUPTED, "negative entry accepted");
        injections++;

        m[i] = NAN;
        ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_ERR_MODEL_CORRUPTED, "NaN entry accepted");
        injections++;

        m[i] = INFINITY;
        ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_ERR_MODEL_CORRUPTED, "+Inf entry accepted");
        injections++;

        m[i] = -INFINITY;
        ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_ERR_MODEL_CORRUPTED, "-Inf entry accepted");
        injections++;

        m[i] = original;
        ASSERT_EQ(hiram_kernel_validate_loss_matrix(m), HIRAM_OK, "restored matrix still rejected");
    }

    ASSERT_EQ(hiram_kernel_validate_loss_matrix(NULL), HIRAM_ERR_NULL_POINTER, "NULL matrix not rejected");

    printf("  [PASS] %u loss-matrix corruptions rejected as HIRAM_ERR_MODEL_CORRUPTED.\n", injections);
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 8: NaN / Inf / negative likelihood injection (M2)
 *
 * hiram_kernel_decide() is the decision half of hiram_kernel_step(), taking
 * the joint distribution as an argument -- so a corrupted CPT's effect can be
 * injected where it actually lands, in the joint, and the production guards
 * are the ones under test.
 * ----------------------------------------------------------------------------- */
static void test_nan_likelihood_injection(void) {
    printf("[SIL] Running Test 8: Non-Finite Likelihood Injections...\n");

    /* 8.0: a well-formed joint decides normally, so a NUMERICAL_FAULT below is
       attributable to the injection rather than to the entry point. */
    double joint[HIRAM_N_POSTERIOR];
    hiram_decision_t base;
    hiram_workspace_t ws;
    hiram_evidence_t ev = { .lidar_obs = 0, .tof_obs = 0, .wheel_slip_obs = 0 };
    hiram_decision_t via_step;

    ASSERT_EQ(hiram_kernel_step(&ev, &ws, &via_step), HIRAM_OK, "clear-road step failed");
    ASSERT_EQ(via_step.selected_action, HIRAM_ACTION_ACCEL, "clear-road step should select ACCEL");

    for (int i = 0; i < HIRAM_N_POSTERIOR; i++) {
        joint[i] = 0.1;
    }
    ASSERT_EQ(hiram_kernel_decide(joint, &base), HIRAM_OK, "uniform joint rejected");
    ASSERT_TRUE(!base.fallback_active, "uniform joint should not trip the fallback");

    /* 8.1: a NaN anywhere in the joint. Previously `NaN < guard` was false, so
       the zero-likelihood fallback did not fire, NaN/NaN propagated into every
       posterior, and the status came back HIRAM_OK with fallback_active false. */
    uint32_t injections = 0;
    const double poison[] = { NAN, INFINITY, -INFINITY, -1.0, -1e-30 };
    const size_t n_poison = sizeof(poison) / sizeof(poison[0]);

    for (int i = 0; i < HIRAM_N_POSTERIOR; i++) {
        for (size_t p = 0; p < n_poison; p++) {
            hiram_decision_t dec;
            /* Seed with a decision that is NOT EMERGENCY_BRAKE, so an
               untouched output cannot be mistaken for a correct one. */
            memcpy(&dec, &via_step, sizeof(dec));
            ASSERT_EQ(dec.selected_action, HIRAM_ACTION_ACCEL, "seed decision is not ACCEL");

            for (int k = 0; k < HIRAM_N_POSTERIOR; k++) {
                joint[k] = 0.1;
            }
            joint[i] = poison[p];

            const hiram_status_t rc = hiram_kernel_decide(joint, &dec);
            ASSERT_EQ(rc, HIRAM_ERR_NUMERICAL_FAULT, "poisoned joint not reported as NUMERICAL_FAULT");
            ASSERT_EQ(dec.status, HIRAM_ERR_NUMERICAL_FAULT, "out->status does not mirror the return value");
            ASSERT_EQ(dec.selected_action, HIRAM_ACTION_EMERGENCY_BRAKE, "poisoned joint did not force EMERGENCY_BRAKE");
            ASSERT_EQ(dec.selected_action_index, HIRAM_EMERGENCY_BRAKE_INDEX, "action index not EMERGENCY_BRAKE");
            ASSERT_TRUE(dec.fallback_active, "poisoned joint did not set fallback_active");
            injections++;
        }
    }

    /* 8.2: an overflow that no single finite term reveals. */
    hiram_decision_t dec;
    memcpy(&dec, &via_step, sizeof(dec));
    for (int k = 0; k < HIRAM_N_POSTERIOR; k++) {
        joint[k] = 1.5e308; /* finite individually; the sum is +Inf */
    }
    ASSERT_EQ(hiram_kernel_decide(joint, &dec), HIRAM_ERR_NUMERICAL_FAULT, "summation overflow not caught");
    ASSERT_EQ(dec.selected_action, HIRAM_ACTION_EMERGENCY_BRAKE, "summation overflow did not force EMERGENCY_BRAKE");
    ASSERT_TRUE(dec.fallback_active, "summation overflow did not set fallback_active");
    injections++;

    /* 8.3: NULL joint with a live output struct -- same defensive contract. */
    memcpy(&dec, &via_step, sizeof(dec));
    ASSERT_EQ(hiram_kernel_decide(NULL, &dec), HIRAM_ERR_NULL_POINTER, "NULL joint not rejected");
    ASSERT_EQ(dec.selected_action, HIRAM_ACTION_EMERGENCY_BRAKE, "NULL joint left a stale action");
    ASSERT_TRUE(dec.fallback_active, "NULL joint did not set fallback_active");
    ASSERT_EQ(hiram_kernel_decide(joint, NULL), HIRAM_ERR_NULL_POINTER, "NULL output not rejected");

    printf("  [PASS] %u non-finite/negative likelihood injections trapped as\n", injections);
    printf("         HIRAM_ERR_NUMERICAL_FAULT with fallback_active and EMERGENCY_BRAKE.\n");
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 9: Defensive NULL contract (M2)
 *
 * A caller that reuses one hiram_decision_t across frames and passes a NULL
 * workspace must not be left reading the previous frame's ACCEL back out of a
 * struct whose status field still says HIRAM_OK.
 * ----------------------------------------------------------------------------- */
static void test_defensive_null_workspace(void) {
    printf("[SIL] Running Test 9: Defensive NULL Argument Contract...\n");

    hiram_workspace_t ws;
    hiram_decision_t out;
    const hiram_evidence_t clear_road = { .lidar_obs = 0, .tof_obs = 0, .wheel_slip_obs = 0 };

    /* Establish a stale, non-conservative decision in `out`. */
    ASSERT_EQ(hiram_kernel_step(&clear_road, &ws, &out), HIRAM_OK, "priming step failed");
    ASSERT_EQ(out.selected_action, HIRAM_ACTION_ACCEL, "priming step should select ACCEL");
    ASSERT_EQ(out.status, HIRAM_OK, "priming step status should be HIRAM_OK");
    ASSERT_TRUE(!out.fallback_active, "priming step should not be a fallback");

    /* NULL workspace, live output. */
    ASSERT_EQ(hiram_kernel_step(&clear_road, NULL, &out), HIRAM_ERR_NULL_POINTER,
              "NULL workspace not rejected");
    ASSERT_EQ(out.selected_action, HIRAM_ACTION_EMERGENCY_BRAKE,
              "NULL workspace left the stale ACCEL in *out");
    ASSERT_EQ(out.selected_action_index, HIRAM_EMERGENCY_BRAKE_INDEX,
              "NULL workspace left a stale selected_action_index");
    ASSERT_EQ(out.status, HIRAM_ERR_NULL_POINTER, "out->status does not mirror the return value");
    ASSERT_TRUE(out.fallback_active, "NULL workspace did not set fallback_active");

    /* NULL evidence, live output -- same contract. */
    ASSERT_EQ(hiram_kernel_step(&clear_road, &ws, &out), HIRAM_OK, "re-priming step failed");
    ASSERT_EQ(out.selected_action, HIRAM_ACTION_ACCEL, "re-priming step should select ACCEL");
    ASSERT_EQ(hiram_kernel_step(NULL, &ws, &out), HIRAM_ERR_NULL_POINTER, "NULL evidence not rejected");
    ASSERT_EQ(out.selected_action, HIRAM_ACTION_EMERGENCY_BRAKE, "NULL evidence left the stale ACCEL in *out");
    ASSERT_EQ(out.status, HIRAM_ERR_NULL_POINTER, "out->status does not mirror the return value");
    ASSERT_TRUE(out.fallback_active, "NULL evidence did not set fallback_active");

    /* The posterior must be the model's true prior, not the previous frame's. */
    for (int i = 0; i < HIRAM_N_POSTERIOR; i++) {
        ASSERT_TRUE(out.posterior[i] == HIRAM_PRIOR_MARGINAL[i],
                    "NULL-argument fallback did not publish the model prior");
    }

    /* NULL output is the one case with nothing to write to. */
    ASSERT_EQ(hiram_kernel_step(&clear_road, &ws, NULL), HIRAM_ERR_NULL_POINTER, "NULL output not rejected");

    printf("  [PASS] NULL evidence/workspace clear the stale decision to EMERGENCY_BRAKE.\n");
    g_tests_passed++;
}

/* -----------------------------------------------------------------------------
 * TEST 10: Live post-startup bit-flip (B1) -- DESTRUCTIVE, RUNS LAST
 *
 * The test the pre-audit suite did not have. Everything above injected into a
 * copy; this mutates HIRAM_POLICY_LUT itself, after startup and after the CRC32
 * has already passed, and then asks hiram_policy_lut_step() what it returns for
 * the affected state.
 *
 * On the STM32H723 the table is DTCM RAM (stm32h723zg.ld .dtcm_lut; Reset_Handler
 * copies it from Flash), which is why an SEU can reach it at all and why the
 * symbol is not const-qualified. On the host it is ordinary .data, so the same
 * injection is expressible here.
 *
 * The violation latches the table out of service permanently, by design -- so
 * no suite that needs a working LUT may run after this one.
 * ----------------------------------------------------------------------------- */
static void test_live_lut_bit_flip_latches(void) {
    printf("[SIL] Running Test 10: Live Post-Startup Bit-Flip (destructive)...\n");

    /* Index 40 is L=+1,T=+1,S=-1 -> EMERGENCY_BRAKE: obstacle on both range
       sensors. Under the old bare-index encoding, clearing bit 1 of this cell
       turned it into ACCEL and the evaluator reported fault_flags == 0. */
    const size_t victim = 40;

    ASSERT_TRUE(hiram_policy_lut_is_operational(), "LUT not in service at the start of the destructive test");
    ASSERT_TRUE(hiram_policy_lut_verify_integrity(), "LUT failed CRC before injection");

    hiram_lut_result_t before = hiram_policy_lut_step(1, 1, -1);
    ASSERT_EQ(before.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "victim state is not EMERGENCY_BRAKE");
    ASSERT_EQ(before.fault_flags, HIRAM_LUT_FAULT_NONE, "victim state faulted before injection");

    const uint8_t original = HIRAM_POLICY_LUT[victim];

    /* Flip bit 1 -- the single bit that used to change EMERGENCY_BRAKE into
       ACCEL. Post-startup: the CRC32 above has already run and passed. */
    HIRAM_POLICY_LUT[victim] = (uint8_t)(original ^ 0x02u);

    hiram_lut_result_t after = hiram_policy_lut_step(1, 1, -1);
    ASSERT_EQ(after.fault_flags, HIRAM_LUT_FAULT_INTEGRITY, "live bit-flip did not raise FAULT_INTEGRITY");
    ASSERT_EQ(after.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "live bit-flip did not force EMERGENCY_BRAKE");
    ASSERT_TRUE(!hiram_policy_lut_is_operational(), "live bit-flip did not take the LUT out of service");

    /* Every other state is now refused too: one observed upset condemns the
       table, not just the cell it landed in. */
    for (int32_t l = -1; l <= 2; l++) {
        for (int32_t t = -1; t <= 2; t++) {
            for (int32_t s = -1; s <= 2; s++) {
                hiram_lut_result_t r = hiram_policy_lut_step(l, t, s);
                ASSERT_EQ(r.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "latched LUT returned a trusted action");
                ASSERT_EQ(r.fault_flags, HIRAM_LUT_FAULT_INTEGRITY, "latched LUT did not report FAULT_INTEGRITY");
            }
        }
    }

    /* Repairing the memory must not re-arm the table. A transient that clears
       on re-read is evidence the memory is unreliable, not evidence the table
       is sound; only a reset may clear the latch. */
    HIRAM_POLICY_LUT[victim] = original;
    ASSERT_TRUE(hiram_policy_lut_verify_integrity(), "restored table failed CRC");
    ASSERT_TRUE(!hiram_policy_lut_init(), "init re-armed a table that had been observed corrupt");
    ASSERT_TRUE(!hiram_policy_lut_is_operational(), "latch cleared without a reset");

    hiram_lut_result_t post_repair = hiram_policy_lut_step(1, 1, -1);
    ASSERT_EQ(post_repair.fault_flags, HIRAM_LUT_FAULT_INTEGRITY, "repaired-but-latched LUT stopped reporting the fault");
    ASSERT_EQ(post_repair.action, HIRAM_LUT_ACTION_EMERGENCY_BRAKE, "repaired-but-latched LUT returned a trusted action");

    printf("  [PASS] Post-CRC bit-flip at index %u trapped on the very next call;\n", (unsigned)victim);
    printf("         table latched out of service and not re-armed by repair.\n");
    g_tests_passed++;
}

int main(void) {
    printf("====================================================================\n");
    printf("  HIRAM SIL FAULT INJECTION & ASIL-D REGULATORY SUITE               \n");
    printf("====================================================================\n");

    /* Before init, deliberately. */
    test_step_before_init_is_fail_safe();

    if (!hiram_policy_lut_init()) {
        printf("  [FATAL] hiram_policy_lut_init() failed on the sealed table.\n");
        return 1;
    }
    printf("[SIL] Policy LUT initialised: CRC32 + 64-cell dual-rail sweep passed.\n");

    test_64_state_parity();
    test_input_oob_fault_traps();
    test_memory_bit_flip_seu();
    test_dual_rail_decode_coverage();
    test_burst_corruption();
    test_loss_matrix_sign_mutation();
    test_nan_likelihood_injection();
    test_defensive_null_workspace();

    /* Destructive: latches the LUT out of service. Must stay last. */
    test_live_lut_bit_flip_latches();

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
