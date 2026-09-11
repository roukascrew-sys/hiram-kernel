/*
 * HIRAM Safety Kernel - Task 3.1: C kernel unit tests.
 * Self-contained (no external test framework, matching the kernel's own
 * zero-dependency target). Validates:
 *   1. hiram_kernel_verify_custody() succeeds against the frozen digest.
 *   2. Parity against tools/inference_engine.py oracle scenarios: exact
 *      match on selected_action/fallback_active, tight-tolerance match on
 *      posterior/expected_losses (see doubles_nearly_equal for why exact
 *      summation-order equality isn't the right bar here).
 *   3. All 64 combinations of {0, 1, 2 (DROPOUT), HIRAM_UNOBSERVED} across
 *      the three sensors: a sensor reporting 2 must produce a bit-identical
 *      decision to the same evidence with that sensor's 2 replaced by
 *      HIRAM_UNOBSERVED (dropout === unobserved, tri-state equivalence).
 *   4. Basic input validation (NULL pointers, out-of-range evidence).
 */
#include "hiram_kernel.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int g_checks_run = 0;
static int g_checks_failed = 0;

#define HIRAM_CHECK(cond, msg)                                                                   \
    do {                                                                                         \
        g_checks_run++;                                                                          \
        if (!(cond)) {                                                                           \
            g_checks_failed++;                                                                   \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));                      \
        }                                                                                         \
    } while (0)

static int doubles_bit_equal(double a, double b) {
    /* Bit-exact comparison (not tolerance-based): every value in this
     * kernel traces back to a hex-float literal generated verbatim from
     * the Python reference engine's own float.hex() output, so a correct
     * port must reproduce the exact same IEEE-754 bit pattern, not merely
     * something "close". */
    return memcmp(&a, &b, sizeof(double)) == 0;
}

/*
 * Relative+absolute tolerance for comparing this kernel's computed values
 * against tools/inference_engine.py oracle output. Not bit-exactness: the
 * Python reference engine sums the joint likelihood and the expected-loss
 * dot product with math.fsum (a correctly-rounded exact-summation
 * algorithm), while this C kernel intentionally uses plain left-to-right
 * double accumulation (Shewchuk/fsum-style exact summation buys no
 * decision-relevant precision for 6-term sums of similarly-scaled
 * non-negative doubles on a Cortex-M7, at the cost of real code/cycle
 * budget on a safety-critical embedded target). The two summation orders
 * can differ by a few ULP, which this tolerance absorbs -- it is still
 * ~1e6x tighter than the smallest loss-value gap that could ever flip a
 * decision, so it cannot mask an actual factor-contraction bug. The CPT
 * and loss table *inputs* to that arithmetic remain bit-exact (see
 * doubles_bit_equal / include/hiram_tables.h's hex-float literals) --
 * only the runtime summation result is compared with tolerance.
 */
static int doubles_nearly_equal(double a, double b) {
    const double tol = 1e-9 * (1.0 > fabs(a) ? (1.0 > fabs(b) ? 1.0 : fabs(b)) : fabs(a));
    return fabs(a - b) <= tol;
}

static void expect_decision_matches_oracle(const hiram_decision_t *got, const double *posterior_expected,
                                            const double *losses_expected, int32_t action_expected,
                                            bool fallback_expected, const char *label) {
    int i;
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        HIRAM_CHECK(doubles_nearly_equal(got->posterior[i], posterior_expected[i]), label);
    }
    for (i = 0; i < HIRAM_N_ACTIONS; i++) {
        HIRAM_CHECK(doubles_nearly_equal(got->expected_losses[i], losses_expected[i]), label);
    }
    HIRAM_CHECK(got->selected_action == action_expected, label);
    HIRAM_CHECK(got->fallback_active == fallback_expected, label);
}

/* ------------------------------------------------------------------ */
/* 1. Custody                                                          */
/* ------------------------------------------------------------------ */
static void test_verify_custody(void) {
    HIRAM_CHECK(hiram_kernel_verify_custody() == HIRAM_OK, "verify_custody should succeed against the frozen digest");
}

/* ------------------------------------------------------------------ */
/* 2. Parity against tools/inference_engine.py oracle scenarios        */
/*    (values captured from BayesianInferenceEngine().step(evidence);  */
/*    see the .hex() reproductions in the comments below.)             */
/* ------------------------------------------------------------------ */
static void test_parity_full_clear(void) {
    /* evidence={"LidarObs":0,"TofObs":0,"WheelSlipObs":0} -> ACCEL */
    const hiram_evidence_t ev = {0, 0, 0};
    hiram_workspace_t ws;
    hiram_decision_t out;
    HIRAM_CHECK(hiram_kernel_step(&ev, &ws, &out) == HIRAM_OK, "full_clear step should succeed");

    const double posterior[HIRAM_N_POSTERIOR] = {
        0x1.cfb5bd4428704p-1, 0x1.6b7420d51c28bp-4, 0x1.6a6f801cf0569p-8,
        0x1.6e180cd702b11p-15, 0x1.0f941c69e5cf5p-17, 0x1.7acaa6411f804p-21,
    };
    const double losses[HIRAM_N_ACTIONS] = {
        0x1.009eb667b3b1bp-5, 0x1.00eda0ddf81c7p+1, 0x1.8ffc54d3a0942p+4,
    };
    expect_decision_matches_oracle(&out, posterior, losses, HIRAM_ACTION_ACCEL, false, "full_clear parity");
}

static void test_parity_full_detected(void) {
    /* evidence={"LidarObs":1,"TofObs":1,"WheelSlipObs":0} -> EMERGENCY_BRAKE */
    const hiram_evidence_t ev = {1, 1, 0};
    hiram_workspace_t ws;
    hiram_decision_t out;
    HIRAM_CHECK(hiram_kernel_step(&ev, &ws, &out) == HIRAM_OK, "full_detected step should succeed");

    const double posterior[HIRAM_N_POSTERIOR] = {
        0x1.b6da17b6338c7p-9, 0x1.57f8551061aeep-12, 0x1.5701ad0703789p-16,
        0x1.a883d5df2ff04p-1, 0x1.3aeab94fc88bep-3, 0x1.b73d29cae7be0p-7,
    };
    const double losses[HIRAM_N_ACTIONS] = {
        0x1.29923b827610fp+9, 0x1.178c896e42303p+7, 0x1.fef438e7f9223p+2,
    };
    expect_decision_matches_oracle(&out, posterior, losses, HIRAM_ACTION_EMERGENCY_BRAKE, false, "full_detected parity");
}

static void test_parity_full_detected_high_slip(void) {
    /* evidence={"LidarObs":1,"TofObs":1,"WheelSlipObs":1} -> EMERGENCY_BRAKE */
    const hiram_evidence_t ev = {1, 1, 1};
    hiram_workspace_t ws;
    hiram_decision_t out;
    HIRAM_CHECK(hiram_kernel_step(&ev, &ws, &out) == HIRAM_OK, "full_detected_high_slip step should succeed");

    const double posterior[HIRAM_N_POSTERIOR] = {
        0x1.d84aa8fc4e985p-13, 0x1.37601482e3d7ap-10, 0x1.3438520ea9389p-11,
        0x1.c8dcbc6ad63a6p-5, 0x1.1d13432a96bc7p-1, 0x1.8ab15c8acdfdap-2,
    };
    const double losses[HIRAM_N_ACTIONS] = {
        0x1.52e3a03b9383bp+10, 0x1.e0f891b6fc72ep+8, 0x1.ebc9a64b648b6p+4,
    };
    expect_decision_matches_oracle(&out, posterior, losses, HIRAM_ACTION_EMERGENCY_BRAKE, false,
                               "full_detected_high_slip parity");
}

static void test_parity_no_evidence(void) {
    /* evidence={} -> COAST (the true physical prior; equals HIRAM_PRIOR_MARGINAL) */
    const hiram_evidence_t ev = {HIRAM_UNOBSERVED, HIRAM_UNOBSERVED, HIRAM_UNOBSERVED};
    hiram_workspace_t ws;
    hiram_decision_t out;
    HIRAM_CHECK(hiram_kernel_step(&ev, &ws, &out) == HIRAM_OK, "no_evidence step should succeed");

    const double posterior[HIRAM_N_POSTERIOR] = {
        0x1.4dd97f62b6ae8p-1, 0x1.9e4f765fd8adbp-3, 0x1.35dcc63f14121p-4,
        0x1.2eb1c432ca57bp-5, 0x1.63886594af4f2p-6, 0x1.73eab367a0f90p-7,
    };
    const double losses[HIRAM_N_ACTIONS] = {
        0x1.f700000000000p+5, 0x1.5251eb851eb85p+4, 0x1.86fae147ae148p+4,
    };
    expect_decision_matches_oracle(&out, posterior, losses, HIRAM_ACTION_COAST, false, "no_evidence parity");

    /* HIRAM_PRIOR_MARGINAL is itself just this same no-evidence posterior,
     * precomputed once at export time via Python's math.fsum -- confirm the
     * two never drift (tolerance, not bit-exact, for the same summation-
     * order reason as expect_decision_matches_oracle above). */
    int i;
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        HIRAM_CHECK(doubles_nearly_equal(out.posterior[i], HIRAM_PRIOR_MARGINAL[i]),
                    "no_evidence posterior should equal HIRAM_PRIOR_MARGINAL");
    }
}

static void test_parity_lidar_only_detected(void) {
    /* evidence={"LidarObs":1} (TofObs, WheelSlipObs unobserved) -> EMERGENCY_BRAKE */
    const hiram_evidence_t ev = {1, HIRAM_UNOBSERVED, HIRAM_UNOBSERVED};
    hiram_workspace_t ws;
    hiram_decision_t out;
    HIRAM_CHECK(hiram_kernel_step(&ev, &ws, &out) == HIRAM_OK, "lidar_only_detected step should succeed");

    const double posterior[HIRAM_N_POSTERIOR] = {
        0x1.60590d8947e57p-4, 0x1.b5445467dbe75p-6, 0x1.4708141a0bb9ep-7,
        0x1.da3539f461ce9p-2, 0x1.167e2d1eefa77p-2, 0x1.23539f461ce8ep-3,
    };
    const double losses[HIRAM_N_ACTIONS] = {
        0x1.8a016f7381036p+9, 0x1.e3e34af9ebbc7p+7, 0x1.1ef3d778e85bep+4,
    };
    expect_decision_matches_oracle(&out, posterior, losses, HIRAM_ACTION_EMERGENCY_BRAKE, false,
                               "lidar_only_detected parity");
}

/* ------------------------------------------------------------------ */
/* 3. All 64 dropout(2)-vs-HIRAM_UNOBSERVED(-1) combinations           */
/* ------------------------------------------------------------------ */
static int32_t raw_evidence_codes[4] = {0, 1, 2, HIRAM_UNOBSERVED};

static void test_all_64_dropout_combinations_match_unobserved(void) {
    int li, ti, wi;
    for (li = 0; li < 4; li++) {
        for (ti = 0; ti < 4; ti++) {
            for (wi = 0; wi < 4; wi++) {
                const hiram_evidence_t raw = {
                    raw_evidence_codes[li], raw_evidence_codes[ti], raw_evidence_codes[wi],
                };
                /* Canonical form: every occurrence of the DROPOUT index (2)
                 * replaced with HIRAM_UNOBSERVED. Per the tri-state
                 * contract, this must produce a bit-identical decision. */
                const hiram_evidence_t canonical = {
                    raw.lidar_obs == 2 ? HIRAM_UNOBSERVED : raw.lidar_obs,
                    raw.tof_obs == 2 ? HIRAM_UNOBSERVED : raw.tof_obs,
                    raw.wheel_slip_obs == 2 ? HIRAM_UNOBSERVED : raw.wheel_slip_obs,
                };

                hiram_workspace_t ws_raw, ws_canonical;
                hiram_decision_t out_raw, out_canonical;
                HIRAM_CHECK(hiram_kernel_step(&raw, &ws_raw, &out_raw) == HIRAM_OK,
                            "raw evidence step should succeed for all 64 combinations");
                HIRAM_CHECK(hiram_kernel_step(&canonical, &ws_canonical, &out_canonical) == HIRAM_OK,
                            "canonical evidence step should succeed for all 64 combinations");

                int i;
                for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
                    HIRAM_CHECK(doubles_bit_equal(out_raw.posterior[i], out_canonical.posterior[i]),
                                "dropout(2) vs unobserved(-1) posterior must be bit-identical");
                }
                for (i = 0; i < HIRAM_N_ACTIONS; i++) {
                    HIRAM_CHECK(doubles_bit_equal(out_raw.expected_losses[i], out_canonical.expected_losses[i]),
                                "dropout(2) vs unobserved(-1) expected_losses must be bit-identical");
                }
                HIRAM_CHECK(out_raw.selected_action == out_canonical.selected_action,
                            "dropout(2) vs unobserved(-1) selected_action must match");
                HIRAM_CHECK(out_raw.fallback_active == out_canonical.fallback_active,
                            "dropout(2) vs unobserved(-1) fallback_active must match");
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* 4. Input validation                                                 */
/* ------------------------------------------------------------------ */
static void test_null_pointer_rejected(void) {
    const hiram_evidence_t ev = {0, 0, 0};
    hiram_workspace_t ws;
    hiram_decision_t out;
    HIRAM_CHECK(hiram_kernel_step(NULL, &ws, &out) == HIRAM_ERR_NULL_POINTER, "NULL evidence should be rejected");
    HIRAM_CHECK(hiram_kernel_step(&ev, NULL, &out) == HIRAM_ERR_NULL_POINTER, "NULL workspace should be rejected");
    HIRAM_CHECK(hiram_kernel_step(&ev, &ws, NULL) == HIRAM_ERR_NULL_POINTER, "NULL output should be rejected");
}

static void test_out_of_range_evidence_rejected(void) {
    hiram_workspace_t ws;
    hiram_decision_t out;

    const hiram_evidence_t bad_lidar = {3, 0, 0};
    HIRAM_CHECK(hiram_kernel_step(&bad_lidar, &ws, &out) == HIRAM_ERR_INVALID_EVIDENCE,
                "lidar_obs=3 (out of range) should be rejected");

    const hiram_evidence_t bad_tof = {0, -2, 0};
    HIRAM_CHECK(hiram_kernel_step(&bad_tof, &ws, &out) == HIRAM_ERR_INVALID_EVIDENCE,
                "tof_obs=-2 (not HIRAM_UNOBSERVED, not in range) should be rejected");

    const hiram_evidence_t bad_wheel = {0, 0, 100};
    HIRAM_CHECK(hiram_kernel_step(&bad_wheel, &ws, &out) == HIRAM_ERR_INVALID_EVIDENCE,
                "wheel_slip_obs=100 (out of range) should be rejected");
}

int main(void) {
    test_verify_custody();
    test_parity_full_clear();
    test_parity_full_detected();
    test_parity_full_detected_high_slip();
    test_parity_no_evidence();
    test_parity_lidar_only_detected();
    test_all_64_dropout_combinations_match_unobserved();
    test_null_pointer_rejected();
    test_out_of_range_evidence_rejected();

    printf("hiram_kernel tests: %d checks run, %d failed\n", g_checks_run, g_checks_failed);
    return g_checks_failed == 0 ? 0 : 1;
}
