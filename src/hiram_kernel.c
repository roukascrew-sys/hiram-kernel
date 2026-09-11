/*
 * HIRAM Safety Kernel - Task 3.1: Flat-Memory C Kernel
 * Implements the contract declared in include/hiram_kernel.h by porting
 * tools/inference_engine.py's BayesianInferenceEngine._compute_posterior_
 * with_diagnostics / evaluate_expected_loss / step() to flat arithmetic
 * over the tables in include/hiram_tables.h. See that Python module's
 * docstring for the mathematical justification; this file mirrors its
 * structure line-for-line so the two stay easy to compare.
 *
 * No dynamic memory: this translation unit must never reference malloc,
 * free, calloc, realloc, or alloca. The #pragma below turns any accidental
 * use of one of those identifiers into a hard compile error rather than
 * relying on code review or a post-build symbol scan alone (the Makefile's
 * verify_zero_alloc target additionally checks the compiled object, as a
 * second, independent line of defense against anything pulled in via
 * linking rather than written directly in this file).
 */
#include "hiram_kernel.h"

#include <math.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC poison malloc free calloc realloc alloca
#endif

/* Mirrors tools.inference_engine._ZERO_LIKELIHOOD_GUARD exactly. */
#define HIRAM_ZERO_LIKELIHOOD_GUARD 1e-15

static bool hiram_evidence_value_in_range(int32_t v) {
    return v == HIRAM_UNOBSERVED || (v >= 0 && v < HIRAM_N_SENSOR_STATES);
}

/*
 * Mirrors BayesianInferenceEngine._extract_evidence_value's
 * marginalize_dropout_state=True (default) behavior: an explicit
 * HIRAM_UNOBSERVED *or* the sensor's own DROPOUT state index both collapse
 * to "not observed" -- returned as HIRAM_UNOBSERVED so the rest of the
 * pipeline has exactly one "unobserved" representation to branch on.
 */
static int32_t hiram_canonicalize_evidence(int32_t v, int32_t dropout_index) {
    if (v == dropout_index) {
        return HIRAM_UNOBSERVED;
    }
    return v;
}

hiram_status_t hiram_kernel_verify_custody(void) {
    if (strcmp(HIRAM_FROZEN_IR_DIGEST, HIRAM_KERNEL_EXPECTED_IR_DIGEST) != 0) {
        return HIRAM_ERR_CUSTODY_MISMATCH;
    }
    return HIRAM_OK;
}

hiram_status_t hiram_kernel_step(const hiram_evidence_t *evidence, hiram_workspace_t *ws, hiram_decision_t *out) {
    if (evidence == NULL || ws == NULL || out == NULL) {
        return HIRAM_ERR_NULL_POINTER;
    }
    if (!hiram_evidence_value_in_range(evidence->lidar_obs) ||
        !hiram_evidence_value_in_range(evidence->tof_obs) ||
        !hiram_evidence_value_in_range(evidence->wheel_slip_obs)) {
        return HIRAM_ERR_INVALID_EVIDENCE;
    }

    const int32_t lidar_val = hiram_canonicalize_evidence(evidence->lidar_obs, HIRAM_LIDAROBS_DROPOUT_INDEX);
    const int32_t tof_val = hiram_canonicalize_evidence(evidence->tof_obs, HIRAM_TOFOBS_DROPOUT_INDEX);
    const int32_t wheel_val = hiram_canonicalize_evidence(evidence->wheel_slip_obs, HIRAM_WHEELSLIPOBS_DROPOUT_INDEX);

    int32_t i;
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        ws->unnormalized_joint[i] = 0.0;
    }

    /* Direct marginalization over TrackCondition, mirroring
     * _compute_posterior_with_diagnostics's t/o/d nested loop exactly:
     * index order, stride usage, and per-sensor "unobserved -> factor 1.0"
     * semantics are all identical to the Python reference. */
    int32_t t, o, d;
    for (t = 0; t < HIRAM_N_TRACK; t++) {
        const double p_t = HIRAM_CPT_TRACKCONDITION[t];

        for (o = 0; o < HIRAM_N_OBSTACLE; o++) {
            const double p_o_given_t = HIRAM_CPT_TRUEOBSTACLE[t * HIRAM_STRIDE_TRACK_TO_OBSTACLE + o];

            const double f_lidar = (lidar_val == HIRAM_UNOBSERVED)
                ? 1.0
                : HIRAM_CPT_LIDAROBS[o * HIRAM_STRIDE_OBSTACLE_TO_LIDAR + lidar_val];
            const double f_tof = (tof_val == HIRAM_UNOBSERVED)
                ? 1.0
                : HIRAM_CPT_TOFOBS[o * HIRAM_STRIDE_OBSTACLE_TO_TOF + tof_val];
            const double obstacle_evidence_factor = f_lidar * f_tof;

            for (d = 0; d < HIRAM_N_DECEL; d++) {
                const double p_d_given_t = HIRAM_CPT_DECELCAPABILITY[t * HIRAM_STRIDE_TRACK_TO_DECEL + d];

                const double f_wheel = (wheel_val == HIRAM_UNOBSERVED)
                    ? 1.0
                    : HIRAM_CPT_WHEELSLIPOBS[d * HIRAM_STRIDE_DECEL_TO_WHEEL + wheel_val];

                ws->unnormalized_joint[o * HIRAM_N_DECEL + d] +=
                    p_t * p_o_given_t * p_d_given_t * obstacle_evidence_factor * f_wheel;
            }
        }
    }

    double likelihood = 0.0;
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        likelihood += ws->unnormalized_joint[i];
    }

    const bool fallback_active = likelihood < HIRAM_ZERO_LIKELIHOOD_GUARD;
    if (fallback_active) {
        /* Degenerate/contradictory evidence: revert to the model's true
         * physical prior rather than dividing by (near) zero, matching
         * _compute_posterior_with_diagnostics's fallback branch. */
        for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
            out->posterior[i] = HIRAM_PRIOR_MARGINAL[i];
        }
    } else {
        for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
            out->posterior[i] = ws->unnormalized_joint[i] / likelihood;
        }
    }

    /* evaluate_expected_loss is unconditional in Python's step() (even on
     * the fallback path, to populate expected_losses for diagnostics) --
     * mirrored here the same way. */
    int32_t a;
    for (a = 0; a < HIRAM_N_ACTIONS; a++) {
        double total = 0.0;
        for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
            total += out->posterior[i] * HIRAM_LOSS_MATRIX[a * HIRAM_LOSS_ACTION_STRIDE + i];
        }
        out->expected_losses[a] = total;
    }

    int32_t best_idx;
    if (fallback_active) {
        /* Python: the fallback path unconditionally overrides whatever
         * evaluate_expected_loss would have picked. */
        best_idx = HIRAM_EMERGENCY_BRAKE_INDEX;
    } else {
        bool all_finite = true;
        for (a = 0; a < HIRAM_N_ACTIONS; a++) {
            if (!isfinite(out->expected_losses[a])) {
                all_finite = false;
                break;
            }
        }
        if (!all_finite) {
            best_idx = HIRAM_EMERGENCY_BRAKE_INDEX;
        } else {
            best_idx = 0;
            for (a = 1; a < HIRAM_N_ACTIONS; a++) {
                const bool strictly_lower_loss = out->expected_losses[a] < out->expected_losses[best_idx];
                const bool tied_but_higher_priority =
                    out->expected_losses[a] == out->expected_losses[best_idx] &&
                    HIRAM_TIE_BREAK_PRIORITY[a] < HIRAM_TIE_BREAK_PRIORITY[best_idx];
                if (strictly_lower_loss || tied_but_higher_priority) {
                    best_idx = a;
                }
            }
        }
    }

    out->selected_action = best_idx;
    out->fallback_active = fallback_active;
    return HIRAM_OK;
}
