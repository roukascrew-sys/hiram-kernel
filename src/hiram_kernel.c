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

/* E[loss | action] = sum_{o,d} posterior[o,d] * loss[action,o,d], mirroring
 * evaluate_expected_loss's dot product exactly. Factored out so both the
 * normal inference path and the safe-fallback path below (zero-likelihood
 * *and* invalid-evidence) populate expected_losses identically instead of
 * one of them risking drifting out of step with the other. */
static void hiram_evaluate_expected_losses(const double *posterior, double *expected_losses_out) {
    int32_t a, i;
    for (a = 0; a < HIRAM_N_ACTIONS; a++) {
        double total = 0.0;
        for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
            total += posterior[i] * HIRAM_LOSS_MATRIX[a * HIRAM_LOSS_ACTION_STRIDE + i];
        }
        expected_losses_out[a] = total;
    }
}

hiram_status_t hiram_kernel_validate_loss_matrix(const double *matrix) {
    int32_t i;

    if (matrix == NULL) {
        return HIRAM_ERR_NULL_POINTER;
    }
    for (i = 0; i < (HIRAM_N_ACTIONS * HIRAM_LOSS_ACTION_STRIDE); i++) {
        /* isfinite() rejects NaN and both infinities; the second test then
         * only ever sees an ordered value, so the sign comparison means what
         * it looks like it means. */
        if (!isfinite(matrix[i]) || (matrix[i] < 0.0)) {
            return HIRAM_ERR_MODEL_CORRUPTED;
        }
    }
    return HIRAM_OK;
}

/*
 * Populates `*out` with the conservative safe-fallback decision: the
 * model's true prior, EMERGENCY_BRAKE, and fallback_active=true. Used both
 * when evidence is contradictory/degenerate (zero-likelihood) and when
 * evidence is outright invalid -- in the latter case specifically so a
 * caller that reuses a `hiram_decision_t` across calls never observes a
 * stale decision (e.g. an earlier call's ACCEL) left over from before the
 * rejected call: every field is overwritten, never left untouched.
 */
static void hiram_populate_safe_fallback(hiram_decision_t *out, hiram_status_t status) {
    int32_t i;
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        out->posterior[i] = HIRAM_PRIOR_MARGINAL[i];
    }
    if (hiram_kernel_validate_loss_matrix(HIRAM_LOSS_MATRIX) == HIRAM_OK) {
        hiram_evaluate_expected_losses(out->posterior, out->expected_losses);
    } else {
        /* The diagnostic field cannot be filled from a matrix that has just
         * been rejected -- doing so would publish numbers derived from known-
         * corrupt data next to a status that says the data is corrupt. Zero
         * is the one value that carries no claim. The action below is
         * EMERGENCY_BRAKE regardless, so nothing downstream depends on it. */
        for (i = 0; i < HIRAM_N_ACTIONS; i++) {
            out->expected_losses[i] = 0.0;
        }
    }
    out->selected_action = HIRAM_EMERGENCY_BRAKE_INDEX;
    out->selected_action_index = HIRAM_EMERGENCY_BRAKE_INDEX;
    out->fallback_active = true;
    out->status = status;
}

hiram_status_t hiram_kernel_verify_custody(void) {
    if (strcmp(HIRAM_FROZEN_IR_DIGEST, HIRAM_KERNEL_EXPECTED_IR_DIGEST) != 0) {
        return HIRAM_ERR_CUSTODY_MISMATCH;
    }
    return HIRAM_OK;
}

hiram_status_t hiram_kernel_decide(const double *joint, hiram_decision_t *out) {
    int32_t i;
    int32_t a;
    int32_t best_idx;
    double joint_sum = 0.0;

    if (out == NULL) {
        return HIRAM_ERR_NULL_POINTER;
    }
    if (joint == NULL) {
        hiram_populate_safe_fallback(out, HIRAM_ERR_NULL_POINTER);
        return HIRAM_ERR_NULL_POINTER;
    }

    /* B2: the model is checked before it is used. An argmin over a matrix
     * that is not a cost matrix will still return an index, and that index
     * is what actuates the brakes. */
    if (hiram_kernel_validate_loss_matrix(HIRAM_LOSS_MATRIX) != HIRAM_OK) {
        hiram_populate_safe_fallback(out, HIRAM_ERR_MODEL_CORRUPTED);
        return HIRAM_ERR_MODEL_CORRUPTED;
    }

    /* M2: every mass must be a real, non-negative probability mass before it
     * is summed. A NaN from a corrupted CPT used to survive the whole
     * pipeline -- `NaN < HIRAM_ZERO_LIKELIHOOD_GUARD` is false, so the
     * zero-likelihood fallback did not fire, and NaN/NaN then propagated
     * through the posterior into every expected loss. The downstream
     * finiteness check did force EMERGENCY_BRAKE, but it reported HIRAM_OK
     * with fallback_active == false: a silent degradation, indistinguishable
     * in the telemetry from a genuine braking decision. */
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        if (!isfinite(joint[i]) || (joint[i] < 0.0)) {
            hiram_populate_safe_fallback(out, HIRAM_ERR_NUMERICAL_FAULT);
            return HIRAM_ERR_NUMERICAL_FAULT;
        }
        joint_sum += joint[i];
    }
    /* Catches an overflow to +Inf in the summation itself, which no single
     * term being finite can rule out. */
    if (!isfinite(joint_sum)) {
        hiram_populate_safe_fallback(out, HIRAM_ERR_NUMERICAL_FAULT);
        return HIRAM_ERR_NUMERICAL_FAULT;
    }

    const bool fallback_active = joint_sum < HIRAM_ZERO_LIKELIHOOD_GUARD;
    if (fallback_active) {
        /* Degenerate/contradictory evidence: revert to the model's true
         * physical prior rather than dividing by (near) zero, matching
         * _compute_posterior_with_diagnostics's fallback branch. */
        for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
            out->posterior[i] = HIRAM_PRIOR_MARGINAL[i];
        }
    } else {
        for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
            out->posterior[i] = joint[i] / joint_sum;
        }
    }

    /* evaluate_expected_loss is unconditional in Python's step() (even on
     * the fallback path, to populate expected_losses for diagnostics) --
     * mirrored here the same way. */
    hiram_evaluate_expected_losses(out->posterior, out->expected_losses);

    /* Both inputs to the dot product are now known finite and non-negative,
     * so only overflow in the accumulation can produce a non-finite loss.
     * That is a numerical fault in its own right, and is reported as one
     * rather than folded into a silent EMERGENCY_BRAKE with HIRAM_OK. */
    for (a = 0; a < HIRAM_N_ACTIONS; a++) {
        if (!isfinite(out->expected_losses[a])) {
            hiram_populate_safe_fallback(out, HIRAM_ERR_NUMERICAL_FAULT);
            return HIRAM_ERR_NUMERICAL_FAULT;
        }
    }

    if (fallback_active) {
        /* Python: the fallback path unconditionally overrides whatever
         * evaluate_expected_loss would have picked. */
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

    out->selected_action = best_idx;
    out->selected_action_index = best_idx;
    out->fallback_active = fallback_active;
    out->status = HIRAM_OK;
    return HIRAM_OK;
}

hiram_status_t hiram_kernel_step(const hiram_evidence_t *evidence, hiram_workspace_t *ws, hiram_decision_t *out) {
    if (evidence == NULL || ws == NULL || out == NULL) {
        /* M2: `out` is overwritten whenever there is an `out` to overwrite.
         * The previous contract left it untouched, so a caller reusing one
         * hiram_decision_t across calls kept reading the last successful
         * decision -- an ACCEL that no longer had anything to do with the
         * current step -- out of a struct whose status field still said OK. */
        if (out != NULL) {
            hiram_populate_safe_fallback(out, HIRAM_ERR_NULL_POINTER);
        }
        return HIRAM_ERR_NULL_POINTER;
    }
    /* Model integrity outranks evidence validity: a corrupted loss matrix
     * must be reported as such even on a step whose evidence would have been
     * rejected anyway, rather than being masked by the cheaper rejection. */
    if (hiram_kernel_validate_loss_matrix(HIRAM_LOSS_MATRIX) != HIRAM_OK) {
        hiram_populate_safe_fallback(out, HIRAM_ERR_MODEL_CORRUPTED);
        return HIRAM_ERR_MODEL_CORRUPTED;
    }
    if (!hiram_evidence_value_in_range(evidence->lidar_obs) ||
        !hiram_evidence_value_in_range(evidence->tof_obs) ||
        !hiram_evidence_value_in_range(evidence->wheel_slip_obs)) {
        hiram_populate_safe_fallback(out, HIRAM_ERR_INVALID_EVIDENCE);
        return HIRAM_ERR_INVALID_EVIDENCE;
    }

    const int32_t lidar_val = hiram_canonicalize_evidence(evidence->lidar_obs, HIRAM_LIDAROBS_DROPOUT_INDEX);
    const int32_t tof_val = hiram_canonicalize_evidence(evidence->tof_obs, HIRAM_TOFOBS_DROPOUT_INDEX);
    const int32_t wheel_val = hiram_canonicalize_evidence(evidence->wheel_slip_obs, HIRAM_WHEELSLIPOBS_DROPOUT_INDEX);

    int32_t i;
    for (i = 0; i < HIRAM_N_POSTERIOR; i++) {
        ws->posterior_scratch[i] = 0.0;
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

                ws->posterior_scratch[o * HIRAM_N_DECEL + d] +=
                    p_t * p_o_given_t * p_d_given_t * obstacle_evidence_factor * f_wheel;
            }
        }
    }

    /* The marginalisation above is the only part of a step that is specific
     * to evidence; normalisation, loss evaluation, the numerical guards and
     * the argmin are shared with every other caller that can produce a joint
     * distribution, so they live in one place. */
    return hiram_kernel_decide(ws->posterior_scratch, out);
}
