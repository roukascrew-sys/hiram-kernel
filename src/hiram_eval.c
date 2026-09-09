/* HAND-MAINTAINED ZERO-MALLOC EMBEDDED C RUNTIME (not generated; only
   include/hiram/hiram_circuit_data.h is emitted by tools/hiram_c_exporter.py) */
/* MISRA-C:2012 Rule 17.2 Compliant: Flat Non-Recursive Topological Evaluation */
#include "hiram_circuit_data.h"   /* defines circuit dimensions, then includes hiram_eval.h */

#ifdef HIRAM_STANDALONE_TEST
#include <stdio.h>
#endif

/*
 * Compile-time structural invariants (C99: negative-array-size idiom, because
 * _Static_assert is C11 and this unit builds under -std=c99 -pedantic-errors).
 *
 * hiram_eval.h must size HiramContext.memo from the SAME node count the
 * generated circuit tables use. If a model change alters CIRCUIT_NODE_COUNT and
 * the header's fallback is not updated, every hiram_eval_evidence() call writes
 * past the end of the caller's context. That must be a build failure, not a
 * runtime memory corruption discovered in flight.
 */
#define HIRAM_STATIC_ASSERT(cond, tag)     typedef char hiram_static_assert_##tag[(cond) ? 1 : -1]

HIRAM_STATIC_ASSERT(
    (sizeof(((HiramContext *)0)->memo) / sizeof(Rational)) >= CIRCUIT_NODE_COUNT,
    context_memo_covers_every_circuit_node);
HIRAM_STATIC_ASSERT(CIRCUIT_ROOT_ID < CIRCUIT_NODE_COUNT,
    root_node_index_within_circuit);
HIRAM_STATIC_ASSERT(CIRCUIT_VAR_COUNT <= 32,
    variable_count_fits_evidence_bitmask);
HIRAM_STATIC_ASSERT(HAZARD_VAR_ID < CIRCUIT_VAR_COUNT,
    scalar_hazard_variable_in_range);

#if defined(HIRAM_FORCE_64BIT_ACC) || !defined(__SIZEOF_INT128__)
typedef int64_t hiram_acc_t;
#else
typedef __int128_t hiram_acc_t;
#endif

static inline hiram_acc_t gcd_acc(hiram_acc_t a, hiram_acc_t b) {
    if (a < 0) { a = -a; }
    if (b < 0) { b = -b; }
    while (b != 0) {
        hiram_acc_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

static inline Rational rational_make(hiram_acc_t n, hiram_acc_t d) {
    if (d < 0) { n = -n; d = -d; }
    if (n == 0) { return (Rational){0LL, 1LL}; }
    hiram_acc_t g = gcd_acc(n, d);
    return (Rational){(int64_t)(n / g), (int64_t)(d / g)};
}

static inline Rational rational_mul(Rational a, Rational b) {
    if (a.num == 0 || b.num == 0) { return (Rational){0LL, 1LL}; }
    hiram_acc_t g1 = gcd_acc(a.num, b.den);
    hiram_acc_t g2 = gcd_acc(b.num, a.den);
    hiram_acc_t n = (hiram_acc_t)(a.num / g1) * (b.num / g2);
    hiram_acc_t d = (hiram_acc_t)(a.den / g2) * (b.den / g1);
    return (Rational){(int64_t)n, (int64_t)d};
}

static inline Rational rational_add(Rational a, Rational b) {
    if (a.num == 0) { return b; }
    if (b.num == 0) { return a; }
    hiram_acc_t g = gcd_acc(a.den, b.den);
    hiram_acc_t d = ((hiram_acc_t)a.den / g) * b.den;
    hiram_acc_t n = ((hiram_acc_t)a.num * (b.den / g)) + ((hiram_acc_t)b.num * (a.den / g));
    return rational_make(n, d);
}

static inline Rational rational_div(Rational a, Rational b) {
    return rational_mul(a, (Rational){b.den, b.num});
}

/*
 * Overflow-Free Euclidean Rational Comparator:
 * Compares a/b vs c/d strictly using integer division and remainders.
 * Never multiplies large denominators, eliminating 64-bit integer overflow.
 * Returns:  1 if a/b > c/d
 *          -1 if a/b < c/d
 *           0 if a/b == c/d
 */
static int rational_cmp(int64_t a_num, int64_t b_den, int64_t c_num, int64_t d_den) {
    if (a_num == 0 && c_num == 0) { return 0; }
    if (a_num == 0) { return -1; }
    if (c_num == 0) { return 1; }

    int64_t a = a_num;
    int64_t b = b_den;
    int64_t c = c_num;
    int64_t d = d_den;

    while (1) {
        int64_t q1 = a / b;
        int64_t q2 = c / d;

        if (q1 != q2) {
            return (q1 > q2) ? 1 : -1;
        }

        int64_t r1 = a % b;
        int64_t r2 = c % d;

        if (r1 == 0 && r2 == 0) { return 0; }
        if (r1 == 0) { return -1; }
        if (r2 == 0) { return 1; }

        int64_t next_a = d;
        int64_t next_b = r2;
        int64_t next_c = b;
        int64_t next_d = r1;

        a = next_a;
        b = next_b;
        c = next_c;
        d = next_d;
    }
}

void hiram_evidence_init(Evidence* ev) {
    ev->observed_mask = 0;
    ev->values_mask = 0;
}

void hiram_evidence_set(Evidence* ev, uint16_t var_id, bool val) {
    if (var_id >= 32) { return; }
    ev->observed_mask |= (1u << var_id);
    if (val) {
        ev->values_mask |= (1u << var_id);
    } else {
        ev->values_mask &= ~(1u << var_id);
    }
}

void hiram_evidence_clear(Evidence* ev, uint16_t var_id) {
    if (var_id >= 32) { return; }
    ev->observed_mask &= ~(1u << var_id);
    ev->values_mask &= ~(1u << var_id);
}

void hiram_context_init(HiramContext* ctx) {
    if (ctx == NULL) { return; }
    ctx->execution_flags = 0;
    for (uint16_t i = 0; i < CIRCUIT_NODE_COUNT; i++) {
        ctx->memo[i] = (Rational){0LL, 1LL};
    }
}

Rational hiram_eval_evidence(const Evidence* ev, HiramContext* ctx) {
    if (ctx == NULL || ev == NULL) {
        return (Rational){0LL, 1LL};
    }

    for (uint16_t i = 0; i < CIRCUIT_NODE_COUNT; i++) {
        const CircuitNode* n = &g_circuit_nodes[i];
        Rational res;

        if (n->type == NODE_LITERAL) {
            uint32_t mask = (1u << n->var_id);
            if ((ev->observed_mask & mask) == 0) {
                res = (Rational){1LL, 1LL};
            } else {
                bool val = (ev->values_mask & mask) != 0;
                bool sat = (val && !n->is_negated) || (!val && n->is_negated);
                res = sat ? (Rational){1LL, 1LL} : (Rational){0LL, 1LL};
            }
        } else if (n->type == NODE_PROD) {
            res = (Rational){1LL, 1LL};
            for (uint16_t c = 0; c < n->child_count; c++) {
                uint16_t child_id = g_circuit_children[n->child_offset + c];
                res = rational_mul(res, ctx->memo[child_id]);
                if (res.num == 0) { break; }
            }
        } else { /* NODE_SUM */
            res = (Rational){0LL, 1LL};
            for (uint16_t c = 0; c < n->child_count; c++) {
                uint16_t child_id = g_circuit_children[n->child_offset + c];
                Rational w = g_circuit_weights[n->weight_offset + c];
                Rational branch = rational_mul(w, ctx->memo[child_id]);
                res = rational_add(res, branch);
            }
        }

        ctx->memo[i] = res;
    }

    return ctx->memo[CIRCUIT_ROOT_ID];
}

HiramAuditReport hiram_audit_hazard(const Evidence* base_ev, Rational max_acceptable_risk, HiramContext* ctx) {
    HiramAuditReport rep;
    rep.hazard_threshold = max_acceptable_risk;

    /* 0. Validate Context & Input Pointers */
    if (ctx == NULL || base_ev == NULL) {
        rep.decision = HIRAM_REJECTED_INVALID_INPUT;
        rep.hazard_probability = (Rational){0LL, 1LL};
        return rep;
    }

    /* 1. Validate Input Threshold Contract ([0, 1] range) */
    if (max_acceptable_risk.den <= 0 || max_acceptable_risk.num < 0 ||
        max_acceptable_risk.num > max_acceptable_risk.den) {
        rep.decision = HIRAM_REJECTED_INVALID_INPUT;
        rep.hazard_probability = (Rational){0LL, 1LL};
        return rep;
    }

    /* 2. Validate Evidence Mask Boundaries & Canonical Form (values_mask <= observed_mask) */
    uint32_t valid_mask = (1u << CIRCUIT_VAR_COUNT) - 1u;
    if ((base_ev->observed_mask & ~valid_mask) != 0 ||
        (base_ev->values_mask & ~valid_mask) != 0 ||
        (base_ev->values_mask & ~base_ev->observed_mask) != 0) {
        rep.decision = HIRAM_REJECTED_INVALID_INPUT;
        rep.hazard_probability = (Rational){0LL, 1LL};
        return rep;
    }

    /* 3. Evaluate Marginal Evidence Mass P(E) */
    Rational p_e = hiram_eval_evidence(base_ev, ctx);
    if (p_e.num == 0) {
        rep.decision = HIRAM_REJECTED_CONTRADICTION;
        rep.hazard_probability = (Rational){0LL, 1LL};
        return rep;
    }

    /* 4. Handle Explicitly Observed Hazard Values */
    uint32_t hazard_mask = (1u << HAZARD_VAR_ID);
    if ((base_ev->observed_mask & hazard_mask) != 0) {
        bool hazard_observed = (base_ev->values_mask & hazard_mask) != 0;
        if (!hazard_observed) {
            /* Explicit hazard=false => P(Hazard | E) = 0 */
            rep.hazard_probability = (Rational){0LL, 1LL};
            rep.decision = HIRAM_APPROVED;
            return rep;
        } else {
            /* Explicit hazard=true => P(Hazard | E) = 1 */
            rep.hazard_probability = (Rational){1LL, 1LL};
            if (max_acceptable_risk.num < max_acceptable_risk.den) {
                rep.decision = HIRAM_VETOED_SAFETY_VIOLATION;
            } else {
                rep.decision = HIRAM_APPROVED;
            }
            return rep;
        }
    }

    /* 5. Hazard is Unobserved: Compute Posterior P(Hazard, E) / P(E) */
    Evidence ev_hazard = *base_ev;
    hiram_evidence_set(&ev_hazard, HAZARD_VAR_ID, true);
    Rational p_hazard_and_e = hiram_eval_evidence(&ev_hazard, ctx);
    Rational posterior = rational_div(p_hazard_and_e, p_e);
    rep.hazard_probability = posterior;

    /* 6. Overflow-Free Rational Risk Comparison */
    int cmp = rational_cmp(posterior.num, posterior.den,
                           max_acceptable_risk.num, max_acceptable_risk.den);
    if (cmp > 0) {
        rep.decision = HIRAM_VETOED_SAFETY_VIOLATION;
    } else {
        rep.decision = HIRAM_APPROVED;
    }
    return rep;
}

static const uint16_t g_monitored_hazard_vars[MONITORED_HAZARDS_COUNT] = {
    VAR_STALL_HAZARD,
    VAR_ICING_HAZARD,
    VAR_STRUCTURAL_HAZARD
};

HiramVectorReport hiram_audit_hazard_vector(
    const Evidence* base_ev,
    const Rational  thresholds[MONITORED_HAZARDS_COUNT],
    HiramContext*   ctx
) {
    HiramVectorReport rep;
    rep.overall_decision = HIRAM_APPROVED;
    rep.veto_mask = 0;
    for (uint16_t h = 0; h < MONITORED_HAZARDS_COUNT; h++) {
        rep.hazard_posteriors[h] = (Rational){0LL, 1LL};
        rep.hazard_thresholds[h] = (thresholds != NULL) ? thresholds[h] : (Rational){0LL, 1LL};
    }

    /* 0. Defensive Validation */
    if (ctx == NULL || base_ev == NULL || thresholds == NULL) {
        rep.overall_decision = HIRAM_REJECTED_INVALID_INPUT;
        return rep;
    }

    /* 1. Validate Thresholds */
    for (uint16_t h = 0; h < MONITORED_HAZARDS_COUNT; h++) {
        Rational th = thresholds[h];
        if (th.den <= 0 || th.num < 0 || th.num > th.den) {
            rep.overall_decision = HIRAM_REJECTED_INVALID_INPUT;
            return rep;
        }
    }

    /* 2. Validate Evidence Masks */
    uint32_t valid_mask = (1u << CIRCUIT_VAR_COUNT) - 1u;
    if ((base_ev->observed_mask & ~valid_mask) != 0 ||
        (base_ev->values_mask & ~valid_mask) != 0 ||
        (base_ev->values_mask & ~base_ev->observed_mask) != 0) {
        rep.overall_decision = HIRAM_REJECTED_INVALID_INPUT;
        return rep;
    }

    /* 3. Evaluate Marginal Evidence Mass P(E) once */
    Rational p_e = hiram_eval_evidence(base_ev, ctx);
    if (p_e.num == 0) {
        rep.overall_decision = HIRAM_REJECTED_CONTRADICTION;
        return rep;
    }

    /* 4. Evaluate each monitored hazard */
    for (uint16_t h = 0; h < MONITORED_HAZARDS_COUNT; h++) {
        uint16_t var_id = g_monitored_hazard_vars[h];
        uint32_t mask = (1u << var_id);
        Rational post;
        Rational th = thresholds[h];

        if ((base_ev->observed_mask & mask) != 0) {
            bool obs_val = (base_ev->values_mask & mask) != 0;
            post = obs_val ? (Rational){1LL, 1LL} : (Rational){0LL, 1LL};
        } else {
            Evidence ev_h = *base_ev;
            hiram_evidence_set(&ev_h, var_id, true);
            Rational p_joint = hiram_eval_evidence(&ev_h, ctx);
            post = rational_div(p_joint, p_e);
        }
        rep.hazard_posteriors[h] = post;

        if (rational_cmp(post.num, post.den, th.num, th.den) > 0) {
            rep.veto_mask |= (1u << h);
        }
    }

    if (rep.veto_mask != 0) {
        rep.overall_decision = HIRAM_VETOED_SAFETY_VIOLATION;
    }

    return rep;
}

#ifdef HIRAM_STANDALONE_TEST
int main(void) {
    Evidence ev;
    hiram_evidence_init(&ev);
    HiramContext ctx;
    hiram_context_init(&ctx);
    HiramAuditReport rep = hiram_audit_hazard(&ev, (Rational){1, 20}, &ctx);
    printf("Nominal descent decision=%d\n", rep.decision);
    return (rep.decision == HIRAM_APPROVED) ? 0 : 1;
}
#endif