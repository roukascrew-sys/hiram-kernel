#ifndef HIRAM_EVAL_H
#define HIRAM_EVAL_H

/*
 * Accumulator width is selected in hiram_eval.c via HIRAM_FORCE_64BIT_ACC.
 * This header deliberately defines no compiler-reserved identifiers: defining
 * names in the implementation's reserved namespace (C99 7.1.3) is undefined
 * behaviour, and the previous __SIZEOF_INT128__ override also silently
 * disabled the 128-bit accumulator it claimed to be protecting.
 */

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CIRCUIT_VAR_COUNT 12

typedef struct {
    int64_t num;
    int64_t den;
} Rational;

typedef struct {
    uint32_t observed_mask;
    uint32_t values_mask;
} Evidence;

typedef enum {
    NODE_LITERAL = 0,
    NODE_PROD    = 1,
    NODE_SUM     = 2
} NodeType;

typedef struct {
    NodeType type;
    uint16_t var_id;
    uint8_t  is_negated;
    uint16_t child_count;
    uint16_t child_offset;
    uint16_t weight_offset;
} CircuitNode;

typedef enum {
    HIRAM_APPROVED                  = 0,
    HIRAM_VETOED_SAFETY_VIOLATION   = 1,
    HIRAM_REJECTED_CONTRADICTION    = 2,
    HIRAM_REJECTED_INVALID_INPUT    = 3
} HiramDecision;

typedef struct {
    HiramDecision decision;
    Rational      hazard_probability;
    Rational      hazard_threshold;
} HiramAuditReport;

/* No default node count on purpose. A default would let one translation unit
   size HiramContext.memo smaller than the node count the kernel writes into --
   a silent cross-module buffer overflow that no single-file diagnostic catches.
   Include hiram_circuit_data.h; it defines the dimensions then includes this. */
#ifndef CIRCUIT_NODE_COUNT
#error "Include hiram_circuit_data.h (not hiram_eval.h) to obtain CIRCUIT_NODE_COUNT."
#endif

typedef struct {
    Rational memo[CIRCUIT_NODE_COUNT];
    uint32_t execution_flags;
} HiramContext;

#define MONITORED_HAZARDS_COUNT 3

typedef struct {
    HiramDecision overall_decision;
    uint32_t      veto_mask;
    Rational      hazard_posteriors[MONITORED_HAZARDS_COUNT];
    Rational      hazard_thresholds[MONITORED_HAZARDS_COUNT];
} HiramVectorReport;

void hiram_evidence_init(Evidence* ev);
void hiram_evidence_set(Evidence* ev, uint16_t var_id, bool val);
void hiram_evidence_clear(Evidence* ev, uint16_t var_id);
void hiram_context_init(HiramContext* ctx);
Rational hiram_eval_evidence(const Evidence* ev, HiramContext* ctx);
HiramAuditReport hiram_audit_hazard(const Evidence* base_ev, Rational max_acceptable_risk, HiramContext* ctx);
HiramVectorReport hiram_audit_hazard_vector(
    const Evidence* base_ev,
    const Rational  thresholds[MONITORED_HAZARDS_COUNT],
    HiramContext*   ctx
);

#ifdef __cplusplus
}
#endif

#endif /* HIRAM_EVAL_H */