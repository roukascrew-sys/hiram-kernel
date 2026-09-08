#ifndef HIRAM_EVAL_H
#define HIRAM_EVAL_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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

void hiram_evidence_init(Evidence* ev);
void hiram_evidence_set(Evidence* ev, uint16_t var_id, bool val);
void hiram_evidence_clear(Evidence* ev, uint16_t var_id);
Rational hiram_eval_evidence(const Evidence* ev);
HiramAuditReport hiram_audit_hazard(const Evidence* base_ev, Rational max_acceptable_risk);

#ifdef __cplusplus
}
#endif

#endif /* HIRAM_EVAL_H */