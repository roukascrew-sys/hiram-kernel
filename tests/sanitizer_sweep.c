/*
 * Exhaustive API sweep with no Python and no ctypes.
 *
 * Enumerates all 3**CIRCUIT_VAR_COUNT evidence configurations (unobserved /
 * false / true per variable) and calls both audit entry points on each, giving
 * 1,062,882 calls for the shipped 12-variable model.
 *
 * It exists to be run under a sanitizer. The project's undefined-behaviour
 * check previously lived only in an out-of-tree harness that skipped the
 * sanitizer entirely on Windows, so the exhaustive sweep had never once
 * executed with UBSan active. A driver that links straight against the kernel
 * runs anywhere the kernel builds, CI included.
 *
 * Exit codes: 0 all decisions well-formed; 1 a canonical evidence mask was
 * rejected as invalid, which would mean the validation contract is wrong.
 *
 * Build (plain):
 *   gcc -std=c99 -Iinclude -Iinclude/hiram src/hiram_eval.c tests/sanitizer_sweep.c -o sweep
 * Build (instrumented):
 *   gcc -std=c99 -g -O1 -fsanitize=undefined,address -fno-sanitize-recover=all ...
 */

#include <stdio.h>
#include "hiram_circuit_data.h"

int main(void)
{
    HiramContext ctx;
    hiram_context_init(&ctx);

    const Rational scalar_threshold = { 1, 20 };
    const Rational vector_thresholds[MONITORED_HAZARDS_COUNT] = {
        { 1, 20 },   /* stall      */
        { 1, 50 },   /* icing      */
        { 1, 500 }   /* structural */
    };

    unsigned long scalar_counts[4] = { 0UL, 0UL, 0UL, 0UL };
    unsigned long vector_counts[4] = { 0UL, 0UL, 0UL, 0UL };
    unsigned long veto_bits[MONITORED_HAZARDS_COUNT] = { 0UL, 0UL, 0UL };

    unsigned long total = 1UL;
    for (int i = 0; i < CIRCUIT_VAR_COUNT; i++) {
        total *= 3UL;
    }

    for (unsigned long n = 0UL; n < total; n++) {
        unsigned long digits = n;
        Evidence ev;
        hiram_evidence_init(&ev);

        for (int v = 0; v < CIRCUIT_VAR_COUNT; v++) {
            unsigned long d = digits % 3UL;
            digits /= 3UL;
            if (d != 0UL) {
                hiram_evidence_set(&ev, (uint16_t)v, (d == 2UL));
            }
        }

        HiramAuditReport rep = hiram_audit_hazard(&ev, scalar_threshold, &ctx);
        scalar_counts[(int)rep.decision]++;

        HiramVectorReport vrep =
            hiram_audit_hazard_vector(&ev, vector_thresholds, &ctx);
        vector_counts[(int)vrep.overall_decision]++;

        for (int h = 0; h < MONITORED_HAZARDS_COUNT; h++) {
            if ((vrep.veto_mask & (1U << (unsigned)h)) != 0U) {
                veto_bits[h]++;
            }
        }
    }

    printf("configurations : %lu\n", total);
    printf("api calls      : %lu\n", total * 2UL);
    printf("scalar         : approved=%lu vetoed=%lu contradiction=%lu invalid=%lu\n",
           scalar_counts[0], scalar_counts[1], scalar_counts[2], scalar_counts[3]);
    printf("vector         : approved=%lu vetoed=%lu contradiction=%lu invalid=%lu\n",
           vector_counts[0], vector_counts[1], vector_counts[2], vector_counts[3]);
    printf("vetoes by hazard: stall=%lu icing=%lu structural=%lu\n",
           veto_bits[0], veto_bits[1], veto_bits[2]);

    /* Every mask generated above is canonical by construction, so a single
       INVALID_INPUT means the validation contract rejects something it should
       accept. That is a failure, not a statistic. */
    if (scalar_counts[3] != 0UL || vector_counts[3] != 0UL) {
        fprintf(stderr, "FAIL: canonical evidence rejected as invalid input\n");
        return 1;
    }

    printf("OK\n");
    return 0;
}
