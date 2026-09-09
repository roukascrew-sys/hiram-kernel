#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <windows.h>
#include "hiram_eval.h"
#include "hiram_circuit_data.h"

#define NUM_TRIALS 10000

static int compare_doubles(const void* a, const void* b) {
    double diff = (*(double*)a - *(double*)b);
    return (diff > 0) - (diff < 0);
}

int main(void) {
    LARGE_INTEGER freq, start, end;
    QueryPerformanceFrequency(&freq);
    double latency_us[NUM_TRIALS];

    printf("================================================================\n");
    printf("HIRAM NATIVE BARE-METAL C LATENCY & WCET BENCHMARK\n");
    printf("================================================================\n");
    printf("Target Frame: 400 Hz (2,500.0 us budget)\n");
    printf("Executing %d consecutive evaluation cycles...\n", NUM_TRIALS);

    Evidence ev;
    hiram_evidence_init(&ev);
    hiram_evidence_set(&ev, VAR_TERRAIN_CLEAR, true);
    hiram_evidence_set(&ev, VAR_ALTITUDE_STABLE, true);
    Rational risk_threshold = {5LL, 100LL};

    HiramContext ctx;
    hiram_context_init(&ctx);

    /* Warm-up L1 instruction cache */
    for (int i = 0; i < 500; i++) {
        volatile HiramAuditReport r = hiram_audit_hazard(&ev, risk_threshold, &ctx);
        (void)r;
    }

    /* Timed benchmark loop */
    double sum_us = 0.0;
    for (int i = 0; i < NUM_TRIALS; i++) {
        QueryPerformanceCounter(&start);

        /* Alternating telemetry to defeat branch predictor bias */
        if (i & 1) {
            hiram_evidence_set(&ev, VAR_LOW_AIRSPEED, true);
        } else {
            hiram_evidence_clear(&ev, VAR_LOW_AIRSPEED);
        }

        volatile HiramAuditReport rep = hiram_audit_hazard(&ev, risk_threshold, &ctx);
        (void)rep;

        QueryPerformanceCounter(&end);

        double elapsed = (double)(end.QuadPart - start.QuadPart) * 1000000.0 / (double)freq.QuadPart;
        latency_us[i] = elapsed;
        sum_us += elapsed;
    }

    /* Compute statistical percentiles */
    qsort(latency_us, NUM_TRIALS, sizeof(double), compare_doubles);

    double mean_us = sum_us / NUM_TRIALS;
    double p50_us  = latency_us[(int)(NUM_TRIALS * 0.50)];
    double p95_us  = latency_us[(int)(NUM_TRIALS * 0.95)];
    double p99_us  = latency_us[(int)(NUM_TRIALS * 0.99)];
    double wcet_us = latency_us[NUM_TRIALS - 1];

    printf("----------------------------------------------------------------\n");
    printf("METRIC                   | TIME (us)      | 400Hz FRAME BUDGET UTIL\n");
    printf("----------------------------------------------------------------\n");
    printf("Mean Latency             | %8.2f us    | %7.4f%%\n", mean_us, (mean_us / 2500.0) * 100.0);
    printf("Median (P50)             | %8.2f us    | %7.4f%%\n", p50_us,  (p50_us  / 2500.0) * 100.0);
    printf("P95 Latency              | %8.2f us    | %7.4f%%\n", p95_us,  (p95_us  / 2500.0) * 100.0);
    printf("P99 Latency              | %8.2f us    | %7.4f%%\n", p99_us,  (p99_us  / 2500.0) * 100.0);
    printf("Worst-Case (WCET)        | %8.2f us    | %7.4f%%\n", wcet_us, (wcet_us / 2500.0) * 100.0);
    printf("----------------------------------------------------------------\n");

    if (wcet_us < 2500.0) {
        printf("400 Hz COMPLIANCE: PASSED (Headroom: %.2f us / %.2f%%)\n",
               2500.0 - wcet_us, ((2500.0 - wcet_us) / 2500.0) * 100.0);
    } else {
        printf("400 Hz COMPLIANCE: FAILED\n");
    }
    printf("================================================================\n");

    return 0;
}