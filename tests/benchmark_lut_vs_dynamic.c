#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include "hiram_kernel.h"
#include "hiram_policy_lut.h"

#define BENCHMARK_ITERATIONS 1000000

static inline double get_time_sec(void) {
    struct timespec ts;
#if defined(CLOCK_MONOTONIC_RAW)
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
#else
    clock_gettime(CLOCK_MONOTONIC, &ts);
#endif
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(void) {
    hiram_workspace_t ws;
    int mismatches = 0;

    printf("====================================================================\n");
    printf("     HIRAM BENCHMARK: DYNAMIC BAYESIAN ENGINE VS POLICY LUT         \n");
    printf("====================================================================\n");

    /* 1. Bit-Exact Policy Parity across all 64 combinations */
    for (int l = -1; l <= 2; ++l) {
        for (int t = -1; t <= 2; ++t) {
            for (int s = -1; s <= 2; ++s) {
                hiram_evidence_t ev = { .lidar_obs = l, .tof_obs = t, .wheel_slip_obs = s };
                hiram_decision_t dec;
                hiram_status_t status = hiram_kernel_step(&ev, &ws, &dec);
                
                uint8_t lut_action = hiram_policy_lut_step(l, t, s);

                if (status != HIRAM_OK || dec.selected_action != lut_action) {
                    printf("[PARITY ERROR] L:%d T:%d S:%d -> Dynamic:%d, LUT:%d\n",
                           l, t, s, dec.selected_action, lut_action);
                    mismatches++;
                }
            }
        }
    }

    if (mismatches > 0) {
        printf("FAILED: %d parity mismatches found!\n", mismatches);
        return 1;
    }
    printf("[PASS] Parity Check: 64/64 combinations bit-exact match.\n\n");

    /* Prepare test vectors */
    int32_t *test_l = (int32_t *)malloc(sizeof(int32_t) * BENCHMARK_ITERATIONS);
    int32_t *test_t = (int32_t *)malloc(sizeof(int32_t) * BENCHMARK_ITERATIONS);
    int32_t *test_s = (int32_t *)malloc(sizeof(int32_t) * BENCHMARK_ITERATIONS);

    if (!test_l || !test_t || !test_s) {
        printf("Memory allocation failed for benchmark vectors.\n");
        return 1;
    }

    srand(42);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        test_l[i] = (rand() % 4) - 1;
        test_t[i] = (rand() % 4) - 1;
        test_s[i] = (rand() % 4) - 1;
    }

    /* 2. Benchmark Dynamic Bayesian Engine */
    volatile uint32_t dynamic_sink = 0;
    double t0_dyn = get_time_sec();
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        hiram_evidence_t ev = { .lidar_obs = test_l[i], .tof_obs = test_t[i], .wheel_slip_obs = test_s[i] };
        hiram_decision_t dec;
        hiram_kernel_step(&ev, &ws, &dec);
        dynamic_sink += (uint32_t)dec.selected_action;
    }
    double t1_dyn = get_time_sec();
    double dyn_total_us = (t1_dyn - t0_dyn) * 1e6;
    double dyn_per_step_ns = (dyn_total_us / BENCHMARK_ITERATIONS) * 1e3;

    /* 3. Benchmark Policy Lookup Table (LUT) */
    volatile uint32_t lut_sink = 0;
    double t0_lut = get_time_sec();
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        lut_sink += (uint32_t)hiram_policy_lut_step(test_l[i], test_t[i], test_s[i]);
    }
    double t1_lut = get_time_sec();
    double lut_total_us = (t1_lut - t0_lut) * 1e6;
    double lut_per_step_ns = (lut_total_us / BENCHMARK_ITERATIONS) * 1e3;

    /* Summary Report */
    printf("Benchmark Results (%d iterations):\n", BENCHMARK_ITERATIONS);
    printf("--------------------------------------------------------------------\n");
    printf("Implementation             | Total Time  | Avg Latency  | Speedup   \n");
    printf("--------------------------------------------------------------------\n");
    printf("Dynamic C Kernel (Floating)| %8.2f ms | %7.2f ns   | Baseline  \n",
           dyn_total_us / 1000.0, dyn_per_step_ns);
    printf("Synthesized Policy LUT     | %8.2f ms | %7.2f ns   | %6.1fx    \n",
           lut_total_us / 1000.0, lut_per_step_ns, dyn_per_step_ns / lut_per_step_ns);
    printf("--------------------------------------------------------------------\n");
    printf("Footprint & Sinks:\n");
    printf("  - Dynamic Kernel Workspace: %lu bytes (RAM)\n", (unsigned long)sizeof(hiram_workspace_t));
    printf("  - Policy LUT Footprint:     64 bytes (Flash / ROM)\n");
    printf("  - Sinks (Anti-dead-code):   dyn=%u, lut=%u\n", (unsigned int)dynamic_sink, (unsigned int)lut_sink);
    printf("====================================================================\n");

    free(test_l);
    free(test_t);
    free(test_s);
    return 0;
}