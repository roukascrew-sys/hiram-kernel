/*
 * dwt_bench_policy_lut.c - DWT cycle-count benchmark for hiram_policy_lut_step().
 *
 * B4: this file used to carry a second vector table, Reset_Handler, clock_init()
 * and UART driver, and the Makefile linked *that* startup path -- not
 * src/target_main.c's -- into the artifact it called the release image. The MPU
 * lockdown therefore existed in the tree but not in the binary, and the two
 * clock_init()s had drifted apart: this one wrote RCC_PLLCKSELR = 0 (DIVM1 = 0,
 * which disables PLL1 per RM0468), polled PLL1RDY in a loop whose result it
 * discarded, and then reported 550 MHz regardless of what the PLL had done.
 *
 * All of that now lives once, in src/target_main.c, behind include/hiram_board.h.
 * What remains here is the measurement: one routine, called from target_main()
 * after the clock is up, the LUT is in service and the MPU is locked -- so the
 * figure is measured against the configuration that actually ships, at a clock
 * rate the banner has already reported.
 */
#include <stdint.h>
#include <stdbool.h>
#include "hiram_policy_lut.h"
#include "hiram_board.h"

#define BENCH_ITERATIONS 10000u

void hiram_bench_run(void) {
    volatile uint32_t sink = 0u;
    volatile uint32_t dummy_sink = 0u;
    uint32_t i;
    const uint32_t cpu_hz = hiram_cpu_hz();

    /* A latched-out LUT would make every call below return EMERGENCY_BRAKE
       from the guard, timing the fault path instead of the lookup. */
    if (!hiram_policy_lut_is_operational()) {
        hiram_uart_puts("\n  [BENCH SKIPPED] policy LUT is not in service.\n");
        return;
    }

    __asm__ volatile ("cpsid i" : : : "memory");

    /* Calibration pass: the same index arithmetic with no call, so the
       reported net figure is the evaluator rather than the loop around it. */
    const uint32_t t_cal0 = hiram_dwt_cyccnt();
    for (i = 0u; i < BENCH_ITERATIONS; i++) {
        const uint32_t combo = i & 0x3Fu;
        const int32_t lidar = (int32_t)(combo >> 4u)        - 1;
        const int32_t tof   = (int32_t)((combo >> 2u) & 3u) - 1;
        const int32_t slip  = (int32_t)(combo & 3u)         - 1;
        dummy_sink += (uint32_t)(lidar + tof + slip);
    }
    const uint32_t t_cal1 = hiram_dwt_cyccnt();
    const uint32_t loop_overhead = t_cal1 - t_cal0;

    const uint32_t t0 = hiram_dwt_cyccnt();
    for (i = 0u; i < BENCH_ITERATIONS; i++) {
        const uint32_t combo = i & 0x3Fu;
        const int32_t lidar = (int32_t)(combo >> 4u)        - 1;
        const int32_t tof   = (int32_t)((combo >> 2u) & 3u) - 1;
        const int32_t slip  = (int32_t)(combo & 3u)         - 1;
        const hiram_lut_result_t r = hiram_policy_lut_step(lidar, tof, slip);
        sink += r.action;
    }
    const uint32_t t1 = hiram_dwt_cyccnt();

    __asm__ volatile ("cpsie i" : : : "memory");

    const uint32_t total_cycles = t1 - t0;
    const uint32_t net_cycles = (total_cycles > loop_overhead)
        ? (total_cycles - loop_overhead)
        : total_cycles;

    const uint32_t gross_avg_x100 = (uint32_t)(((uint64_t)total_cycles * 100ULL) / BENCH_ITERATIONS);
    const uint32_t net_avg_x100   = (uint32_t)(((uint64_t)net_cycles * 100ULL) / BENCH_ITERATIONS);
    const uint32_t ns_per_call_x10 = (uint32_t)(((uint64_t)net_cycles * 10000000000ULL)
                                              / ((uint64_t)cpu_hz * BENCH_ITERATIONS));

    hiram_uart_puts("\n================================================================\n");
    hiram_uart_puts("  HIRAM DWT BENCHMARK: hiram_policy_lut_step()\n");
    hiram_uart_puts("================================================================\n");
    hiram_uart_puts("  Iterations        : ");
    hiram_uart_print_u32(BENCH_ITERATIONS);
    hiram_uart_puts("\n  CPU Clock         : ");
    hiram_uart_print_u32(cpu_hz / 1000000u);
    hiram_uart_puts(" MHz\n");
    hiram_uart_puts("  Total Gross Cyc   : ");
    hiram_uart_print_u32(total_cycles);
    hiram_uart_puts(" cyc\n");
    hiram_uart_puts("  Loop Overhead Cyc : ");
    hiram_uart_print_u32(loop_overhead);
    hiram_uart_puts(" cyc\n");
    hiram_uart_puts("  Net LUT Cycles    : ");
    hiram_uart_print_u32(net_cycles);
    hiram_uart_puts(" cyc (isolated)\n");
    hiram_uart_puts("  Gross Avg/Call    : ");
    hiram_uart_print_u32(gross_avg_x100 / 100u);
    hiram_uart_putc('.');
    hiram_uart_print_u32(gross_avg_x100 % 100u);
    hiram_uart_puts(" cycles\n");
    hiram_uart_puts("  Net Avg/Call      : ");
    hiram_uart_print_u32(net_avg_x100 / 100u);
    hiram_uart_putc('.');
    hiram_uart_print_u32(net_avg_x100 % 100u);
    hiram_uart_puts(" cycles\n");
    hiram_uart_puts("  Net Time/Call     : ");
    hiram_uart_print_u32(ns_per_call_x10 / 10u);
    hiram_uart_putc('.');
    hiram_uart_print_u32(ns_per_call_x10 % 10u);
    hiram_uart_puts(" ns\n");
    hiram_uart_puts("  Sink (Anti-DCE)   : ");
    hiram_uart_print_u32(sink + dummy_sink);
    hiram_uart_puts("\n================================================================\n\n");
}
