/*
 * hiram_board.h - STM32H723ZG board services shared by the firmware image.
 *
 * B4 root cause: src/target_main.c and benchmarks/dwt_bench_policy_lut.c each
 * carried their own vector table, Reset_Handler, clock_init() and UART driver.
 * Two startup paths cannot both be the one that was reviewed, and the Makefile
 * linked the one *without* the MPU lockdown into the release artifact. There is
 * now exactly one of each: src/target_main.c owns startup, clocking, console,
 * MPU and the actuator safe state, and every other target translation unit
 * consumes them through this header.
 *
 * Nothing here is silicon-validated. Register offsets and field positions are
 * documented at their definitions in src/target_main.c against RM0468
 * (STM32H72x/H73x) and the ARMv7-M ARM; they must be re-checked on a board
 * before any timing figure produced by this image is quoted.
 */
#ifndef HIRAM_BOARD_H
#define HIRAM_BOARD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Outcome of hiram_clock_init(). The distinction is reported in the startup
 * banner because a DWT cycle count is meaningless without the rate it was
 * counted at, and the failure direction here is always "slower than intended"
 * -- never "faster than the verified voltage scale permits".
 */
typedef enum {
    HIRAM_CLOCK_PLL_VOS0 = 0,    /* overdrive confirmed; PLL1 at the VOS0 profile */
    HIRAM_CLOCK_PLL_VOS1 = 1,    /* VOS1 confirmed, VOS0 refused; PLL1 at the VOS1 profile */
    HIRAM_CLOCK_HSI_FALLBACK = 2 /* a bounded wait expired or a read-back disagreed; still on the HSI */
} hiram_clock_result_t;

/*
 * Brings the CPU clock up. Every wait is bounded by a counter and every
 * register write that gates a later decision is read back before that decision
 * is taken, so a wrong constant or an unpowered rail costs a clearly-labelled
 * slow clock rather than a part that never finishes booting.
 */
hiram_clock_result_t hiram_clock_init(void);

uint32_t    hiram_cpu_hz(void);
uint32_t    hiram_pclk1_hz(void);
const char *hiram_clock_note(void);

/*
 * USART3 console. hiram_uart_putc() waits on TXE for a bounded number of
 * iterations and then drops the character: a fault reaction must not be able
 * to stall forever inside a diagnostic print, which is what the blocking
 * `while (!TXE) {}` in the pre-audit handlers did.
 */
void hiram_uart_putc(char c);
void hiram_uart_puts(const char *s);
void hiram_uart_print_u32(uint32_t val);
void hiram_uart_print_u32_pad(uint32_t val, uint32_t width);

/*
 * B5: drives the traction/brake PWM command registers to their inactive state
 * -- outputs off at the timer's master enable first, then channel enables,
 * then duty registers, then the counter itself. A fixed number of stores, no
 * loops, no polling: it is safe to call from a fault handler before the
 * console is known to work, and it is called there *before* any UART output.
 *
 * Idempotent and safe to re-issue, which the trap loops do on every pass so a
 * subsequent upset cannot re-enable an output that this call turned off.
 */
void hiram_actuator_inhibit(void);

/* DWT cycle counter, enabled by the console/DWT init in src/target_main.c. */
uint32_t hiram_dwt_cyccnt(void);

/*
 * Terminal safe state. Inhibits the actuators, reports `reason` best-effort on
 * the console, then parks re-asserting the inhibit. Does not return: a fault
 * handler that returned would resume the instruction that faulted.
 */
void hiram_safe_state_trap(const char *reason);

/* The DWT benchmark over hiram_policy_lut_step(), in
   benchmarks/dwt_bench_policy_lut.c. Runs once per call and prints its result
   on the console; the LUT must already be in service. */
void hiram_bench_run(void);

#ifdef __cplusplus
}
#endif

#endif /* HIRAM_BOARD_H */
