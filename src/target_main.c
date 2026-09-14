/*
 * target_main.c - STM32H723ZG (Cortex-M7) firmware image for the HIRAM kernel.
 *
 * Single owner of the target's startup path: vector table, Reset_Handler, the
 * TCM boot copies, clocking, console, MPU lockdown, the actuator safe state and
 * every fault handler. benchmarks/dwt_bench_policy_lut.c used to duplicate all
 * of that (B4); it is now a benchmark routine and nothing else, and the Makefile
 * links this file into the release artifact so the MPU lockdown is actually
 * present in the image that ships.
 *
 * NOT VALIDATED ON SILICON. Every register offset and field position below
 * carries the RM0468 (STM32H72x/H73x) or ARMv7-M ARM reference it was written
 * from; re-check them against the documents before trusting this on hardware.
 * The code is written so that a wrong constant fails toward a slower clock or a
 * stopped actuator rather than toward a hang or an over-clock.
 */
#include <stdint.h>
#include <stdbool.h>
#include "hiram_circuit_data.h"   /* defines circuit dimensions, then includes hiram_eval.h */
#include "hiram_policy_lut.h"
#include "hiram_board.h"

/* -----------------------------------------------------------------------------
 * RCC (RM0468 s8.7, base 0x5802_4400)
 * ----------------------------------------------------------------------------- */
#define RCC_BASE            0x58024400UL
#define RCC_CR              (*(volatile uint32_t *)(RCC_BASE + 0x000))
#define RCC_CFGR            (*(volatile uint32_t *)(RCC_BASE + 0x010))
#define RCC_D1CFGR          (*(volatile uint32_t *)(RCC_BASE + 0x018))
#define RCC_D2CFGR          (*(volatile uint32_t *)(RCC_BASE + 0x01C))
#define RCC_PLLCKSELR       (*(volatile uint32_t *)(RCC_BASE + 0x028))
#define RCC_PLLCFGR         (*(volatile uint32_t *)(RCC_BASE + 0x02C))
#define RCC_PLL1DIVR        (*(volatile uint32_t *)(RCC_BASE + 0x030))
#define RCC_AHB4ENR         (*(volatile uint32_t *)(RCC_BASE + 0x0E0))
#define RCC_APB1LENR        (*(volatile uint32_t *)(RCC_BASE + 0x0E8))
#define RCC_APB2ENR         (*(volatile uint32_t *)(RCC_BASE + 0x0F0))
#define RCC_APB4ENR         (*(volatile uint32_t *)(RCC_BASE + 0x0F4))

#define RCC_CR_HSION        (1u << 0)
#define RCC_CR_HSIRDY       (1u << 2)
#define RCC_CR_PLL1ON       (1u << 24)
#define RCC_CR_PLL1RDY      (1u << 25)

#define RCC_CFGR_SW_Msk     (7u << 0)
#define RCC_CFGR_SW_PLL1    (3u << 0)
#define RCC_CFGR_SWS_Pos    3u
#define RCC_CFGR_SWS_Msk    (7u << 3)

#define RCC_APB4ENR_SYSCFGEN (1u << 1)
#define RCC_APB2ENR_TIM1EN   (1u << 0)
#define RCC_APB2ENR_TIM8EN   (1u << 1)

/* -----------------------------------------------------------------------------
 * PWR (RM0468 s6.8, base 0x5802_4800) and the SYSCFG overdrive gate.
 *
 * On this part the D3CR voltage-scale field only reaches VOS1; VOS0 -- the
 * scale the >400 MHz range requires -- is unlocked separately by SYSCFG_PWRCR's
 * ODEN bit. hiram_clock_init() requests it, reads it back, and picks the PLL
 * profile from what the read-back actually reports rather than from what was
 * written (B5 root cause: the pre-audit code assumed 550 MHz unconditionally).
 * ----------------------------------------------------------------------------- */
#define PWR_BASE            0x58024800UL
#define PWR_D3CR            (*(volatile uint32_t *)(PWR_BASE + 0x018))
#define PWR_D3CR_VOS_Msk    (3u << 14)
#define PWR_D3CR_VOS_SCALE1 (3u << 14)  /* highest scale selectable from D3CR alone */
#define PWR_D3CR_VOSRDY     (1u << 13)

#define SYSCFG_BASE         0x58000400UL
#define SYSCFG_PWRCR        (*(volatile uint32_t *)(SYSCFG_BASE + 0x02C))
#define SYSCFG_PWRCR_ODEN   (1u << 0)

/* -----------------------------------------------------------------------------
 * Embedded Flash interface (RM0468 s4.9, base 0x5200_2000)
 * ----------------------------------------------------------------------------- */
#define FLASH_BASE          0x52002000UL
#define FLASH_ACR           (*(volatile uint32_t *)(FLASH_BASE + 0x000))
#define FLASH_ACR_LATENCY(ws)   (((uint32_t)(ws)) << 0)
#define FLASH_ACR_WRHIGHFREQ(v) (((uint32_t)(v)) << 4)

/* -----------------------------------------------------------------------------
 * GPIO / USART3 console (PD8 = USART3_TX AF7, PB0 = user LED)
 * ----------------------------------------------------------------------------- */
#define GPIOB_BASE          0x58020400UL
#define GPIOB_MODER         (*(volatile uint32_t *)(GPIOB_BASE + 0x000))
#define GPIOB_ODR           (*(volatile uint32_t *)(GPIOB_BASE + 0x014))

#define GPIOD_BASE          0x58020C00UL
#define GPIOD_MODER         (*(volatile uint32_t *)(GPIOD_BASE + 0x000))
#define GPIOD_OSPEEDR       (*(volatile uint32_t *)(GPIOD_BASE + 0x008))
#define GPIOD_AFRH          (*(volatile uint32_t *)(GPIOD_BASE + 0x024))

#define USART3_BASE         0x40004800UL
#define USART3_CR1          (*(volatile uint32_t *)(USART3_BASE + 0x000))
#define USART3_BRR          (*(volatile uint32_t *)(USART3_BASE + 0x00C))
#define USART3_ISR          (*(volatile uint32_t *)(USART3_BASE + 0x01C))
#define USART3_TDR          (*(volatile uint32_t *)(USART3_BASE + 0x028))
#define USART3_ISR_TXE      (1u << 7)
#define UART_BAUD           115200u

/* -----------------------------------------------------------------------------
 * Actuator command path (B5).
 *
 * TIM1 and TIM8 are the advanced-control timers whose complementary outputs
 * drive the traction inverter and the brake-pressure valve on this bring-up
 * board. They are the registers that must be at rest before anything else
 * happens in a fault reaction -- the pre-audit handlers printed a banner over a
 * blocking UART and spun forever with whatever duty cycle was last commanded
 * still being driven.
 *
 * Offsets are the standard STM32 advanced-control timer map (RM0468 s28.4).
 * ----------------------------------------------------------------------------- */
#define TIM1_BASE           0x40010000UL
#define TIM8_BASE           0x40010400UL
#define TIM_OFF_CR1         0x000u
#define TIM_OFF_CCER        0x020u
#define TIM_OFF_CCR1        0x034u
#define TIM_OFF_CCR2        0x038u
#define TIM_OFF_CCR3        0x03Cu
#define TIM_OFF_CCR4        0x040u
#define TIM_OFF_BDTR        0x044u

/* -----------------------------------------------------------------------------
 * Cortex-M7 core peripherals (ARMv7-M ARM B3.2 / C1.8)
 * ----------------------------------------------------------------------------- */
#define CoreDebug_DEMCR     (*(volatile uint32_t *)0xE000EDFCUL)
#define DEMCR_TRCENA        (1u << 24)
#define DWT_CTRL            (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT          (*(volatile uint32_t *)0xE0001004UL)
#define DWT_LAR             (*(volatile uint32_t *)0xE0001FB0UL)
#define DWT_LAR_UNLOCK      0xC5ACCE55UL
#define DWT_CTRL_CYCCNTENA  (1u << 0)

#define SCB_SHCSR           (*(volatile uint32_t *)0xE000ED24UL)
#define SCB_SHCSR_MEMFAULTENA (1u << 16)
#define SCB_SHCSR_BUSFAULTENA (1u << 17)
#define SCB_SHCSR_USGFAULTENA (1u << 18)

/* ARMv7-M Memory Protection Unit (ARMv7-M ARM B3.5). 8 regions on this part. */
#define MPU_TYPE            (*(volatile uint32_t *)0xE000ED90UL)
#define MPU_CTRL            (*(volatile uint32_t *)0xE000ED94UL)
#define MPU_RNR             (*(volatile uint32_t *)0xE000ED98UL)
#define MPU_RBAR            (*(volatile uint32_t *)0xE000ED9CUL)
#define MPU_RASR            (*(volatile uint32_t *)0xE000EDA0UL)

#define MPU_CTRL_ENABLE     (1u << 0)
#define MPU_CTRL_PRIVDEFENA (1u << 2)  /* unconfigured addresses keep the default privileged map */

/*
 * MPU_RASR field positions, ARMv7-M ARM B3.5.5 Table B3-13:
 *   [0]     ENABLE        [5:1]  SIZE        [15:8] SRD
 *   [16]    B             [17]   C           [18]   S
 *   [21:19] TEX           [26:24] AP         [28]   XN
 *
 * B3: MPU_RASR_NORMAL_WT was (1u << 20). Bit 20 is TEX[1], not the C bit, so
 * the descriptor encoded TEX=0b010, C=0, B=0 -- a *different memory type* from
 * the Normal/write-through the comment claimed, applied to ITCM and to the
 * policy LUT. The intended encoding is TEX=0b000, C=1, B=0, S=0: Normal
 * memory, write-through, no write-allocate, non-shareable, which is the type
 * the default background map already assigns Flash and TCM -- so locking these
 * regions changes permissions without also changing cacheability, which is the
 * whole point of doing it this way.
 */
#define MPU_RASR_ENABLE     (1u << 0)
#define MPU_RASR_SIZE(pow2_minus1) (((uint32_t)(pow2_minus1)) << 1)
#define MPU_RASR_B          (1u << 16)
#define MPU_RASR_C          (1u << 17)
#define MPU_RASR_S          (1u << 18)
#define MPU_RASR_TEX(v)     (((uint32_t)(v)) << 19)
#define MPU_RASR_AP_RO_RO   (0x6u << 24) /* privileged RO, unprivileged RO (Table B3-15) */
#define MPU_RASR_XN         (1u << 28)   /* Execute-Never: set on data, cleared on code */

/* TEX=000, C=1, B=0, S=0. Spelled out from the named fields rather than as a
   single magic constant, so the next reader checks three documented bits
   instead of trusting one number. */
#define MPU_RASR_NORMAL_WT  (MPU_RASR_TEX(0u) | MPU_RASR_C)

/* -----------------------------------------------------------------------------
 * Clock configuration (B5)
 *
 * Two profiles, both from the internal HSI so the measurement needs no external
 * oscillator and no board solder-bridge configuration. Which one is programmed
 * is decided by what the voltage-scale read-back reports, never by assumption:
 *
 *   VOS0 confirmed (SYSCFG_PWRCR.ODEN latched, VOSRDY set)
 *     HSI 64 MHz / DIVM1 8 =   8 MHz ref   (PLL1RGE 0b11, the 8-16 MHz range)
 *     8 MHz * DIVN1 68     = 544 MHz VCO   (wide VCO range, 192-836 MHz)
 *     544 MHz / DIVP1 1    = 544 MHz SYSCLK  (<= the 550 MHz part maximum)
 *     HPRE /2              = 272 MHz AXI/AHB (<= 275 MHz at VOS0)
 *     D2PPRE1 /2           = 136 MHz APB1    (<= 137.5 MHz at VOS0)
 *
 *   VOS1 only (overdrive refused or unavailable)
 *     HSI 64 MHz / DIVM1 8 =   8 MHz ref
 *     8 MHz * DIVN1 50     = 400 MHz VCO
 *     400 MHz / DIVP1 2    = 200 MHz SYSCLK
 *     HPRE /2              = 100 MHz AXI/AHB
 *     D2PPRE1 /2           =  50 MHz APB1
 *
 * The pre-audit clock_init() in the benchmark wrote RCC_PLLCKSELR = 0, which
 * sets DIVM1 = 0 and disables PLL1 outright (RM0468 s8.7.11), then polled
 * PLL1RDY in a loop it ignored the result of, then reported 550 MHz regardless.
 * Every wait below is bounded and every decision is taken on a read-back.
 * ----------------------------------------------------------------------------- */
#define HSI_HZ              64000000u
#define CLOCK_TIMEOUT       200000u

#define PLL_DIVM1           8u   /* both profiles: 64 MHz / 8 = 8 MHz PLL1 reference */
#define PLL_RGE_8_16MHZ     (3u << 2)
#define PLL_DIVP1EN         (1u << 16)

#define PROFILE_VOS0_DIVN1  68u
#define PROFILE_VOS0_DIVP1  1u
#define PROFILE_VOS1_DIVN1  50u
#define PROFILE_VOS1_DIVP1  2u

/* Flash wait states are set for the faster of the two profiles and left there.
   Excess wait states cost speed; too few at 272 MHz AXI is a bus fault, and a
   latency that depends on which branch was taken is one more thing that can be
   wrong in only one of them. */
#define FLASH_WAIT_STATES   4u
#define FLASH_WRHIGHFREQ    2u

static uint32_t    g_cpu_hz      = HSI_HZ;
static uint32_t    g_pclk1_hz    = HSI_HZ;
static const char *g_clock_note  = "HSI 64 MHz (PLL not engaged)";

uint32_t    hiram_cpu_hz(void)     { return g_cpu_hz; }
uint32_t    hiram_pclk1_hz(void)   { return g_pclk1_hz; }
const char *hiram_clock_note(void) { return g_clock_note; }

uint32_t hiram_dwt_cyccnt(void) { return DWT_CYCCNT; }

/* Bounded register poll. Returns false on expiry instead of spinning: a clock
   bring-up that cannot finish must report that it could not, so the caller can
   stay on the HSI and say so in the banner. */
static bool wait_bit(volatile uint32_t *reg, uint32_t mask, bool want_set) {
    uint32_t i;
    for (i = 0u; i < CLOCK_TIMEOUT; i++) {
        const bool is_set = ((*reg & mask) != 0u);
        if (is_set == want_set) {
            return true;
        }
    }
    return false;
}

/*
 * Requests the VOS0 overdrive and reports whether it was actually granted.
 *
 * The read-back is the load-bearing part. If SYSCFG_PWRCR.ODEN does not latch
 * -- wrong offset, gated SYSCFG clock, a package that does not offer overdrive
 * -- this returns false and the caller programs the VOS1 profile. The failure
 * direction is a 200 MHz part instead of a 544 MHz one, never a 544 MHz clock
 * on a core voltage that cannot sustain it.
 */
static bool request_overdrive_vos0(void) {
    RCC_APB4ENR |= RCC_APB4ENR_SYSCFGEN;
    (void)RCC_APB4ENR; /* RCC enable writes need a read-back before the peripheral is touched */

    SYSCFG_PWRCR |= SYSCFG_PWRCR_ODEN;
    if ((SYSCFG_PWRCR & SYSCFG_PWRCR_ODEN) == 0u) {
        return false;
    }
    return wait_bit(&PWR_D3CR, PWR_D3CR_VOSRDY, true);
}

hiram_clock_result_t hiram_clock_init(void) {
    uint32_t divn1;
    uint32_t divp1;
    uint32_t sysclk_hz;
    hiram_clock_result_t result;

    RCC_CR |= RCC_CR_HSION;
    if (!wait_bit(&RCC_CR, RCC_CR_HSIRDY, true)) {
        return HIRAM_CLOCK_HSI_FALLBACK;
    }

    /* Core voltage before core clock. Raising the clock first is the ordering
       that hangs the part, and it is not recoverable in software. */
    PWR_D3CR = (PWR_D3CR & ~PWR_D3CR_VOS_Msk) | PWR_D3CR_VOS_SCALE1;
    if (!wait_bit(&PWR_D3CR, PWR_D3CR_VOSRDY, true)) {
        return HIRAM_CLOCK_HSI_FALLBACK;
    }
    if ((PWR_D3CR & PWR_D3CR_VOS_Msk) != PWR_D3CR_VOS_SCALE1) {
        return HIRAM_CLOCK_HSI_FALLBACK;
    }

    if (request_overdrive_vos0()) {
        divn1     = PROFILE_VOS0_DIVN1;
        divp1     = PROFILE_VOS0_DIVP1;
        sysclk_hz = (HSI_HZ / PLL_DIVM1) * PROFILE_VOS0_DIVN1 / PROFILE_VOS0_DIVP1;
        result    = HIRAM_CLOCK_PLL_VOS0;
    } else {
        divn1     = PROFILE_VOS1_DIVN1;
        divp1     = PROFILE_VOS1_DIVP1;
        sysclk_hz = (HSI_HZ / PLL_DIVM1) * PROFILE_VOS1_DIVN1 / PROFILE_VOS1_DIVP1;
        result    = HIRAM_CLOCK_PLL_VOS1;
    }

    FLASH_ACR = FLASH_ACR_LATENCY(FLASH_WAIT_STATES) | FLASH_ACR_WRHIGHFREQ(FLASH_WRHIGHFREQ);

    /* PLL1 source HSI (PLLSRC 0b00), DIVM1 in [9:4]. Writing 0 here -- the
       pre-audit bug -- selects DIVM1 = 0, which disables the prescaler and the
       PLL with it. */
    RCC_PLLCKSELR = (PLL_DIVM1 << 4u);
    RCC_PLLCFGR   = PLL_RGE_8_16MHZ | PLL_DIVP1EN; /* PLL1VCOSEL 0 = wide; PLL1FRACEN 0 */
    /* DIVN1 in [8:0] and DIVP1 in [15:9], both stored as value-1. */
    RCC_PLL1DIVR  = ((divn1 - 1u) << 0u) | ((divp1 - 1u) << 9u);

    /* Bus dividers before the switch, so nothing is briefly overclocked. */
    RCC_D1CFGR = (8u << 0u);  /* HPRE /2 (0b1000); D1CPRE /1 */
    RCC_D2CFGR = (4u << 4u);  /* D2PPRE1 /2 (0b100) */

    RCC_CR |= RCC_CR_PLL1ON;
    if (!wait_bit(&RCC_CR, RCC_CR_PLL1RDY, true)) {
        return HIRAM_CLOCK_HSI_FALLBACK; /* no lock: the HSI is still driving the core */
    }

    RCC_CFGR = (RCC_CFGR & ~RCC_CFGR_SW_Msk) | RCC_CFGR_SW_PLL1;

    /* SWS is a 3-bit value, not a flag, so this polls for equality rather than
       for "any bit set" -- CSI (0b001) and HSE (0b010) would both satisfy a
       mask test against the PLL1 encoding (0b011) and report a switch that did
       not happen, at a rate four to eight times lower than the one the banner
       would then print. */
    {
        uint32_t i;
        bool switched = false;
        for (i = 0u; i < CLOCK_TIMEOUT; i++) {
            if (((RCC_CFGR & RCC_CFGR_SWS_Msk) >> RCC_CFGR_SWS_Pos) == RCC_CFGR_SW_PLL1) {
                switched = true;
                break;
            }
        }
        if (!switched) {
            return HIRAM_CLOCK_HSI_FALLBACK; /* switch did not take */
        }
    }

    g_cpu_hz   = sysclk_hz;
    g_pclk1_hz = sysclk_hz / 4u; /* HPRE /2 then D2PPRE1 /2 */
    g_clock_note = (result == HIRAM_CLOCK_PLL_VOS0)
        ? "PLL1 544 MHz from HSI (VOS0 overdrive confirmed)"
        : "PLL1 200 MHz from HSI (VOS1; overdrive not granted)";
    return result;
}

/* -----------------------------------------------------------------------------
 * Actuator safe state (B5)
 * ----------------------------------------------------------------------------- */
static void actuator_inhibit_timer(uint32_t base) {
    /* Order matters: the master output enable is the single bit that takes
       every channel off the pins, so it goes first and the rest is cleanup. */
    *(volatile uint32_t *)(base + TIM_OFF_BDTR) = 0u; /* MOE = 0: outputs forced inactive */
    *(volatile uint32_t *)(base + TIM_OFF_CCER) = 0u; /* all channel outputs disabled */
    *(volatile uint32_t *)(base + TIM_OFF_CCR1) = 0u; /* zero duty on every channel, so a */
    *(volatile uint32_t *)(base + TIM_OFF_CCR2) = 0u; /* later re-enable by anything cannot */
    *(volatile uint32_t *)(base + TIM_OFF_CCR3) = 0u; /* resurrect the last command */
    *(volatile uint32_t *)(base + TIM_OFF_CCR4) = 0u;
    *(volatile uint32_t *)(base + TIM_OFF_CR1)  = 0u; /* CEN = 0: counter stopped */
}

void hiram_actuator_inhibit(void) {
    /* The timer clocks are enabled first because a fault can arrive before
       hw_init() ran, and a write to a clock-gated APB peripheral does not
       reach the register. RCC itself is always clocked. */
    RCC_APB2ENR |= RCC_APB2ENR_TIM1EN | RCC_APB2ENR_TIM8EN;
    (void)RCC_APB2ENR;

    actuator_inhibit_timer(TIM1_BASE);
    actuator_inhibit_timer(TIM8_BASE);

    __asm__ volatile ("dsb" ::: "memory");
}

/* -----------------------------------------------------------------------------
 * Console
 * ----------------------------------------------------------------------------- */
#define UART_TXE_TIMEOUT    200000u

void hiram_uart_putc(char c) {
    uint32_t guard;
    for (guard = 0u; guard < UART_TXE_TIMEOUT; guard++) {
        if ((USART3_ISR & USART3_ISR_TXE) != 0u) {
            USART3_TDR = (uint32_t)(uint8_t)c;
            return;
        }
    }
    /* TXE never came -- no console, or one whose flow control is stuck. Drop
       the character. A diagnostic print must never be able to hold a fault
       reaction open indefinitely, which the pre-audit `while (!TXE) {}` did. */
}

void hiram_uart_puts(const char *s) {
    const char *p = s;
    while (*p != '\0') {
        if (*p == '\n') {
            hiram_uart_putc('\r');
        }
        hiram_uart_putc(*p);
        p++;
    }
}

void hiram_uart_print_u32(uint32_t val) {
    char buf[11];
    int i = 10;
    buf[i] = '\0';
    i--;
    if (val == 0u) {
        hiram_uart_putc('0');
        return;
    }
    while ((val > 0u) && (i >= 0)) {
        buf[i] = (char)('0' + (int)(val % 10u));
        i--;
        val /= 10u;
    }
    hiram_uart_puts(&buf[i + 1]);
}

void hiram_uart_print_u32_pad(uint32_t val, uint32_t width) {
    uint32_t div = 1u;
    uint32_t k;
    for (k = 1u; k < width; k++) {
        div *= 10u;
    }
    while (div > 0u) {
        const uint32_t digit = (val / div) % 10u;
        hiram_uart_putc((char)('0' + (int)digit));
        div /= 10u;
    }
}

/* Report a DWT cycle count as cycles, wall time, and share of the control
   frame. The harness used to print raw cycles only, which is meaningless
   without the clock they were counted at -- and the clock was not printed
   either, so a capture could be converted at the wrong rate with nothing in
   the log to contradict it. */
static void uart_print_timing(uint32_t cycles) {
    const uint64_t ns = ((uint64_t)cycles * 1000000000ULL) / (uint64_t)g_cpu_hz;
    const uint32_t us_int = (uint32_t)(ns / 1000ULL);
    const uint32_t us_frac = (uint32_t)(ns % 1000ULL);
    const uint32_t pct_x100 = (uint32_t)(ns / 250ULL); /* of a 2.5 ms / 400 Hz frame */

    hiram_uart_print_u32(cycles);
    hiram_uart_puts(" cyc = ");
    hiram_uart_print_u32(us_int);
    hiram_uart_putc('.');
    hiram_uart_print_u32_pad(us_frac, 3u);
    hiram_uart_puts(" us = ");
    hiram_uart_print_u32(pct_x100 / 100u);
    hiram_uart_putc('.');
    hiram_uart_print_u32_pad(pct_x100 % 100u, 2u);
    hiram_uart_puts("% of frame");
}

static void hw_init(void) {
    /* GPIOB, GPIOD and USART3 clocks */
    RCC_AHB4ENR  |= (1u << 1) | (1u << 3); /* GPIOBEN, GPIODEN */
    RCC_APB1LENR |= (1u << 18);            /* USART3EN */
    (void)RCC_APB1LENR;

    /* PB0 (user LED, green) as general-purpose output */
    GPIOB_MODER &= ~(3u << 0);
    GPIOB_MODER |=  (1u << 0);

    /* PD8 (USART3_TX) to alternate function 7 */
    GPIOD_MODER   &= ~(3u << 16);
    GPIOD_MODER   |=  (2u << 16);
    GPIOD_OSPEEDR |=  (2u << 16);
    GPIOD_AFRH    &= ~(0xFu << 0);
    GPIOD_AFRH    |=  (7u << 0);

    /* Divisor follows whatever hiram_clock_init() actually achieved, so the
       console stays readable on the PLL or on the HSI fallback. */
    USART3_CR1 = 0u;
    USART3_BRR = (g_pclk1_hz + (UART_BAUD / 2u)) / UART_BAUD;
    USART3_CR1 = (1u << 0) | (1u << 3); /* UE, TE */

    /* Configurable faults to their own handlers rather than escalating to
       HardFault: an MPU permission violation on the locked LUT should be
       reported as the memory-management fault it is. */
    SCB_SHCSR |= SCB_SHCSR_MEMFAULTENA | SCB_SHCSR_BUSFAULTENA | SCB_SHCSR_USGFAULTENA;

    /* DWT cycle counter */
    CoreDebug_DEMCR |= DEMCR_TRCENA;
    DWT_LAR = DWT_LAR_UNLOCK;
    DWT_CYCCNT = 0u;
    DWT_CTRL |= DWT_CTRL_CYCCNTENA;
}

static void delay_ms(volatile uint32_t count) {
    while (count-- != 0u) {
        volatile uint32_t i;
        for (i = 0u; i < 8000u; i++) {
            __asm__ volatile ("nop");
        }
    }
}

/* -----------------------------------------------------------------------------
 * Terminal safe state (B5)
 *
 * The loop does not terminate, and that is the intended behaviour: returning
 * from a fault handler resumes the instruction that faulted. What the audit
 * required, and what this provides, is that every operation *inside* the
 * reaction is bounded and that the actuators are commanded off before anything
 * that can block -- so the reaction cannot be delayed by a stuck console, and
 * an upset that re-enables an output is corrected on the next pass rather than
 * persisting until someone notices.
 * ----------------------------------------------------------------------------- */
void hiram_safe_state_trap(const char *reason) {
    hiram_actuator_inhibit(); /* first, before any output */

    hiram_uart_puts("\n!! HIRAM SAFE STATE: ");
    hiram_uart_puts(reason);
    hiram_uart_puts("\n!! Actuator command registers zeroed; outputs inhibited.\n");

    for (;;) {
        hiram_actuator_inhibit();
        GPIOB_ODR ^= (1u << 0); /* fast toggle distinguishes this from the 1 Hz run loop */
        delay_ms(100u);
    }
}

/* -----------------------------------------------------------------------------
 * MPU lockdown (Task 3.3)
 *
 * Called once, after the ITCM/DTCM boot copy has finished and the LUT has been
 * brought into service -- locking a region before it is fully written would
 * fault on the copy itself. Two regions, both Read-Only for every privilege
 * level from this point on:
 *
 *   Region 0: the whole 64 KB ITCM (.itcm_text lives inside it). Execution
 *             stays allowed (XN=0); only writes are blocked, so a wild pointer
 *             cannot patch the evaluator that flight-critical code depends on.
 *   Region 1: the 256 B .dtcm_lut region (HIRAM_POLICY_LUT, padded/aligned for
 *             exactly this by stm32h723zg.ld). Execute-Never (XN=1).
 *
 * This is the hardware half of the LUT's protection; the dual-rail decode in
 * hiram_policy_lut_step() is the other half, and covers the corruption the MPU
 * cannot (an upset in the cell itself rather than a write to it).
 *
 * Everything else in DTCM (stack, .bss, .data, the .dtcm_data HiramContext
 * scratch region) is left to the MPU's default background map via PRIVDEFENA.
 *
 * Not static, and named in the Makefile's verify_release_image check: B4 was a
 * build-system defect, where this function existed but was not linked into the
 * artifact that shipped. A symbol the release check can look for is what makes
 * that non-regressable.
 * ----------------------------------------------------------------------------- */
extern uint32_t _itcm_start, _itcm_end;
extern uint32_t _dtcm_lut_start, _dtcm_lut_end;

void hiram_mpu_lock_regions(void) {
    /* Region 0: ITCM, 64 KB, Read-Only + Executable. 64 KB = 2^16, so
       SIZE = 16 - 1 = 15. Base 0x00000000 is trivially 64 KB aligned. */
    MPU_RNR  = 0u;
    MPU_RBAR = (uint32_t)&_itcm_start;
    MPU_RASR = MPU_RASR_AP_RO_RO | MPU_RASR_NORMAL_WT
             | MPU_RASR_SIZE(15u) | MPU_RASR_ENABLE;

    /* Region 1: .dtcm_lut, 256 B, Read-Only + Execute-Never. 256 B = 2^8, so
       SIZE = 8 - 1 = 7. stm32h723zg.ld 256-aligns _dtcm_lut_start so a single
       region covers it exactly. */
    MPU_RNR  = 1u;
    MPU_RBAR = (uint32_t)&_dtcm_lut_start;
    MPU_RASR = MPU_RASR_AP_RO_RO | MPU_RASR_XN | MPU_RASR_NORMAL_WT
             | MPU_RASR_SIZE(7u) | MPU_RASR_ENABLE;

    __asm__ volatile ("dsb" ::: "memory");
    MPU_CTRL = MPU_CTRL_ENABLE | MPU_CTRL_PRIVDEFENA;
    __asm__ volatile ("dsb" ::: "memory");
    __asm__ volatile ("isb" ::: "memory");
}

/* -----------------------------------------------------------------------------
 * Target entry point
 * ----------------------------------------------------------------------------- */
__attribute__((section(".dtcm_data"), aligned(8)))
static HiramContext g_hiram_ctx;

void target_main(void) {
    Rational thresh = {5LL, 100LL}; /* 5% safety veto threshold */
    uint32_t frame_count = 1u;

    (void)hiram_clock_init();
    hw_init();

    /* Before anything else can command motion. The reset state of the timers
       is already inactive, but "already" is an assumption about the last reset
       cause, not a guarantee about this one. */
    hiram_actuator_inhibit();

    hiram_context_init(&g_hiram_ctx);

    /* The LUT was copied from Flash to DTCM by Reset_Handler. Bring it into
       service -- CRC32 plus a dual-rail sweep of all 64 cells -- before
       anything can call hiram_policy_lut_step(). Until this succeeds, the
       evaluator reports EMERGENCY_BRAKE on every call, so the failure is
       fail-safe rather than fail-silent; the trap below makes it fail-stop.
       This must precede hiram_mpu_lock_regions(). */
    if (!hiram_policy_lut_init()) {
        hiram_safe_state_trap("policy LUT failed its startup integrity check");
    }
    hiram_mpu_lock_regions();

    hiram_uart_puts("\n\n================================================================\n");
    hiram_uart_puts("  HIRAM FLIGHT SAFETY KERNEL - STM32H723ZG (CORTEX-M7)\n");
    hiram_uart_puts("  Zero-Heap MISRA-C Safety Monitor Online | DWT Active\n");
    hiram_uart_puts("================================================================\n");
    hiram_uart_puts("  Clock  : ");
    hiram_uart_puts(hiram_clock_note());
    hiram_uart_puts("\n  CPU    : ");
    hiram_uart_print_u32(g_cpu_hz / 1000000u);
    hiram_uart_puts(" MHz  (DWT_CYCCNT counts at this rate)\n");
    hiram_uart_puts("  Frame  : 400 Hz = 2500 us = ");
    hiram_uart_print_u32(g_cpu_hz / 400u);
    hiram_uart_puts(" cycles\n");
    hiram_uart_puts("  MPU    : ITCM + .dtcm_lut locked read-only (");
    hiram_uart_print_u32((MPU_TYPE >> 8u) & 0xFFu); /* MPU_TYPE.DREGION [15:8] */
    hiram_uart_puts(" regions available)\n");
    hiram_uart_puts("  LUT    : dual-rail, operational\n");
    hiram_uart_puts("================================================================\n\n");

    /* The DWT benchmark now runs inside the one firmware image, against the
       same MPU-locked, integrity-checked LUT the audit loop uses -- rather
       than from a second image with its own startup path and no MPU. */
    hiram_bench_run();

    while (1) {
        GPIOB_ODR ^= (1u << 0);

        hiram_uart_puts("--- [FRAME ");
        hiram_uart_print_u32(frame_count);
        frame_count++;
        hiram_uart_puts("] ---------------------------------------------\n");

        /* [TEST 1] Nominal Flight Conditions */
        Evidence ev1;
        hiram_evidence_init(&ev1);
        hiram_evidence_set(&ev1, VAR_TERRAIN_CLEAR, true);
        hiram_evidence_set(&ev1, VAR_ALTITUDE_STABLE, true);

        uint32_t t0 = DWT_CYCCNT;
        HiramAuditReport rep1 = hiram_audit_hazard(&ev1, thresh, &g_hiram_ctx);
        uint32_t t1 = DWT_CYCCNT;

        hiram_uart_puts("Test 1 (Nominal):      Decision = ");
        hiram_uart_puts(rep1.decision == HIRAM_APPROVED ? "APPROVED" : "VETOED");
        hiram_uart_puts("  | ");
        uart_print_timing(t1 - t0);
        hiram_uart_puts("\n");

        /* [TEST 2] Severe Stall Hazard Injected */
        Evidence ev2;
        hiram_evidence_init(&ev2);
        hiram_evidence_set(&ev2, VAR_LOW_AIRSPEED, true);
        hiram_evidence_set(&ev2, VAR_HIGH_ANGLE_OF_ATTACK, true);

        t0 = DWT_CYCCNT;
        HiramAuditReport rep2 = hiram_audit_hazard(&ev2, thresh, &g_hiram_ctx);
        t1 = DWT_CYCCNT;

        hiram_uart_puts("Test 2 (Stall Hazard): Decision = ");
        hiram_uart_puts(rep2.decision == HIRAM_VETOED_SAFETY_VIOLATION ? "VETOED" : "APPROVED");
        hiram_uart_puts("    | ");
        uart_print_timing(t1 - t0);
        hiram_uart_puts("\n");

        /* [TEST 3] Sensor Contradiction Injected */
        Evidence ev3;
        hiram_evidence_init(&ev3);
        hiram_evidence_set(&ev3, VAR_LOW_AIRSPEED, true);
        hiram_evidence_set(&ev3, VAR_HIGH_ANGLE_OF_ATTACK, true);
        hiram_evidence_set(&ev3, HAZARD_VAR_ID, false);

        t0 = DWT_CYCCNT;
        HiramAuditReport rep3 = hiram_audit_hazard(&ev3, thresh, &g_hiram_ctx);
        t1 = DWT_CYCCNT;

        hiram_uart_puts("Test 3 (Contradiction):Decision = ");
        hiram_uart_puts(rep3.decision == HIRAM_REJECTED_CONTRADICTION ? "REJECTED" : "UNEXPECTED");
        hiram_uart_puts("  | ");
        uart_print_timing(t1 - t0);
        hiram_uart_puts("\n\n");

        /* A step whose LUT has been latched out of service by a dual-rail
           violation since the last frame must not be flown on. */
        if (!hiram_policy_lut_is_operational()) {
            hiram_safe_state_trap("policy LUT latched out of service (dual-rail parity violation)");
        }

        delay_ms(1000u); /* 1-second cadence between audit frames */
    }
}

/* -----------------------------------------------------------------------------
 * Vector table and startup
 * ----------------------------------------------------------------------------- */
extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern uint32_t _itcm_load;
extern uint32_t _dtcm_lut_load;

void Reset_Handler(void) {
    /* Copy initialized .data from Flash to DTCM RAM */
    uint32_t *pSrc = &_sidata;
    uint32_t *pDst = &_sdata;
    while (pDst < &_edata) {
        *pDst++ = *pSrc++;
    }

    /* Zero out .bss in DTCM RAM */
    pDst = &_sbss;
    while (pDst < &_ebss) {
        *pDst++ = 0;
    }

    /* Task 3.3: copy hiram_policy_lut_step() from Flash into Instruction TCM
       before it is ever called. Nothing verifies the *code* copy beyond the
       link-time ASSERT that .itcm_text fits ITCM; a torn copy would crash on
       first call, which is a detectable, fail-stop outcome on this target. */
    pSrc = &_itcm_load;
    pDst = &_itcm_start;
    while (pDst < &_itcm_end) {
        *pDst++ = *pSrc++;
    }

    /* Task 3.3: copy HIRAM_POLICY_LUT from Flash into Data TCM. Unlike
       .dtcm_data (self-initialising scratch, NOLOAD), this section has real
       content that must reach RAM before hiram_policy_lut_init() runs in
       target_main() -- that check is what catches a torn or corrupted copy. */
    pSrc = &_dtcm_lut_load;
    pDst = &_dtcm_lut_start;
    while (pDst < &_dtcm_lut_end) {
        *pDst++ = *pSrc++;
    }

    target_main();

    /* target_main() does not return. If it somehow does, that is itself a
       control-flow fault, and the actuators go off rather than being left
       wherever the last frame put them. */
    hiram_safe_state_trap("target_main returned");
}

/*
 * B5: every trap commands the actuator safe state before it reports anything.
 * Distinct handlers rather than one Default_Handler so the console names the
 * fault -- a MemManage on this image most likely means something tried to
 * write the MPU-locked policy LUT, which is a different diagnosis from a
 * BusFault.
 */
void NMI_Handler(void)        { hiram_safe_state_trap("NMI"); }
void HardFault_Handler(void)  { hiram_safe_state_trap("HardFault"); }
void MemManage_Handler(void)  { hiram_safe_state_trap("MemManage (MPU permission violation)"); }
void BusFault_Handler(void)   { hiram_safe_state_trap("BusFault"); }
void UsageFault_Handler(void) { hiram_safe_state_trap("UsageFault"); }
void Default_Handler(void)    { hiram_safe_state_trap("unexpected exception"); }

__attribute__((section(".isr_vector"), used))
const uint32_t vector_table[] = {
    (uint32_t)&_estack,
    (uint32_t)Reset_Handler,
    (uint32_t)NMI_Handler,
    (uint32_t)HardFault_Handler,
    (uint32_t)MemManage_Handler,
    (uint32_t)BusFault_Handler,
    (uint32_t)UsageFault_Handler,
    0, 0, 0, 0,                /* Reserved */
    (uint32_t)Default_Handler, /* SVCall */
    (uint32_t)Default_Handler, /* DebugMon */
    0,                         /* Reserved */
    (uint32_t)Default_Handler, /* PendSV */
    (uint32_t)Default_Handler, /* SysTick */
};
