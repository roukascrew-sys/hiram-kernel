#include <stdint.h>
#include <stdbool.h>
#include "hiram_circuit_data.h"   /* defines circuit dimensions, then includes hiram_eval.h */

/* -----------------------------------------------------------------------------
 * STM32H723ZG Hardware Register Addresses
 * ----------------------------------------------------------------------------- */
#define RCC_BASE            0x58024400UL
#define RCC_AHB4ENR         (*(volatile uint32_t *)(RCC_BASE + 0x0E0))
#define RCC_APB1LENR        (*(volatile uint32_t *)(RCC_BASE + 0x0E8))

#define GPIOB_BASE          0x58020400UL
#define GPIOB_MODER         (*(volatile uint32_t *)(GPIOB_BASE + 0x000))
#define GPIOB_ODR           (*(volatile uint32_t *)(GPIOB_BASE + 0x014))

#define GPIOD_BASE          0x58020C00UL
#define GPIOD_MODER         (*(volatile uint32_t *)(GPIOD_BASE + 0x000))
#define GPIOD_OSPEEDR       (*(volatile uint32_t *)(GPIOD_BASE + 0x008))
#define GPIOD_AFRH          (*(volatile uint32_t *)(GPIOD_BASE + 0x024))

/* Clock tree. Every value below is marked with the register field it drives so
   it can be checked against RM0468 (STM32H72x/H73x) without reading the code.
   None of it has been executed on hardware -- see clock_init(). */
#define RCC_CR              (*(volatile uint32_t *)(RCC_BASE + 0x000))
#define RCC_CFGR            (*(volatile uint32_t *)(RCC_BASE + 0x010))
#define RCC_D1CFGR          (*(volatile uint32_t *)(RCC_BASE + 0x018))
#define RCC_D2CFGR          (*(volatile uint32_t *)(RCC_BASE + 0x01C))
#define RCC_PLLCKSELR       (*(volatile uint32_t *)(RCC_BASE + 0x028))
#define RCC_PLLCFGR         (*(volatile uint32_t *)(RCC_BASE + 0x02C))
#define RCC_PLL1DIVR        (*(volatile uint32_t *)(RCC_BASE + 0x030))

#define PWR_BASE            0x58024800UL
#define PWR_D3CR            (*(volatile uint32_t *)(PWR_BASE + 0x018))

#define FLASH_BASE          0x52002000UL
#define FLASH_ACR           (*(volatile uint32_t *)(FLASH_BASE + 0x000))

#define USART3_BASE         0x40004800UL
#define USART3_CR1          (*(volatile uint32_t *)(USART3_BASE + 0x000))
#define USART3_BRR          (*(volatile uint32_t *)(USART3_BASE + 0x00C))
#define USART3_ISR          (*(volatile uint32_t *)(USART3_BASE + 0x01C))
#define USART3_TDR          (*(volatile uint32_t *)(USART3_BASE + 0x028))

/* Cortex-M7 Core Debug & DWT Registers */
#define CoreDebug_DEMCR     (*(volatile uint32_t *)0xE000EDFCUL)
#define DWT_CTRL            (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT          (*(volatile uint32_t *)0xE0001004UL)
#define DWT_LAR             (*(volatile uint32_t *)0xE0001FB0UL) /* Software Lock Access */

/* -----------------------------------------------------------------------------
 * Clock configuration
 *
 * The firmware previously ran on the 64 MHz HSI default with no PLL setup at
 * all, so every DWT figure it produced was at 64 MHz -- not the 550 MHz the part
 * is capable of, and not a configuration anyone would fly. Converting those
 * cycle counts at the headline rate would have understated latency by 8.6x.
 *
 * Target here is a deliberately conservative 200 MHz CPU clock from the HSI, so
 * the measurement needs no external oscillator and no board solder-bridge
 * configuration.
 *
 *   HSI 64 MHz / DIVM1 8 = 8 MHz reference  (PLL1RGE = 8-16 MHz range)
 *   8 MHz * DIVN1 50     = 400 MHz VCO      (wide VCO range, 192-836 MHz)
 *   400 MHz / DIVP1 2    = 200 MHz SYSCLK
 *   D1CPRE /1            = 200 MHz CPU  (what DWT_CYCCNT counts)
 *   HPRE   /2            = 100 MHz AHB
 *   D2PPRE1 /2           =  50 MHz APB1 (USART3 kernel clock)
 *
 * FAIL-SAFE: every wait is bounded, and any timeout leaves the part on the HSI
 * and reports it over UART. A wrong constant here therefore costs a slower,
 * clearly-labelled measurement rather than a board that will not boot -- which
 * matters because none of this has been run on hardware.
 *
 * NOT VALIDATED ON SILICON. Check the register offsets and field positions
 * against RM0468 before trusting a 200 MHz figure; if the PLL fails to lock the
 * banner will say so and the numbers remain valid at 64 MHz.
 * ----------------------------------------------------------------------------- */
#define HSI_HZ              64000000u
#define PLL_DIVM1           8u      /* RCC_PLLCKSELR DIVM1  */
#define PLL_DIVN1           50u     /* RCC_PLL1DIVR  DIVN1  */
#define PLL_DIVP1           2u      /* RCC_PLL1DIVR  DIVP1  */
#define TARGET_SYSCLK_HZ    ((HSI_HZ / PLL_DIVM1) * PLL_DIVN1 / PLL_DIVP1)
#define CLOCK_TIMEOUT       200000u

static uint32_t g_cpu_hz  = HSI_HZ;
static uint32_t g_pclk1_hz = HSI_HZ;
static const char *g_clock_note = "HSI 64 MHz (PLL not engaged)";

static bool wait_bit(volatile uint32_t *reg, uint32_t mask, bool want_set) {
    for (uint32_t i = 0u; i < CLOCK_TIMEOUT; i++) {
        bool is_set = ((*reg & mask) != 0u);
        if (is_set == want_set) { return true; }
    }
    return false;
}

static void clock_init(void) {
    /* Highest voltage scale first: raising the clock before the core voltage
       is the ordering that hangs the part. VOS1 = 0b11 in PWR_D3CR[15:14]. */
    PWR_D3CR |= (3u << 14);
    if (!wait_bit(&PWR_D3CR, (1u << 13), true)) { return; }   /* VOSRDY */

    /* Flash latency is set generously on purpose. Too many wait states only
       costs speed; too few at 200 MHz is a bus fault. 4 WS, WRHIGHFREQ 0b10. */
    FLASH_ACR = (4u << 0) | (2u << 4);

    if (!wait_bit(&RCC_CR, (1u << 2), true)) { return; }       /* HSIRDY */

    /* PLL1 source HSI (PLLSRC 0b00), DIVM1 in bits [9:4]. */
    RCC_PLLCKSELR = (PLL_DIVM1 << 4);

    /* PLL1RGE 0b11 (8-16 MHz input), PLL1VCOSEL 0 (wide), DIVP1 output on. */
    RCC_PLLCFGR = (3u << 2) | (1u << 16);

    /* DIVN1 in [8:0] and DIVP1 in [15:9], both stored as value-1. */
    RCC_PLL1DIVR = ((PLL_DIVN1 - 1u) << 0) | ((PLL_DIVP1 - 1u) << 9);

    /* Bus dividers before the switch, so nothing is overclocked mid-flight. */
    RCC_D1CFGR = (8u << 0);      /* HPRE /2 (0b1000); D1CPRE /1 */
    RCC_D2CFGR = (4u << 4);      /* D2PPRE1 /2 (0b100) */

    RCC_CR |= (1u << 24);                                     /* PLL1ON */
    if (!wait_bit(&RCC_CR, (1u << 25), true)) { return; }      /* PLL1RDY */

    RCC_CFGR = (RCC_CFGR & ~7u) | 3u;                         /* SW = PLL1 */
    for (uint32_t i = 0u; i < CLOCK_TIMEOUT; i++) {
        if (((RCC_CFGR >> 3) & 7u) == 3u) {                   /* SWS == PLL1 */
            g_cpu_hz = TARGET_SYSCLK_HZ;
            g_pclk1_hz = TARGET_SYSCLK_HZ / 4u;               /* HPRE/2, D2PPRE1/2 */
            g_clock_note = "PLL1 200 MHz from HSI";
            return;
        }
    }
    /* Switch did not take: the HSI is still driving the core. */
}

/* -----------------------------------------------------------------------------
 * Serial Output (USART3: PD8 = TX, 115200 Baud, divisor derived from PCLK1)
 * ----------------------------------------------------------------------------- */
static void hw_init(void) {
    /* 1. Enable GPIOB, GPIOD, and USART3 clocks */
    RCC_AHB4ENR |= (1u << 1) | (1u << 3); /* GPIOBEN, GPIODEN */
    RCC_APB1LENR |= (1u << 18);           /* USART3EN */

    /* 2. Configure PB0 (User LED1 - Green) as General Output */
    GPIOB_MODER &= ~(3u << 0);
    GPIOB_MODER |=  (1u << 0);

    /* 3. Configure PD8 (USART3_TX) to Alternate Function 7 */
    GPIOD_MODER   &= ~(3u << 16);
    GPIOD_MODER   |=  (2u << 16);         /* Alternate function mode */
    GPIOD_OSPEEDR |=  (2u << 16);         /* High speed */
    GPIOD_AFRH    &= ~(0xFu << 0);
    GPIOD_AFRH    |=  (7u << 0);          /* AF7 = USART3_TX */

    /* 4. Configure USART3. The divisor follows whatever clock_init() actually
          achieved, so the console stays readable on either the PLL or the HSI
          fallback rather than assuming 64 MHz. */
    USART3_CR1 = 0;
    USART3_BRR = g_pclk1_hz / 115200u;
    USART3_CR1 = (1u << 0) | (1u << 3);   /* UE (Enable), TE (Transmitter) */

    /* 5. Enable Cortex-M7 DWT Cycle Counter */
    CoreDebug_DEMCR |= (1u << 24);        /* Enable trace */
    DWT_LAR = 0xC5ACCE55;                 /* Unlock DWT access */
    DWT_CYCCNT = 0;
    DWT_CTRL |= (1u << 0);                /* Enable CYCCNT */
}

static void uart_putc(char c) {
    while (!(USART3_ISR & (1u << 7))) {}  /* Wait for TXE (Transmit Empty) */
    USART3_TDR = (uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) {
        if (*s == '\n') uart_putc('\r');
        uart_putc(*s++);
    }
}

static void uart_print_u32(uint32_t val) {
    char buf[11];
    int i = 10;
    buf[i--] = '\0';
    if (val == 0) {
        uart_putc('0');
        return;
    }
    while (val > 0 && i >= 0) {
        buf[i--] = (char)('0' + (val % 10));
        val /= 10;
    }
    uart_puts(&buf[i + 1]);
}

static void uart_print_u32_pad(uint32_t val, uint32_t width) {
    uint32_t div = 1u;
    for (uint32_t k = 1u; k < width; k++) { div *= 10u; }
    while (div > 0u) {
        uint32_t digit = (val / div) % 10u;
        uart_putc((char)('0' + (int)digit));
        div /= 10u;
    }
}

/* Report a DWT cycle count as cycles, wall time, and share of the control
   frame. The harness used to print raw cycles only, which is meaningless
   without the clock they were counted at -- and the clock was not printed
   either, so a capture could be converted at the wrong rate with nothing in
   the log to contradict it. */
static void uart_print_timing(uint32_t cycles) {
    uint64_t ns = ((uint64_t)cycles * 1000000000ULL) / (uint64_t)g_cpu_hz;
    uint32_t us_int = (uint32_t)(ns / 1000ULL);
    uint32_t us_frac = (uint32_t)(ns % 1000ULL);
    uint32_t pct_x100 = (uint32_t)(ns / 250ULL);   /* of a 2.5 ms / 400 Hz frame */

    uart_print_u32(cycles);
    uart_puts(" cyc = ");
    uart_print_u32(us_int);
    uart_putc('.');
    uart_print_u32_pad(us_frac, 3u);
    uart_puts(" us = ");
    uart_print_u32(pct_x100 / 100u);
    uart_putc('.');
    uart_print_u32_pad(pct_x100 % 100u, 2u);
    uart_puts("% of frame");
}


static void delay_ms(volatile uint32_t count) {
    while (count--) {
        for (volatile uint32_t i = 0; i < 8000; i++) {
            __asm__ volatile ("nop");
        }
    }
}

/* -----------------------------------------------------------------------------
 * Target Entry Point & Flight Scenarios
 * ----------------------------------------------------------------------------- */
__attribute__((section(".dtcm_data"), aligned(8)))
static HiramContext g_hiram_ctx;

void target_main(void) {
    clock_init();
    hw_init();
    hiram_context_init(&g_hiram_ctx);

    uart_puts("\n\n================================================================\n");
    uart_puts("  HIRAM FLIGHT SAFETY KERNEL - STM32H723ZG (CORTEX-M7)\n");
    uart_puts("  Zero-Heap MISRA-C Safety Monitor Online | DWT Active\n");
    uart_puts("================================================================\n");
    uart_puts("  Clock  : ");
    uart_puts(g_clock_note);
    uart_puts("\n  CPU    : ");
    uart_print_u32(g_cpu_hz / 1000000u);
    uart_puts(" MHz  (DWT_CYCCNT counts at this rate)\n");
    uart_puts("  Frame  : 400 Hz = 2500 us = ");
    uart_print_u32(g_cpu_hz / 400u);
    uart_puts(" cycles\n");
    uart_puts("================================================================\n\n");

    Rational thresh = {5LL, 100LL}; /* 5% safety veto threshold */
    uint32_t frame_count = 1;

    while (1) {
        /* Toggle Green User LED */
        GPIOB_ODR ^= (1u << 0);

        uart_puts("--- [FRAME ");
        uart_print_u32(frame_count++);
        uart_puts("] ---------------------------------------------\n");

        /* [TEST 1] Nominal Flight Conditions */
        Evidence ev1;
        hiram_evidence_init(&ev1);
        hiram_evidence_set(&ev1, VAR_TERRAIN_CLEAR, true);
        hiram_evidence_set(&ev1, VAR_ALTITUDE_STABLE, true);

        uint32_t t0 = DWT_CYCCNT;
        HiramAuditReport rep1 = hiram_audit_hazard(&ev1, thresh, &g_hiram_ctx);
        uint32_t t1 = DWT_CYCCNT;
        uint32_t cycles1 = t1 - t0;

        uart_puts("Test 1 (Nominal):      Decision = ");
        uart_puts(rep1.decision == HIRAM_APPROVED ? "APPROVED" : "VETOED");
        uart_puts("  | ");
        uart_print_timing(cycles1);
        uart_puts("\n");

        /* [TEST 2] Severe Stall Hazard Injected */
        Evidence ev2;
        hiram_evidence_init(&ev2);
        hiram_evidence_set(&ev2, VAR_LOW_AIRSPEED, true);
        hiram_evidence_set(&ev2, VAR_HIGH_ANGLE_OF_ATTACK, true);

        t0 = DWT_CYCCNT;
        HiramAuditReport rep2 = hiram_audit_hazard(&ev2, thresh, &g_hiram_ctx);
        t1 = DWT_CYCCNT;
        uint32_t cycles2 = t1 - t0;

        uart_puts("Test 2 (Stall Hazard): Decision = ");
        uart_puts(rep2.decision == HIRAM_VETOED_SAFETY_VIOLATION ? "VETOED" : "APPROVED");
        uart_puts("    | ");
        uart_print_timing(cycles2);
        uart_puts("\n");

        /* [TEST 3] Sensor Contradiction Injected */
        Evidence ev3;
        hiram_evidence_init(&ev3);
        hiram_evidence_set(&ev3, VAR_LOW_AIRSPEED, true);
        hiram_evidence_set(&ev3, VAR_HIGH_ANGLE_OF_ATTACK, true);
        hiram_evidence_set(&ev3, HAZARD_VAR_ID, false);

        t0 = DWT_CYCCNT;
        HiramAuditReport rep3 = hiram_audit_hazard(&ev3, thresh, &g_hiram_ctx);
        t1 = DWT_CYCCNT;
        uint32_t cycles3 = t1 - t0;

        uart_puts("Test 3 (Contradiction):Decision = ");
        uart_puts(rep3.decision == HIRAM_REJECTED_CONTRADICTION ? "REJECTED" : "UNEXPECTED");
        uart_puts("  | ");
        uart_print_timing(cycles3);
        uart_puts("\n\n");

        delay_ms(1000); /* 1-second cadence between audit frames */
    }
}

/* -----------------------------------------------------------------------------
 * Minimal Vector Table & Startup Routine
 * ----------------------------------------------------------------------------- */
extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;

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

    /* Jump to hardware harness */
    target_main();
    while (1);
}

void Default_Handler(void) {
    while (1);
}

__attribute__((section(".isr_vector"), used))
const uint32_t vector_table[] = {
    (uint32_t)&_estack,
    (uint32_t)Reset_Handler,
    (uint32_t)Default_Handler, /* NMI */
    (uint32_t)Default_Handler, /* HardFault */
    (uint32_t)Default_Handler, /* MemManage */
    (uint32_t)Default_Handler, /* BusFault */
    (uint32_t)Default_Handler, /* UsageFault */
    0, 0, 0, 0,                /* Reserved */
    (uint32_t)Default_Handler, /* SVCall */
    (uint32_t)Default_Handler, /* DebugMon */
    0,                         /* Reserved */
    (uint32_t)Default_Handler, /* PendSV */
    (uint32_t)Default_Handler, /* SysTick */
};