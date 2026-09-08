#include <stdint.h>
#include <stdbool.h>
#include "hiram_eval.h"
#include "hiram_circuit_data.h"

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
 * Serial Output (USART3: PD8 = TX, 115200 Baud @ 64 MHz HSI Default Clock)
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

    /* 4. Configure USART3: 64,000,000 / 115,200 = 556 (0x022C) */
    USART3_CR1 = 0;
    USART3_BRR = 556;
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
void target_main(void) {
    hw_init();

    uart_puts("\n\n================================================================\n");
    uart_puts("  HIRAM FLIGHT SAFETY KERNEL - STM32H723ZG (CORTEX-M7)\n");
    uart_puts("  Zero-Heap MISRA-C Safety Monitor Online | DWT Active\n");
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
        HiramAuditReport rep1 = hiram_audit_hazard(&ev1, thresh);
        uint32_t t1 = DWT_CYCCNT;
        uint32_t cycles1 = t1 - t0;

        uart_puts("Test 1 (Nominal):      Decision = ");
        uart_puts(rep1.decision == HIRAM_APPROVED ? "APPROVED" : "VETOED");
        uart_puts("  | Cycles = ");
        uart_print_u32(cycles1);
        uart_puts("\n");

        /* [TEST 2] Severe Stall Hazard Injected */
        Evidence ev2;
        hiram_evidence_init(&ev2);
        hiram_evidence_set(&ev2, VAR_LOW_AIRSPEED, true);
        hiram_evidence_set(&ev2, VAR_HIGH_ANGLE_OF_ATTACK, true);

        t0 = DWT_CYCCNT;
        HiramAuditReport rep2 = hiram_audit_hazard(&ev2, thresh);
        t1 = DWT_CYCCNT;
        uint32_t cycles2 = t1 - t0;

        uart_puts("Test 2 (Stall Hazard): Decision = ");
        uart_puts(rep2.decision == HIRAM_VETOED_SAFETY_VIOLATION ? "VETOED" : "APPROVED");
        uart_puts("    | Cycles = ");
        uart_print_u32(cycles2);
        uart_puts("\n");

        /* [TEST 3] Sensor Contradiction Injected */
        Evidence ev3;
        hiram_evidence_init(&ev3);
        hiram_evidence_set(&ev3, VAR_LOW_AIRSPEED, true);
        hiram_evidence_set(&ev3, VAR_HIGH_ANGLE_OF_ATTACK, true);
        hiram_evidence_set(&ev3, HAZARD_VAR_ID, false);

        t0 = DWT_CYCCNT;
        HiramAuditReport rep3 = hiram_audit_hazard(&ev3, thresh);
        t1 = DWT_CYCCNT;
        uint32_t cycles3 = t1 - t0;

        uart_puts("Test 3 (Contradiction):Decision = ");
        uart_puts(rep3.decision == HIRAM_REJECTED_CONTRADICTION ? "REJECTED" : "UNEXPECTED");
        uart_puts("  | Cycles = ");
        uart_print_u32(cycles3);
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