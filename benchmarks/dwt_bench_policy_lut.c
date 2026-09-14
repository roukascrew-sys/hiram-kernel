#include <stdint.h>
#include <stdbool.h>
#include "hiram_policy_lut.h"

#define RCC_BASE            0x58024400UL
#define RCC_AHB4ENR         (*(volatile uint32_t *)(RCC_BASE + 0x0E0))
#define RCC_APB1LENR        (*(volatile uint32_t *)(RCC_BASE + 0x0E8))
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

#define GPIOD_BASE          0x58020C00UL
#define GPIOD_MODER         (*(volatile uint32_t *)(GPIOD_BASE + 0x000))
#define GPIOD_OSPEEDR       (*(volatile uint32_t *)(GPIOD_BASE + 0x008))
#define GPIOD_AFRH          (*(volatile uint32_t *)(GPIOD_BASE + 0x024))

#define GPIOB_BASE          0x58020400UL
#define GPIOB_MODER         (*(volatile uint32_t *)(GPIOB_BASE + 0x000))
#define GPIOB_ODR           (*(volatile uint32_t *)(GPIOB_BASE + 0x014))

#define USART3_BASE         0x40004800UL
#define USART3_CR1          (*(volatile uint32_t *)(USART3_BASE + 0x000))
#define USART3_BRR          (*(volatile uint32_t *)(USART3_BASE + 0x00C))
#define USART3_ISR          (*(volatile uint32_t *)(USART3_BASE + 0x01C))
#define USART3_TDR          (*(volatile uint32_t *)(USART3_BASE + 0x028))

#define CoreDebug_DEMCR     (*(volatile uint32_t *)0xE000EDFCUL)
#define DWT_CTRL            (*(volatile uint32_t *)0xE0001000UL)
#define DWT_CYCCNT          (*(volatile uint32_t *)0xE0001004UL)

#define HSI_HZ              64000000u
#define TARGET_SYSCLK_HZ    550000000u
#define UART_BAUD           115200u
#define CLOCK_TIMEOUT       200000u

static uint32_t g_cpu_hz   = HSI_HZ;
static uint32_t g_pclk1_hz = HSI_HZ;

static void clock_init(void) {
    PWR_D3CR |= (3u << 14);
    for (volatile uint32_t i = 0u; i < 10000u; i++) { __asm__ volatile ("nop"); }
    FLASH_ACR = (FLASH_ACR & ~0x0Fu) | 4u;
    RCC_CR |= (1u << 0);
    while (!(RCC_CR & (1u << 2))) {}
    RCC_PLLCKSELR = 0x00000000u;
    RCC_PLLCFGR   = (2u << 2) | (1u << 16);
    RCC_PLL1DIVR  = (3u << 0) | (0u << 9) | (1u << 16) | (1u << 24);
    RCC_CR |= (1u << 24);
    for (uint32_t i = 0u; i < CLOCK_TIMEOUT; i++) {
        if (RCC_CR & (1u << 25)) { break; }
    }
    RCC_D1CFGR = 0x00000000u;
    RCC_D2CFGR = (4u << 4) | (4u << 8);
    RCC_CFGR   = (RCC_CFGR & ~0x07u) | 3u;
    for (uint32_t i = 0u; i < CLOCK_TIMEOUT; i++) {
        if (((RCC_CFGR >> 3) & 7u) == 3u) {
            g_cpu_hz = TARGET_SYSCLK_HZ;
            g_pclk1_hz = TARGET_SYSCLK_HZ / 4u;
            return;
        }
    }
}

static void hw_init(void) {
    RCC_AHB4ENR  |= (1u << 1) | (1u << 3);
    RCC_APB1LENR |= (1u << 18);
    GPIOB_MODER   = (GPIOB_MODER & ~(3u << 0)) | (1u << 0);
    GPIOB_ODR    &= ~(1u << 0);
    GPIOD_MODER   = (GPIOD_MODER   & ~(3u << 16)) | (2u << 16);
    GPIOD_OSPEEDR = (GPIOD_OSPEEDR & ~(3u << 16)) | (2u << 16);
    GPIOD_AFRH    = (GPIOD_AFRH    & ~(0xFu << 0)) | (7u << 0);
    USART3_CR1    = 0u;
    USART3_BRR    = (g_pclk1_hz + (UART_BAUD / 2u)) / UART_BAUD;
    USART3_CR1    = (1u << 3) | (1u << 0);
    CoreDebug_DEMCR |= (1u << 24);
    DWT_CTRL        |= (1u << 0);
    DWT_CYCCNT       = 0u;
}

static void uart_putc(char c) {
    while (!(USART3_ISR & (1u << 7))) {}
    USART3_TDR = (uint8_t)c;
}

static void uart_puts(const char *s) {
    while (*s) { uart_putc(*s++); }
}

static void uart_print_u32(uint32_t val) {
    char buf[11];
    int idx = 0;
    if (val == 0) {
        uart_putc('0');
        return;
    }
    while (val > 0) {
        buf[idx++] = (char)('0' + (val % 10u));
        val /= 10u;
    }
    for (int i = idx - 1; i >= 0; i--) {
        uart_putc(buf[i]);
    }
}

#define BENCH_ITERATIONS 10000u

static void run_benchmark(void) {
    volatile uint32_t sink = 0;
    volatile uint32_t dummy_sink = 0;

    __asm__ volatile ("cpsid i" : : : "memory");

    uint32_t t_cal0 = DWT_CYCCNT;
    for (uint32_t i = 0; i < BENCH_ITERATIONS; i++) {
        uint32_t combo = i & 0x3Fu;
        int32_t lidar = (int32_t)(combo >> 4)           - 1;
        int32_t tof   = (int32_t)((combo >> 2) & 0x3u) - 1;
        int32_t slip  = (int32_t)(combo & 0x3u)        - 1;
        dummy_sink += (uint32_t)(lidar + tof + slip);
    }
    uint32_t t_cal1 = DWT_CYCCNT;
    uint32_t loop_overhead = t_cal1 - t_cal0;

    uint32_t t0 = DWT_CYCCNT;
    for (uint32_t i = 0; i < BENCH_ITERATIONS; i++) {
        uint32_t combo = i & 0x3Fu;
        int32_t lidar = (int32_t)(combo >> 4)           - 1;
        int32_t tof   = (int32_t)((combo >> 2) & 0x3u) - 1;
        int32_t slip  = (int32_t)(combo & 0x3u)        - 1;
        hiram_lut_result_t r = hiram_policy_lut_step(lidar, tof, slip);
        sink += r.action;
    }
    uint32_t t1 = DWT_CYCCNT;

    __asm__ volatile ("cpsie i" : : : "memory");

    uint32_t total_cycles = t1 - t0;
    uint32_t net_cycles = (total_cycles > loop_overhead) ? (total_cycles - loop_overhead) : total_cycles;

    uint32_t gross_avg_x100 = (uint32_t)(((uint64_t)total_cycles * 100ULL) / BENCH_ITERATIONS);
    uint32_t net_avg_x100   = (uint32_t)(((uint64_t)net_cycles * 100ULL) / BENCH_ITERATIONS);
    uint32_t ns_per_call_x10 = (uint32_t)(((uint64_t)net_cycles * 10000000000ULL) / ((uint64_t)g_cpu_hz * BENCH_ITERATIONS));

    uart_puts("\r\n\r\n================================================================\r\n");
    uart_puts("  HIRAM DWT BENCHMARK: hiram_policy_lut_step()\r\n");
    uart_puts("================================================================\r\n");
    uart_puts("  Iterations        : ");
    uart_print_u32(BENCH_ITERATIONS);
    uart_puts("\r\n  CPU Clock         : ");
    uart_print_u32(g_cpu_hz / 1000000u);
    uart_puts(" MHz\r\n");
    uart_puts("  Total Gross Cyc   : ");
    uart_print_u32(total_cycles);
    uart_puts(" cyc\r\n");
    uart_puts("  Loop Overhead Cyc : ");
    uart_print_u32(loop_overhead);
    uart_puts(" cyc\r\n");
    uart_puts("  Net LUT Cycles    : ");
    uart_print_u32(net_cycles);
    uart_puts(" cyc (isolated)\r\n");
    uart_puts("  Gross Avg/Call    : ");
    uart_print_u32(gross_avg_x100 / 100u);
    uart_putc('.');
    uart_print_u32(gross_avg_x100 % 100u);
    uart_puts(" cycles\r\n");
    uart_puts("  Net Avg/Call      : ");
    uart_print_u32(net_avg_x100 / 100u);
    uart_putc('.');
    uart_print_u32(net_avg_x100 % 100u);
    uart_puts(" cycles\r\n");
    uart_puts("  Net Time/Call     : ");
    uart_print_u32(ns_per_call_x10 / 10u);
    uart_putc('.');
    uart_print_u32(ns_per_call_x10 % 10u);
    uart_puts(" ns\r\n");
    uart_puts("  Sink (Anti-DCE)   : ");
    uart_print_u32(sink + dummy_sink);
    uart_puts("\r\n================================================================\r\n");
}

extern uint32_t _sidata, _sdata, _edata, _sbss, _ebss, _estack;
extern uint32_t _itcm_load, _itcm_start, _itcm_end;
extern uint32_t _dtcm_lut_load, _dtcm_lut_start, _dtcm_lut_end;

void bench_main(void) {
    clock_init();
    hw_init();

    if (!hiram_policy_lut_verify_integrity()) {
        uart_puts("\r\n!! LUT INTEGRITY CHECK FAILED -- table unverified.\r\n");
        while (1) {
            GPIOB_ODR ^= (1u << 0);
            for (volatile uint32_t i = 0; i < 800000u; i++) { __asm__ volatile ("nop"); }
        }
    }

    while (1) {
        GPIOB_ODR ^= (1u << 0);
        run_benchmark();
        for (volatile uint32_t i = 0; i < 8000000u; i++) { __asm__ volatile ("nop"); }
    }
}

void Reset_Handler(void) {
    uint32_t *pSrc = &_sidata;
    uint32_t *pDst = &_sdata;
    while (pDst < &_edata) { *pDst++ = *pSrc++; }

    pDst = &_sbss;
    while (pDst < &_ebss) { *pDst++ = 0; }

    pSrc = &_itcm_load;
    pDst = &_itcm_start;
    while (pDst < &_itcm_end) { *pDst++ = *pSrc++; }

    pSrc = &_dtcm_lut_load;
    pDst = &_dtcm_lut_start;
    while (pDst < &_dtcm_lut_end) { *pDst++ = *pSrc++; }

    bench_main();
    while (1);
}

void Default_Handler(void) {
    while (1);
}

__attribute__((section(".isr_vector"), used))
const uint32_t vector_table[] = {
    (uint32_t)&_estack,
    (uint32_t)Reset_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    0, 0, 0, 0,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
    0,
    (uint32_t)Default_Handler,
    (uint32_t)Default_Handler,
};
