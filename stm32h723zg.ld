/* STM32H723ZG Linker Script: 1024 KB Flash, 128 KB DTCM RAM */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    DTCM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(DTCM) + LENGTH(DTCM); /* Top of stack at end of DTCM */

SECTIONS
{
    /* Cortex-M7 Vector Table placed at beginning of Flash */
    .isr_vector :
    {
        . = ALIGN(4);
        KEEP(*(.isr_vector))
        . = ALIGN(4);
    } > FLASH

    /* Program code and immutable Arithmetic Circuit tables */
    .text :
    {
        . = ALIGN(4);
        *(.text)
        *(.text*)
        . = ALIGN(4);
    } > FLASH

    .rodata :
    {
        . = ALIGN(4);
        *(.rodata)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    /* Initialized data copied from Flash to DTCM at boot */
    _sidata = LOADADDR(.data);
    .data :
    {
        . = ALIGN(4);
        _sdata = .;
        *(.data)
        *(.data*)
        . = ALIGN(4);
        _edata = .;
    } > DTCM AT > FLASH

    /* Zero-initialized RAM section */
    .bss :
    {
        . = ALIGN(4);
        _sbss = .;
        *(.bss)
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _ebss = .;
    } > DTCM

    /* Discard unused unwind frames */
    /DISCARD/ :
    {
        *(.ARM.exidx*)
        *(.ARM.extab*)
    }
}