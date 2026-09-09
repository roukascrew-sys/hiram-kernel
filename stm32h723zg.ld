/* STM32H723ZG Linker Script: 1024 KB Flash, 128 KB DTCM RAM */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    DTCM  (rwx) : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(DTCM) + LENGTH(DTCM); /* Top of stack at end of DTCM */

/* Minimum DTCM the main stack must retain after all static allocation.
   Sized for the deepest kernel call chain (hiram_audit_hazard_vector 256 B ->
   hiram_eval_evidence 8 B -> .part.0 112 B = 376 B) plus Cortex-M7 exception
   stacking (32 B basic, 104 B with FPU context) and nested ISR headroom. */
_Min_Stack_Size = 0x1000; /* 4 KB */

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

    /* DTCM scratchpad and context section (zero-wait-state RAM) */
    /* NOLOAD: Reset_Handler copies .data and zeroes .bss, but nothing copies
       .dtcm_data. Marking it LOAD would place initialiser bytes in flash that
       are never transferred to RAM -- wasted flash and a false promise of
       initialisation. Occupants must self-initialise (hiram_context_init). */
    .dtcm_data (NOLOAD) :
    {
        . = ALIGN(8);
        _sdtcm = .;
        KEEP(*(.dtcm_data))
        KEEP(*(.dtcm_data*))
        . = ALIGN(8);
        _edtcm = .;
    } > DTCM

    /* Hard link-time stack budget. Silent stack overflow is the failure mode
       this kernel's context migration exists to remove; leaving the guarantee
       to inspection would reintroduce it. A build that cannot prove the
       separation must not link. */
    ASSERT(_edtcm + _Min_Stack_Size <= _estack,
           "DTCM exhausted: static allocation leaves less than _Min_Stack_Size for the stack")

    /* Discard unused unwind frames */
    /DISCARD/ :
    {
        *(.ARM.exidx*)
        *(.ARM.extab*)
    }
}