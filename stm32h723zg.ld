/* STM32H723ZG Linker Script: 1024 KB Flash, 64 KB ITCM, 128 KB DTCM RAM */
ENTRY(Reset_Handler)

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 1024K
    ITCM  (rx)  : ORIGIN = 0x00000000, LENGTH = 64K
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

    /* Task 3.3: hiram_policy_lut_step() (HIRAM_ITCM_TEXT), copied from Flash
       into Instruction TCM by Reset_Handler before target_main() runs, so the
       evaluator executes zero-wait-state instead of over the AXI bus to
       Flash. KEEP() because nothing in this image calls it directly except
       through a section-scoped copy loop -- --gc-sections must not discard
       it as apparently unreferenced. */
    .itcm_text :
    {
        . = ALIGN(4);
        _itcm_start = .;
        KEEP(*(.itcm_text))
        KEEP(*(.itcm_text*))
        . = ALIGN(4);
        _itcm_end = .;
    } > ITCM AT > FLASH
    /* LOADADDR(.itcm_text) is the flash address of the *section's* start,
       which is not necessarily _itcm_start if the leading ALIGN() above ever
       has to pad (it happens not to today, since ITCM opens fresh at 0x0,
       but a copy loop that assumes so silently by coincidence is exactly the
       kind of thing that breaks on the next reshuffle -- see the 244-byte
       version of this bug that ALIGN(256) actually produced below). */
    _itcm_load = LOADADDR(.itcm_text) + (_itcm_start - ADDR(.itcm_text));

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

    /* Task 3.3: hiram_policy_lut.c's HIRAM_POLICY_LUT table (HIRAM_DTCM_DATA),
       copied from Flash into DTCM by Reset_Handler -- unlike .dtcm_data below,
       this section has real initialised content, so it is a LOAD section (AT>
       FLASH), not NOLOAD. Padded and aligned to 256 B so the post-init MPU
       lock (mpu_lock_dtcm_lut(), src/target_main.c) can cover exactly this
       region with a single power-of-two-sized, power-of-two-aligned ARMv7-M
       MPU region and nothing else in DTCM. */
    .dtcm_lut :
    {
        . = ALIGN(256);
        _dtcm_lut_start = .;
        KEEP(*(.dtcm_lut))
        KEEP(*(.dtcm_lut*))
        . = ALIGN(256);
        _dtcm_lut_end = .;
    } > DTCM AT > FLASH
    /* The 256-byte ALIGN() above almost always pads: .dtcm_lut opens
       wherever .bss happened to end, not on a 256 B boundary. That padding
       exists in both VMA and LMA, so LOADADDR(.dtcm_lut) alone (the flash
       address of the section's un-padded start) is *not* the flash address
       of the real table -- it is up to 252 bytes too early. Correcting by
       the same VMA-side offset (_dtcm_lut_start - ADDR(.dtcm_lut)) gives
       Reset_Handler's copy loop the flash address the table actually lives
       at. Measured on 2026-09-12's build: ADDR=0x2000000c, _dtcm_lut_start=
       0x20000100 -- a 244-byte gap that a naive LOADADDR() would have
       silently fed into the copy as the LUT's first 244 bytes. */
    _dtcm_lut_load = LOADADDR(.dtcm_lut) + (_dtcm_lut_start - ADDR(.dtcm_lut));

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