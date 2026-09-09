# On-target DWT timing capture

How to produce a timing measurement for the HIRAM kernel on real hardware, and
where to record it.

**No such measurement exists yet.** Every latency figure in this repository and
in `FORMAL_CERTIFICATION_REPORT.md` was taken on a desktop host under Windows
`QueryPerformanceCounter`. That is not WCET, and it is not a predictor of the
Cortex-M7 result in either direction: the host runs at roughly 3–4 GHz, and the
expensive part of this kernel is 64-bit Euclidean GCD loops on a core with no
hardware 64-bit divide. This document exists so the real number can be taken
once and interpreted correctly afterwards.

## The image

| | |
|---|---|
| Image | `docs/verification/firmware/hiram_firmware.bin` |
| Source commit | `ef8d021af9418cbb4fdd9a4cc16fda8465fb43df` |
| Toolchain | Arm GNU Toolchain 14.2.Rel1 (`arm-none-eabi-gcc` 14.2.1) |
| SHA256 (.bin) | `fda9f5aeb6b6f468a0d60d21351c5392b8d360a9788705d6f58bcc44e0c5baad` |
| SHA256 (.hex) | `b1c5a013858cb5445838b50e409db5f56b8d3cb13d026a3eeaa5a420a2767d33` |
| Sizes | text 11,528 B · data 12 B · bss 1,304 B |

The image is tracked deliberately, against the usual rule about binaries in
version control, so that a recorded measurement identifies exactly what was
executed rather than "whatever `main` built to that week". It is byte-identical
across repeated builds, verified by building twice and comparing, so you can
also just rebuild it:

```bash
STRICT="-std=c99 -Wall -Wextra -Werror -Wconversion -Wsign-conversion -Wshadow \
        -Wcast-align -Wcast-qual -Wswitch-enum -Wundef -Wformat=2 -pedantic-errors"
arm-none-eabi-gcc -mcpu=cortex-m7 -mthumb -O3 $STRICT -Iinclude -Iinclude/hiram -c src/hiram_eval.c  -o k.o
arm-none-eabi-gcc -mcpu=cortex-m7 -mthumb -O3 $STRICT -Iinclude -Iinclude/hiram -c src/target_main.c -o t.o
arm-none-eabi-gcc -T stm32h723zg.ld -nostartfiles -mcpu=cortex-m7 -mthumb -Wl,--gc-sections k.o t.o -lgcc -o fw.elf
arm-none-eabi-objcopy -O binary fw.elf hiram_firmware.bin
sha256sum hiram_firmware.bin   # must match the table above
```

## Hardware

A NUCLEO-H723ZG, or any STM32H723ZG board exposing USART3 TX on PD8. On the
Nucleo, PD8 is wired to the on-board ST-LINK virtual COM port, so no extra
adapter is needed — one USB cable carries both the programmer and the console.

## Flash

To `0x08000000`. Any of these work; **none has been run on this machine**, so
treat the exact invocations as the documented usage rather than something
verified here.

```bash
# STM32CubeProgrammer
STM32_Programmer_CLI -c port=SWD -w hiram_firmware.bin 0x08000000 -rst

# stlink (open source)
st-flash write hiram_firmware.bin 0x08000000

# OpenOCD
openocd -f board/st_nucleo_h743zi.cfg \
        -c "program hiram_firmware.bin 0x08000000 verify reset exit"
```

## Capture

**115200 8N1**, no flow control.

```bash
# Linux / macOS
screen /dev/ttyACM0 115200          # or: picocom -b 115200 /dev/ttyACM0
# capture to a file instead:
stty -F /dev/ttyACM0 115200 raw -echo && cat /dev/ttyACM0 | tee dwt-capture.txt
```

```powershell
# Windows - find the ST-LINK COM port first
Get-CimInstance Win32_SerialPort | Select-Object DeviceID, Description
# then use PuTTY, Tera Term, or:
mode COM7 BAUD=115200 PARITY=n DATA=8 STOP=1
```

Let it run for at least ten frames. Frames are independent; the spread across
them is the useful part, since a single frame tells you nothing about variance.

## Reading the output

The banner is load-bearing. Record it with the numbers:

```
================================================================
  HIRAM FLIGHT SAFETY KERNEL - STM32H723ZG (CORTEX-M7)
  Zero-Heap MISRA-C Safety Monitor Online | DWT Active
================================================================
  Clock  : PLL1 200 MHz from HSI
  CPU    : 200 MHz  (DWT_CYCCNT counts at this rate)
  Frame  : 400 Hz = 2500 us = 500000 cycles
================================================================

--- [FRAME 1] ---------------------------------------------
Test 1 (Nominal):      Decision = APPROVED  | <cyc> cyc = <us> us = <pct>% of frame
Test 2 (Stall Hazard): Decision = VETOED    | <cyc> cyc = <us> us = <pct>% of frame
```

The firmware converts cycles to time itself using the clock it actually
achieved, so the capture is interpretable without knowing anything else. That
matters: an earlier version printed raw cycles and never printed the clock, so a
log could be converted at the part's 550 MHz headline rate and understate
latency by 8.6× with nothing in the file to contradict it.

### If the clock line says `HSI 64 MHz (PLL not engaged)`

The PLL configuration in `clock_init()` did not take. This is a designed
outcome, not a crash: every wait in that function is bounded and any timeout
leaves the part on the HSI and says so. The measurement is still valid — it is
simply at 64 MHz, where a 2.5 ms frame is 160,000 cycles instead of 500,000.

The clock code has never been executed on silicon. If it does not engage, check
the register offsets and field positions in `clock_init()` against RM0468
(STM32H72x/H73x) before changing anything else; the constants are named after
the fields they drive for exactly this reason.

## Recording the result

Add `docs/verification/WP2-on-target-timing.md` containing:

1. The full banner, verbatim, so the clock is unambiguous.
2. A table of per-frame cycles for each test, at least ten frames.
3. Min / median / max, in cycles and microseconds, with the percentage of frame.
4. Board, toolchain, image SHA256, and date.
5. Explicitly: whether this is a **measured maximum over N frames**, not a WCET.
   A sampled maximum on a quiet board is not a worst-case execution time —
   WCET requires static analysis accounting for pipeline, cache and memory
   behaviour, or a measurement campaign designed to provoke the worst path.
   Calling a sampled maximum "WCET" is the error the existing certification
   report already makes; do not repeat it here.

Send the raw capture and it can be written up against that structure.
