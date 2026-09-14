# HIRAM ASIL-D Build & Verification Orchestrator
CC := gcc
CFLAGS := -std=c99 -Wall -Wextra -Werror -Wvla -pedantic -O2 -Iinclude -Isrc
AR := ar

BUILD_DIR := build
ARM_BUILD_DIR := build/arm

# M3: the shared-library and executable suffixes are selected per host rather
# than hardcoded to Windows. tools/hiram_c_binding.py asks for
# build/libhiram_kernel.so on Linux and build/libhiram_kernel.dll on Windows
# and shells out to this Makefile to build it, so a .dll-only rule made every
# Python binding test fail at import on a Linux runner.
ifeq ($(OS),Windows_NT)
    SHARED_SUFFIX := .dll
    EXE_SUFFIX    := .exe
else
    SHARED_SUFFIX := .so
    EXE_SUFFIX    :=
endif

KERNEL_SRC := src/hiram_kernel.c
KERNEL_OBJ := $(BUILD_DIR)/hiram_kernel.o
KERNEL_OBJ_PIC := $(BUILD_DIR)/hiram_kernel.pic.o
KERNEL_LIB := $(BUILD_DIR)/libhiram_kernel.a
KERNEL_SHARED := $(BUILD_DIR)/libhiram_kernel$(SHARED_SUFFIX)

TEST_BIN := $(BUILD_DIR)/test_c_kernel$(EXE_SUFFIX)
SIL_BIN  := $(BUILD_DIR)/test_sil_fault_injection$(EXE_SUFFIX)

ARM_CC      := arm-none-eabi-gcc
ARM_NM      := arm-none-eabi-nm
ARM_OBJCOPY := arm-none-eabi-objcopy
ARM_SIZE    := arm-none-eabi-size
ARM_CFLAGS  := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -std=c99 -Wall -Wextra -Werror -Wvla -pedantic -O2 -ffreestanding -fstack-usage -DHIRAM_TARGET_STM32H723 -Iinclude -Iinclude/hiram
ARM_LDSCRIPT := stm32h723zg.ld

# B4: one firmware image, not two. src/target_main.c owns the vector table,
# Reset_Handler, clocking, console, the actuator safe state and the MPU
# lockdown; benchmarks/dwt_bench_policy_lut.c contributes the DWT measurement
# routine that target_main() calls. The pre-audit ARM_BENCH_OBJS listed only
# the benchmark and the LUT, so the MPU init in src/target_main.c was compiled
# by CI but never linked into the artifact this Makefile produced.
ARM_FW_ELF := $(ARM_BUILD_DIR)/hiram_firmware.elf
ARM_FW_BIN := $(ARM_BUILD_DIR)/hiram_firmware.bin
ARM_FW_HEX := $(ARM_BUILD_DIR)/hiram_firmware.hex
ARM_FW_OBJS := $(ARM_BUILD_DIR)/target_main.o \
               $(ARM_BUILD_DIR)/dwt_bench_policy_lut.o \
               $(ARM_BUILD_DIR)/hiram_policy_lut.o \
               $(ARM_BUILD_DIR)/hiram_eval.o

# Symbols whose presence in the linked image is the evidence that the release
# artifact contains the safety architecture, rather than merely that the
# sources containing it compile. Checked by verify_release_image.
ARM_REQUIRED_SYMS := target_main hiram_mpu_lock_regions hiram_actuator_inhibit \
                     hiram_safe_state_trap hiram_policy_lut_init HardFault_Handler

.PHONY: all clean verify verify_zero_alloc verify_release_image test sil_test bench firmware dirs

all: $(KERNEL_SHARED) verify

dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(ARM_BUILD_DIR)

$(KERNEL_OBJ): $(KERNEL_SRC) include/hiram_kernel.h include/hiram_tables.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(KERNEL_LIB): $(KERNEL_OBJ)
	$(AR) rcs $@ $<

$(KERNEL_OBJ_PIC): $(KERNEL_SRC) include/hiram_kernel.h include/hiram_tables.h | dirs
	$(CC) $(CFLAGS) -fPIC -c $< -o $@

$(KERNEL_SHARED): $(KERNEL_OBJ_PIC)
	$(CC) $(CFLAGS) -shared $< -o $@ -lm

$(TEST_BIN): tests/test_c_kernel.c $(KERNEL_LIB) | dirs
	$(CC) $(CFLAGS) $< $(KERNEL_LIB) -o $@ -lm

$(SIL_BIN): tests/test_sil_fault_injection.c src/hiram_policy_lut.c $(KERNEL_LIB) | dirs
	$(CC) $(CFLAGS) tests/test_sil_fault_injection.c src/hiram_policy_lut.c $(KERNEL_LIB) -o $@ -lm

test: $(TEST_BIN)
	$(TEST_BIN)

sil_test: $(SIL_BIN)
	$(SIL_BIN)

verify_zero_alloc: $(KERNEL_LIB)
	@echo "Auditing symbols for prohibited heap allocation..."
	@nm $(KERNEL_LIB) > $(BUILD_DIR)/symbols.txt
	@if grep -E "malloc|calloc|realloc|free|alloca" $(BUILD_DIR)/symbols.txt; then echo "CRITICAL VIOLATION: Heap symbols found"; exit 1; else echo "PASS: Zero dynamic allocations detected."; fi

$(ARM_BUILD_DIR)/hiram_policy_lut.o: src/hiram_policy_lut.c include/hiram_policy_lut.h | dirs
	$(ARM_CC) $(ARM_CFLAGS) -c $< -o $@

$(ARM_BUILD_DIR)/dwt_bench_policy_lut.o: benchmarks/dwt_bench_policy_lut.c include/hiram_policy_lut.h include/hiram_board.h | dirs
	$(ARM_CC) $(ARM_CFLAGS) -c $< -o $@

$(ARM_BUILD_DIR)/target_main.o: src/target_main.c include/hiram_policy_lut.h include/hiram_board.h include/hiram/hiram_circuit_data.h | dirs
	$(ARM_CC) $(ARM_CFLAGS) -c $< -o $@

$(ARM_BUILD_DIR)/hiram_eval.o: src/hiram_eval.c include/hiram/hiram_circuit_data.h include/hiram/hiram_eval.h | dirs
	$(ARM_CC) $(ARM_CFLAGS) -c $< -o $@

$(ARM_FW_ELF): $(ARM_FW_OBJS) $(ARM_LDSCRIPT)
	$(ARM_CC) $(ARM_CFLAGS) -T $(ARM_LDSCRIPT) -nostartfiles -Wl,--gc-sections -Wl,-Map=$(ARM_BUILD_DIR)/hiram_firmware.map $(ARM_FW_OBJS) -lgcc -o $@

$(ARM_FW_BIN): $(ARM_FW_ELF)
	$(ARM_OBJCOPY) -O binary $< $@

$(ARM_FW_HEX): $(ARM_FW_ELF)
	$(ARM_OBJCOPY) -O ihex $< $@

# B4 regression gate. The defect was not that the MPU code was wrong -- it was
# that the shipped image did not contain it. Compiling src/target_main.c is not
# evidence of that; finding its symbols in the linked ELF is.
verify_release_image: $(ARM_FW_ELF)
	@echo "Auditing release image for the safety architecture..."
	@$(ARM_NM) $(ARM_FW_ELF) > $(ARM_BUILD_DIR)/firmware_symbols.txt
	@for sym in $(ARM_REQUIRED_SYMS); do \
		if grep -qE " [TtWw] $$sym$$" $(ARM_BUILD_DIR)/firmware_symbols.txt; then \
			echo "  [OK]   $$sym linked into $(ARM_FW_ELF)"; \
		else \
			echo "  [FAIL] $$sym MISSING from $(ARM_FW_ELF)"; exit 1; \
		fi; \
	done
	@if $(ARM_NM) --print-size $(ARM_FW_ELF) | grep -q "_dtcm_lut_start"; then \
		echo "  [OK]   .dtcm_lut region present for the MPU lock"; \
	else \
		echo "  [FAIL] .dtcm_lut region absent -- MPU region 1 would cover nothing"; exit 1; \
	fi
	@echo "PASS: release image contains MPU lockdown, actuator inhibit and fault traps."

firmware: $(ARM_FW_BIN) $(ARM_FW_HEX) verify_release_image
	@echo "--- ARM CORTEX-M7 SECTION SIZES ---"
	@$(ARM_SIZE) $(ARM_FW_ELF)

# Retained as the historical entry point; there is only one image now.
bench: firmware

verify: $(KERNEL_SHARED) verify_zero_alloc test sil_test firmware
	@echo "===================================================================="
	@echo "  AUDIT READY: ALL HOST & CROSS-COMPILED TARGETS FULLY VERIFIED"
	@echo "===================================================================="

clean:
	rm -rf $(BUILD_DIR)
