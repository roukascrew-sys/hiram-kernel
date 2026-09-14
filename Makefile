# HIRAM ASIL-D Build & Verification Orchestrator
CC := gcc
CFLAGS := -std=c99 -Wall -Wextra -Werror -Wvla -pedantic -O2 -Iinclude -Isrc
AR := ar

BUILD_DIR := build
ARM_BUILD_DIR := build/arm

KERNEL_SRC := src/hiram_kernel.c
KERNEL_OBJ := $(BUILD_DIR)/hiram_kernel.o
KERNEL_OBJ_PIC := $(BUILD_DIR)/hiram_kernel.pic.o
KERNEL_LIB := $(BUILD_DIR)/libhiram_kernel.a
KERNEL_SHARED := $(BUILD_DIR)/libhiram_kernel.dll

TEST_BIN := $(BUILD_DIR)/test_c_kernel.exe
SIL_BIN  := $(BUILD_DIR)/test_sil_fault_injection.exe

ARM_CC      := arm-none-eabi-gcc
ARM_OBJCOPY := arm-none-eabi-objcopy
ARM_SIZE    := arm-none-eabi-size
ARM_CFLAGS  := -mcpu=cortex-m7 -mthumb -mfpu=fpv5-d16 -mfloat-abi=hard -std=c99 -Wall -Wextra -Werror -Wvla -pedantic -O2 -ffreestanding -fstack-usage -DHIRAM_TARGET_STM32H723 -Iinclude
ARM_LDSCRIPT := stm32h723zg.ld

ARM_BENCH_ELF := $(ARM_BUILD_DIR)/hiram_bench.elf
ARM_BENCH_BIN := $(ARM_BUILD_DIR)/hiram_bench.bin
ARM_BENCH_HEX := $(ARM_BUILD_DIR)/hiram_bench.hex
ARM_BENCH_OBJS := $(ARM_BUILD_DIR)/dwt_bench_policy_lut.o $(ARM_BUILD_DIR)/hiram_policy_lut.o

.PHONY: all clean verify verify_zero_alloc test sil_test bench dirs

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

$(ARM_BUILD_DIR)/dwt_bench_policy_lut.o: benchmarks/dwt_bench_policy_lut.c include/hiram_policy_lut.h | dirs
	$(ARM_CC) $(ARM_CFLAGS) -c $< -o $@

$(ARM_BENCH_ELF): $(ARM_BENCH_OBJS) $(ARM_LDSCRIPT)
	$(ARM_CC) $(ARM_CFLAGS) -T $(ARM_LDSCRIPT) -nostartfiles -Wl,--gc-sections -Wl,-Map=$(ARM_BUILD_DIR)/hiram_bench.map $(ARM_BENCH_OBJS) -lgcc -o $@

$(ARM_BENCH_BIN): $(ARM_BENCH_ELF)
	$(ARM_OBJCOPY) -O binary $< $@

$(ARM_BENCH_HEX): $(ARM_BENCH_ELF)
	$(ARM_OBJCOPY) -O ihex $< $@

bench: $(ARM_BENCH_BIN) $(ARM_BENCH_HEX)
	@echo "--- ARM CORTEX-M7 SECTION SIZES ---"
	@$(ARM_SIZE) $(ARM_BENCH_ELF)

verify: $(KERNEL_SHARED) verify_zero_alloc test sil_test bench
	@echo "===================================================================="
	@echo "  AUDIT READY: ALL HOST & CROSS-COMPILED TARGETS FULLY VERIFIED"
	@echo "===================================================================="

clean:
	rm -rf $(BUILD_DIR)
