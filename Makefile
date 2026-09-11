# HIRAM Safety Kernel - Task 3.1: Flat-Memory C Kernel build.
#
# Builds src/hiram_kernel.c (+ its generated include/hiram_tables.h) into a
# static library and a standalone unit test binary, both under strict C99.
# This Makefile is intentionally independent of the repo's CMakeLists.txt
# (which builds the unrelated exact-rational-arithmetic hazard evaluator in
# src/hiram_eval.c) -- distinct sources, distinct build system, no overlap.

CC := gcc
CFLAGS := -Wall -Wextra -Werror -pedantic -std=c99 -Iinclude -O2
AR := ar

BUILD_DIR := build_task3_1
KERNEL_SRC := src/hiram_kernel.c
KERNEL_OBJ := $(BUILD_DIR)/hiram_kernel.o
KERNEL_LIB := $(BUILD_DIR)/libhiram_kernel.a

TEST_SRC := tests/test_c_kernel.c
TEST_OBJ := $(BUILD_DIR)/test_c_kernel.o
TEST_BIN := $(BUILD_DIR)/test_c_kernel

.PHONY: all clean test verify_zero_alloc

all: $(KERNEL_LIB) $(TEST_BIN)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(KERNEL_OBJ): $(KERNEL_SRC) include/hiram_kernel.h include/hiram_tables.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(KERNEL_SRC) -o $(KERNEL_OBJ)

$(KERNEL_LIB): $(KERNEL_OBJ)
	$(AR) rcs $(KERNEL_LIB) $(KERNEL_OBJ)

$(TEST_OBJ): $(TEST_SRC) include/hiram_kernel.h include/hiram_tables.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(TEST_SRC) -o $(TEST_OBJ)

$(TEST_BIN): $(TEST_OBJ) $(KERNEL_LIB)
	$(CC) $(CFLAGS) $(TEST_OBJ) $(KERNEL_LIB) -o $(TEST_BIN) -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

# Fails the build if any dynamic heap allocation symbol (malloc/free/calloc/
# realloc/alloca, and their libc __-prefixed/wrapped aliases) is referenced
# by the compiled kernel object or library -- independent of the
# `#pragma GCC poison` source-level guard in src/hiram_kernel.c, so a
# symbol pulled in some other way (e.g. via linking) is still caught.
verify_zero_alloc: $(KERNEL_OBJ) $(KERNEL_LIB)
	@echo "Scanning $(KERNEL_OBJ) and $(KERNEL_LIB) for heap allocation symbols..."
	@FOUND=0; \
	for ART in $(KERNEL_OBJ) $(KERNEL_LIB); do \
		HITS=$$(nm "$$ART" 2>/dev/null | grep -E '\b(malloc|free|calloc|realloc|alloca|reallocarray)\b' || true); \
		if [ -n "$$HITS" ]; then \
			echo "FAIL: heap allocation symbol(s) found in $$ART:"; \
			echo "$$HITS"; \
			FOUND=1; \
		fi; \
	done; \
	if [ "$$FOUND" -ne 0 ]; then \
		echo "verify_zero_alloc: FAILED"; \
		exit 1; \
	fi; \
	echo "verify_zero_alloc: PASSED -- no malloc/free/calloc/realloc/alloca symbols found"

clean:
	rm -rf $(BUILD_DIR)
