# HIRAM Safety Kernel - Task 3.1: Flat-Memory C Kernel build.
#
# Builds src/hiram_kernel.c (+ its generated include/hiram_tables.h) into a
# static library and a standalone unit test binary, both under strict C99.
# This Makefile is intentionally independent of the repo's CMakeLists.txt
# (which builds the unrelated exact-rational-arithmetic hazard evaluator in
# src/hiram_eval.c) -- distinct sources, distinct build system, no overlap.

CC := gcc
CFLAGS := -std=c99 -Wall -Wextra -Werror -Wvla -pedantic -O2 -Iinclude
AR := ar

BUILD_DIR := build
KERNEL_SRC := src/hiram_kernel.c
KERNEL_OBJ := $(BUILD_DIR)/hiram_kernel.o
KERNEL_OBJ_PIC := $(BUILD_DIR)/hiram_kernel.pic.o
KERNEL_LIB := $(BUILD_DIR)/libhiram_kernel.a

ifeq ($(OS),Windows_NT)
KERNEL_SHARED := $(BUILD_DIR)/libhiram_kernel.dll
else
KERNEL_SHARED := $(BUILD_DIR)/libhiram_kernel.so
endif

TEST_SRC := tests/test_c_kernel.c
TEST_OBJ := $(BUILD_DIR)/test_c_kernel.o
TEST_BIN := $(BUILD_DIR)/test_c_kernel

.PHONY: all clean test verify_zero_alloc

# verify_zero_alloc is part of `all` itself (not just a separately-run
# target) -- a build that produces heap-allocating object code is not
# considered to have succeeded.
all: $(KERNEL_LIB) $(KERNEL_SHARED) $(TEST_BIN) verify_zero_alloc

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(KERNEL_OBJ): $(KERNEL_SRC) include/hiram_kernel.h include/hiram_tables.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(KERNEL_SRC) -o $(KERNEL_OBJ)

$(KERNEL_LIB): $(KERNEL_OBJ)
	$(AR) rcs $(KERNEL_LIB) $(KERNEL_OBJ)

# Separate -fPIC object for the shared library target (Task 3.2's Python
# ctypes binding) so the static-library object above stays a plain,
# position-dependent build -- the two are never mixed into the same archive.
$(KERNEL_OBJ_PIC): $(KERNEL_SRC) include/hiram_kernel.h include/hiram_tables.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -fPIC -c $(KERNEL_SRC) -o $(KERNEL_OBJ_PIC)

$(KERNEL_SHARED): $(KERNEL_OBJ_PIC)
	$(CC) $(CFLAGS) -fPIC -shared $(KERNEL_OBJ_PIC) -o $(KERNEL_SHARED) -lm

$(TEST_OBJ): $(TEST_SRC) include/hiram_kernel.h include/hiram_tables.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) -c $(TEST_SRC) -o $(TEST_OBJ)

$(TEST_BIN): $(TEST_OBJ) $(KERNEL_LIB)
	$(CC) $(CFLAGS) $(TEST_OBJ) $(KERNEL_LIB) -o $(TEST_BIN) -lm

test: $(TEST_BIN)
	./$(TEST_BIN)

# Fails the build if any dynamic heap allocation symbol (malloc/free/calloc/
# realloc/alloca) is referenced by the compiled static library -- independent
# of the `#pragma GCC poison` source-level guard in src/hiram_kernel.c, so a
# symbol pulled in some other way (e.g. via linking) is still caught.
#
# The previous version of this target depended on and scanned the undefined
# make variable $(LIB) (rather than $(KERNEL_LIB)) -- nm ran with no file
# argument, silently defaulted to scanning a.out (or errored in a way that
# wasn't what it looked like), and no reference to the just-built kernel
# library was ever actually checked. Fixed to scan $(KERNEL_LIB) explicitly,
# and to hard-fail (rather than mask) an nm execution failure.
verify_zero_alloc: $(KERNEL_LIB)
	@echo "Checking for prohibited dynamic memory symbols..."
	@nm $(KERNEL_LIB) > $(BUILD_DIR)/symbols.txt || (echo "CRITICAL: nm failed to execute" && exit 1)
	@if grep -E "malloc|calloc|realloc|free|alloca" $(BUILD_DIR)/symbols.txt; then \
		echo "CRITICAL VIOLATION: Dynamic memory allocation detected!"; exit 1; \
	else \
		echo "PASS: Zero dynamic allocations detected in kernel binary."; \
	fi

clean:
	rm -rf $(BUILD_DIR)
