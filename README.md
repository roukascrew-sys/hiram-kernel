# HIRAM Flight Safety Kernel

HIRAM (High-Integrity Rational Arithmetic Monitor) is an embedded, zero-allocation flight safety kernel designed for hard real-time avionics control loops.

## System Invariants
- Dynamic Memory: 0 bytes (No malloc/free; fully MISRA-C:2012 Rule 21.3 compliant)
- Recursion: None (Flat topological array sweep; MISRA-C:2012 Rule 17.2 compliant)
- Max Denominator: 51 bits (Operates in 64-bit int64_t registers with __int128_t intermediate accumulation)
- Stack Scratchpad: 1,328 bytes deterministic footprint
- Typical Latency: ~24.5 us (Consumes < 1.0% of a 400 Hz / 2.5 ms control frame)

## Manual Build (GCC)
```powershell
gcc -O3 -Wall -Iinclude -Iinclude/hiram -DHIRAM_STANDALONE_TEST src/hiram_eval.c -o hiram_test.exe
.\hiram_test.exe