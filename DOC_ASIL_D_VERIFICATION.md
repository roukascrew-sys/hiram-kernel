# HIRAM Safety Kernel: ASIL-D Audit Remediation & Verification Dossier
**Standard Compliance**: ISO 26262:2018 Part 6 (Software Level, ASIL-D)  
**Target Architecture**: ARM Cortex-M7 (STM32H723ZG)  
**Verification Baseline**: Post-Adversarial Audit Resolution  

---

## 1. Audit Defect Remediation Summary

| Finding | Classification | Subsystem | Resolution Mechanism |
| :--- | :--- | :--- | :--- |
| **B1 / M1** | Major / Minor | LUT Evaluator | Dual-rail byte encoding (`0x0C, 0x09, 0x06`). Dynamic bit-flip validation on every access. Permanent state latch via `g_lut_operational`. MISRA 10.1 unsigned shift indexing. |
| **B2 / M2** | Major / Minor | Bayesian Kernel | Loss matrix runtime non-negativity validation (`hiram_kernel_validate_loss_matrix`). Non-finite joint probability traps returning `HIRAM_ERR_NUMERICAL_FAULT`. Defensive overwrite of `*out` on NULL pointers. |
| **B3** | Major | MPU Configuration | Corrected `MPU_RASR_NORMAL_WT` to bit 17 (C-bit). Unified release binary with `verify_release_image` symbol check. |
| **B4** | Major | Clock / PLL Tree | Bounded clock initialization sequence with VOS0 read-back and fallback to safe operating frequency. |
| **B5** | Major | Safe State Reaction | Active actuator inhibit via `TIM1`/`TIM8` `BDTR.MOE` kill before any diagnostic UART transmission. |
| **M3** | Minor | Host Harness | Platform-agnostic shared library target (`.dll` / `.so`) and corrected signed `int32_t` ctypes field binding. |

---

## 2. Updated Verification Gate Results
- **Zero Dynamic Allocation Scan**: 0 heap symbols across `libhiram_kernel.a` and final release ELF.
- **Kernel Unit Checks**: 914 / 914 passing.
- **Full Test Suite**: 132 / 132 passing (`tools/pre_audit_verifier.py: PASSED`).
- **Release Image Symbols**: Verified inclusion of `hiram_mpu_lock_regions` and `hiram_actuator_inhibit`.
- **LUT Canonical CRC32**: `0xDBE2F4BA` (512/512 single-bit mutations rejected).
