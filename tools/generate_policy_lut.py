"""
tools/generate_policy_lut.py

Evaluates the sealed Bayesian Inference Engine / C Kernel across all 64 discrete 
tri-state evidence combinations:
    LidarObs     in {-1, 0, 1, 2}
    TofObs       in {-1, 0, 1, 2}
    WheelSlipObs in {-1, 0, 1, 2}

Compiles the optimal Bayesian action decisions into:
  1. include/hiram_policy_lut.h  (Self-contained C99 zero-overhead LUT evaluator)
  2. build/policy_lut_data.json  (Bit-exact mapping verification table)
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hiram_c_binding import HIRAM_ACTION_NAMES, step_c_kernel

VALID_DOMAIN = (-1, 0, 1, 2)


def generate_lut():
    lut_entries = []
    c_table = []

    print(f"Generating 64-state Bayesian policy lookup table...")

    for lidar in VALID_DOMAIN:
        for tof in VALID_DOMAIN:
            for slip in VALID_DOMAIN:
                # Key indexing formula: (val + 1) -> maps {-1, 0, 1, 2} to {0, 1, 2, 3}
                idx = ((lidar + 1) << 4) | ((tof + 1) << 2) | (slip + 1)

                step_res = step_c_kernel(lidar=lidar, tof=tof, slip=slip)
                action_idx = step_res["selected_action_index"]
                action_name = step_res["selected_action"]

                lut_entries.append({
                    "lut_index": idx,
                    "lidar": lidar,
                    "tof": tof,
                    "slip": slip,
                    "action_index": action_idx,
                    "action_name": action_name,
                    "expected_losses": step_res["expected_losses"],
                })

    # Sort strictly by LUT index 0..63
    lut_entries.sort(key=lambda x: x["lut_index"])

    # Format C header
    header_path = REPO_ROOT / "include" / "hiram_policy_lut.h"
    json_path = REPO_ROOT / "build" / "policy_lut_data.json"

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(lut_entries, f, indent=2)

    lines = [
        "/**",
        " * hiram_policy_lut.h - Compiled 64-State Policy-to-LUT Evaluator",
        " *",
        " * Automatically synthesized from Bayesian C-Kernel decision policy.",
        " * Exact action selection over all 64 tri-state permutations.",
        " * Execution cost: 1 memory access / array lookup (~2 to 4 cycles on ARM Cortex-M7).",
        " */",
        "",
        "#ifndef HIRAM_POLICY_LUT_H",
        "#define HIRAM_POLICY_LUT_H",
        "",
        "#include <stdint.h>",
        "#include <stdbool.h>",
        "",
        "/* Action Definitions matching include/hiram_tables.h */",
        "#define HIRAM_LUT_ACTION_ACCEL           0",
        "#define HIRAM_LUT_ACTION_COAST           1",
        "#define HIRAM_LUT_ACTION_EMERGENCY_BRAKE 2",
        "",
        "/* 64-byte pre-computed optimal Bayesian decision policy */",
        "static const uint8_t HIRAM_POLICY_LUT[64] = {",
    ]

    for i in range(0, 64, 4):
        chunk = lut_entries[i:i+4]
        chunk_str = ", ".join(f"{c['action_index']}" for c in chunk)
        comment_items = [
            f"[{c['lut_index']:02d}]: L={c['lidar']:+d},T={c['tof']:+d},S={c['slip']:+d}->{c['action_name']}"
            for c in chunk
        ]
        lines.append(f"    {chunk_str}, /* {'; '.join(comment_items)} */")

    lines.extend([
        "};",
        "",
        "/**",
        " * Evaluates the 64-state policy LUT.",
        " * Returns action index: 0=ACCEL, 1=COAST, 2=EMERGENCY_BRAKE.",
        " * Out-of-bounds inputs return HIRAM_LUT_ACTION_EMERGENCY_BRAKE (fail-safe).",
        " */",
        "static inline uint8_t hiram_policy_lut_step(int32_t lidar, int32_t tof, int32_t slip) {",
        "    if ((lidar < -1 || lidar > 2) || (tof < -1 || tof > 2) || (slip < -1 || slip > 2)) {",
        "        return HIRAM_LUT_ACTION_EMERGENCY_BRAKE; /* Defensive boundary fallback */",
        "    }",
        "    uint32_t idx = (uint32_t)(((lidar + 1) << 4) | ((tof + 1) << 2) | (slip + 1));",
        "    return HIRAM_POLICY_LUT[idx & 0x3FU];",
        "}",
        "",
        "#endif /* HIRAM_POLICY_LUT_H */",
        "",
    ])

    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] Successfully synthesized {header_path}")
    print(f"[OK] Wrote verification dataset to {json_path}")


if __name__ == "__main__":
    generate_lut()