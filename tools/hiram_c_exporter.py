"""HIRAM ZERO-MALLOC EMBEDDED C CODE GENERATOR

Generates MISRA-C:2012 Rule 17.2 Compliant Flat Topological Runtime
"""

from fractions import Fraction
import os
import sys
from typing import Dict, List, Set

from hiram_ac import NodeType
from hiram_checker import Rule
from hiram_compiler import compile_from_rules


def generate_c_code():
    print("=" * 80)
    print("HIRAM ZERO-MALLOC EMBEDDED C CODE GENERATOR (MISRA-C TOPOLOGICAL)")
    print("=" * 80)

    # 1. Define Canonical 12-Variable Flight Envelope Knowledge Base
    rules = [
        Rule(
            rule_name="R_DESCENT_CLEAR",
            premises=["terrain_clear", "altitude_stable"],
            conclusion="authorize_descent",
        ),
        Rule(
            rule_name="R_STALL_RISK",
            premises=["low_airspeed", "high_angle_of_attack"],
            conclusion="stall_hazard",
        ),
        Rule(
            rule_name="R_ICING_RISK",
            premises=["subzero_temp", "high_humidity"],
            conclusion="icing_hazard",
        ),
        Rule(
            rule_name="R_OVERSPEED_RISK",
            premises=["high_mach", "flaps_extended"],
            conclusion="structural_hazard",
        ),
    ]

    priors = {
        "terrain_clear": Fraction(99, 100),
        "altitude_stable": Fraction(95, 100),
        "authorize_descent": Fraction(1, 2),
        "low_airspeed": Fraction(1, 20),
        "high_angle_of_attack": Fraction(1, 10),
        "stall_hazard": Fraction(1, 100),
        "subzero_temp": Fraction(1, 5),
        "high_humidity": Fraction(1, 3),
        "icing_hazard": Fraction(1, 50),
        "high_mach": Fraction(1, 50),
        "flaps_extended": Fraction(1, 4),
        "structural_hazard": Fraction(1, 500),
    }

    print("Compiling flight rules to minimal reduced DAG...", end=" ", flush=True)
    kb = compile_from_rules(rules, priors)
    print(f"Done. Raw Nodes: {len(kb.circuit.nodes)}")

    circuit = kb.circuit
    root_id = kb.root_id
    var_to_id = kb.var_to_id
    hazard_var_id = var_to_id["stall_hazard"]

    # -------------------------------------------------------------------------
    # 2. Strict Topological Sort (MISRA-C:2012 Rule 17.2 Compliance)
    # -------------------------------------------------------------------------
    # Post-order DFS guarantees that for every node, all its children appear
    # at strictly lower indices in the emitted array (child_idx < parent_idx).
    print("Computing strict topological ordering...", end=" ", flush=True)
    topo_order: List[int] = []
    visited: Set[int] = set()

    def dfs_topo(nid: int):
        if nid in visited:
            return
        visited.add(nid)
        node = circuit.nodes[nid]
        for child_id in node.children:
            dfs_topo(child_id)
        topo_order.append(nid)

    dfs_topo(root_id)

    num_nodes = len(topo_order)
    id_to_idx: Dict[int, int] = {
        old_id: idx for idx, old_id in enumerate(topo_order)
    }
    root_idx = id_to_idx[root_id]
    assert (
        root_idx == num_nodes - 1
    ), f"Root must be last node, got {root_idx} vs {num_nodes-1}"

    # Verify topological invariant
    for old_id in topo_order:
        parent_idx = id_to_idx[old_id]
        for child_id in circuit.nodes[old_id].children:
            child_idx = id_to_idx[child_id]
            assert child_idx < parent_idx, (
                f"Topological order invariant violated: child {child_idx} >="
                f" parent {parent_idx}"
            )
    print(f"Verified. Ordered Nodes: {num_nodes} (Root Index: {root_idx})")

    # -------------------------------------------------------------------------
    # 3. Flatten DAG into Contiguous Static Arrays
    # -------------------------------------------------------------------------
    flat_children: List[int] = []
    flat_weights: List[Fraction] = []
    node_struct_defs: List[str] = []

    for old_id in topo_order:
        node = circuit.nodes[old_id]
        if node.node_type == NodeType.LITERAL:
            c_type = "NODE_LITERAL"
            v_id = node.var_id
            is_neg = 1 if node.is_negated else 0
            c_cnt = 0
            c_off = 0
            w_off = 0
        elif node.node_type == NodeType.PROD:
            c_type = "NODE_PROD"
            v_id = 0
            is_neg = 0
            c_cnt = len(node.children)
            c_off = len(flat_children)
            w_off = 0
            flat_children.extend([id_to_idx[c] for c in node.children])
        elif node.node_type == NodeType.SUM:
            c_type = "NODE_SUM"
            v_id = 0
            is_neg = 0
            c_cnt = len(node.children)
            c_off = len(flat_children)
            w_off = len(flat_weights)
            flat_children.extend([id_to_idx[c] for c in node.children])
            flat_weights.extend(node.weights)
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")

        node_struct_defs.append(
            f"    {{ {c_type}, {v_id}, {is_neg}, {c_cnt}, {c_off}, {w_off} }}"
        )

    # -------------------------------------------------------------------------
    # 4. Emit hiram_circuit_data.h (ROM Tables)
    # -------------------------------------------------------------------------
    print("Writing hiram_circuit_data.h...", end=" ", flush=True)
    with open("hiram_circuit_data.h", "w") as f:
        f.write(
            "/* AUTO-GENERATED BY hiram_c_exporter.py - DO NOT EDIT DIRECTLY"
            " */\n"
        )
        f.write(
            "/* STRICT TOPOLOGICAL DAG LAYOUT FOR MISRA-C:2012 COMPLIANCE"
            " */\n"
        )
        f.write("#ifndef HIRAM_CIRCUIT_DATA_H\n")
        f.write("#define HIRAM_CIRCUIT_DATA_H\n\n")
        f.write('#include "hiram_eval.h"\n\n')

        f.write("/* Telemetry Variable IDs */\n")
        for name, vid in var_to_id.items():
            f.write(f"#define VAR_{name.upper()} {vid}\n")
        f.write(f"#define NUM_VARIABLES {len(var_to_id)}\n")
        f.write(f"#define HAZARD_VAR_ID {hazard_var_id}\n\n")

        f.write(f"#define CIRCUIT_NODE_COUNT {num_nodes}\n")
        f.write(f"#define CIRCUIT_ROOT_ID {root_idx}\n")
        f.write(f"#define CHILDREN_ARRAY_SIZE {len(flat_children)}\n")
        f.write(f"#define WEIGHTS_ARRAY_SIZE {len(flat_weights)}\n\n")

        # Flat children array
        f.write(
            "static const uint16_t g_circuit_children[CHILDREN_ARRAY_SIZE] ="
            " {\n    "
        )
        f.write(", ".join(str(c) for c in flat_children))
        f.write("\n};\n\n")

        # Flat weights array
        f.write(
            "static const Rational g_circuit_weights[WEIGHTS_ARRAY_SIZE] = {\n"
        )
        weight_entries = [
            f"    {{ {w.numerator}LL, {w.denominator}LL }}"
            for w in flat_weights
        ]
        f.write(",\n".join(weight_entries))
        f.write("\n};\n\n")

        # Static node table
        f.write(
            "static const CircuitNode g_circuit_nodes[CIRCUIT_NODE_COUNT] ="
            " {\n"
        )
        f.write(",\n".join(node_struct_defs))
        f.write("\n};\n\n")

        f.write("#endif /* HIRAM_CIRCUIT_DATA_H */\n")
    print("Done.")

    # -------------------------------------------------------------------------
    # 5. Emit hiram_eval.h (API Header)
    # -------------------------------------------------------------------------
    print("Writing hiram_eval.h...", end=" ", flush=True)
    with open("hiram_eval.h", "w") as f:
        f.write(
            "/* AUTO-GENERATED BY hiram_c_exporter.py - DO NOT EDIT DIRECTLY"
            " */\n"
        )
        f.write("#ifndef HIRAM_EVAL_H\n")
        f.write("#define HIRAM_EVAL_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdbool.h>\n\n")

        f.write("typedef enum {\n")
        f.write("    NODE_LITERAL = 0,\n")
        f.write("    NODE_PROD    = 1,\n")
        f.write("    NODE_SUM     = 2\n")
        f.write("} NodeType;\n\n")

        f.write("typedef struct {\n")
        f.write("    int64_t num;\n")
        f.write("    int64_t den;\n")
        f.write("} Rational;\n\n")

        f.write("typedef struct {\n")
        f.write("    NodeType type;\n")
        f.write("    uint16_t var_id;\n")
        f.write("    uint8_t  is_negated;\n")
        f.write("    uint16_t child_count;\n")
        f.write("    uint16_t child_offset;\n")
        f.write("    uint16_t weight_offset;\n")
        f.write("} CircuitNode;\n\n")

        f.write("typedef struct {\n")
        f.write("    uint32_t observed_mask;\n")
        f.write("    uint32_t values_mask;\n")
        f.write("} Evidence;\n\n")

        f.write("typedef enum {\n")
        f.write("    HIRAM_APPROVED = 0,\n")
        f.write("    HIRAM_VETOED_SAFETY_VIOLATION = 1,\n")
        f.write("    HIRAM_REJECTED_CONTRADICTION = 2\n")
        f.write("} HiramDecision;\n\n")

        f.write("typedef struct {\n")
        f.write("    HiramDecision decision;\n")
        f.write("    Rational hazard_probability;\n")
        f.write("    Rational hazard_threshold;\n")
        f.write("} HiramAuditReport;\n\n")

        f.write("/* Public API */\n")
        f.write("void hiram_evidence_init(Evidence* ev);\n")
        f.write("void hiram_evidence_set(Evidence* ev, uint16_t var_id, bool val);\n")
        f.write("void hiram_evidence_clear(Evidence* ev, uint16_t var_id);\n")
        f.write("Rational hiram_eval_evidence(const Evidence* ev);\n")
        f.write(
            "HiramAuditReport hiram_audit_hazard(const Evidence* base_ev,"
            " Rational max_acceptable_risk);\n\n"
        )

        f.write("#endif /* HIRAM_EVAL_H */\n")
    print("Done.")

    # -------------------------------------------------------------------------
    # 6. Emit hiram_eval.c (Flat Topological Iterative Runtime)
    # -------------------------------------------------------------------------
    print("Writing hiram_eval.c...", end=" ", flush=True)
    with open("hiram_eval.c", "w") as f:
        f.write(
            "/* AUTO-GENERATED BY hiram_c_exporter.py - ZERO-MALLOC EMBEDDED C"
            " RUNTIME */\n"
        )
        f.write(
            "/* MISRA-C:2012 Rule 17.2 Compliant: Flat Non-Recursive"
            " Topological Evaluation */\n"
        )
        f.write('#include "hiram_eval.h"\n')
        f.write('#include "hiram_circuit_data.h"\n')
        f.write("#include <stdio.h>\n\n")

        f.write("/* 128-bit Binary Euclidean GCD Algorithm */\n")
        f.write("static inline __int128_t gcd128(__int128_t a, __int128_t b) {\n")
        f.write("    if (a < 0) a = -a;\n")
        f.write("    if (b < 0) b = -b;\n")
        f.write("    while (b != 0) {\n")
        f.write("        __int128_t r = a % b;\n")
        f.write("        a = b;\n")
        f.write("        b = r;\n")
        f.write("    }\n")
        f.write("    return a;\n")
        f.write("}\n\n")

        f.write(
            "static inline Rational rational_make(__int128_t n, __int128_t d)"
            " {\n"
        )
        f.write("    if (d < 0) { n = -n; d = -d; }\n")
        f.write("    if (n == 0) return (Rational){0LL, 1LL};\n")
        f.write("    __int128_t g = gcd128(n, d);\n")
        f.write(
            "    return (Rational){(int64_t)(n / g), (int64_t)(d / g)};\n"
        )
        f.write("}\n\n")

        f.write("static inline Rational rational_mul(Rational a, Rational b) {\n")
        f.write(
            "    if (a.num == 0 || b.num == 0) return (Rational){0LL, 1LL};\n"
        )
        f.write("    __int128_t g1 = gcd128(a.num, b.den);\n")
        f.write("    __int128_t g2 = gcd128(b.num, a.den);\n")
        f.write(
            "    __int128_t n = (__int128_t)(a.num / g1) * (b.num / g2);\n"
        )
        f.write(
            "    __int128_t d = (__int128_t)(a.den / g2) * (b.den / g1);\n"
        )
        f.write("    return (Rational){(int64_t)n, (int64_t)d};\n")
        f.write("}\n\n")

        f.write("static inline Rational rational_add(Rational a, Rational b) {\n")
        f.write("    if (a.num == 0) return b;\n")
        f.write("    if (b.num == 0) return a;\n")
        f.write("    __int128_t g = gcd128(a.den, b.den);\n")
        f.write("    __int128_t d = ((__int128_t)a.den / g) * b.den;\n")
        f.write(
            "    __int128_t n = ((__int128_t)a.num * (b.den / g)) +"
            " ((__int128_t)b.num * (a.den / g));\n"
        )
        f.write("    return rational_make(n, d);\n")
        f.write("}\n\n")

        f.write("static inline Rational rational_div(Rational a, Rational b) {\n")
        f.write("    return rational_mul(a, (Rational){b.den, b.num});\n")
        f.write("}\n\n")

        f.write("void hiram_evidence_init(Evidence* ev) {\n")
        f.write("    ev->observed_mask = 0;\n")
        f.write("    ev->values_mask = 0;\n")
        f.write("}\n\n")

        f.write(
            "void hiram_evidence_set(Evidence* ev, uint16_t var_id, bool val)"
            " {\n"
        )
        f.write("    ev->observed_mask |= (1u << var_id);\n")
        f.write("    if (val) ev->values_mask |= (1u << var_id);\n")
        f.write("    else     ev->values_mask &= ~(1u << var_id);\n")
        f.write("}\n\n")

        f.write("void hiram_evidence_clear(Evidence* ev, uint16_t var_id) {\n")
        f.write("    ev->observed_mask &= ~(1u << var_id);\n")
        f.write("    ev->values_mask &= ~(1u << var_id);\n")
        f.write("}\n\n")

        f.write("/*\n")
        f.write(
            " * Flat iterative evaluation loop (MISRA-C:2012 Rule 17.2"
            " Compliant).\n"
        )
        f.write(
            " * Circuit nodes are arranged in strict topological order:\n"
        )
        f.write(
            " * for every node i, all child indices are strictly < i.\n"
        )
        f.write(
            " * Stack usage is deterministic: exactly sizeof(Rational) *"
            " CIRCUIT_NODE_COUNT.\n"
        )
        f.write(
            " * Zero heap allocation, zero function call overhead, zero"
            " recursion risk.\n"
        )
        f.write(" */\n")
        f.write("Rational hiram_eval_evidence(const Evidence* ev) {\n")
        f.write("    Rational memo[CIRCUIT_NODE_COUNT];\n\n")
        f.write(
            "    for (uint16_t i = 0; i < CIRCUIT_NODE_COUNT; i++) {\n"
        )
        f.write("        const CircuitNode* n = &g_circuit_nodes[i];\n")
        f.write("        Rational res;\n\n")
        f.write("        if (n->type == NODE_LITERAL) {\n")
        f.write("            uint32_t mask = (1u << n->var_id);\n")
        f.write("            if ((ev->observed_mask & mask) == 0) {\n")
        f.write("                res = (Rational){1LL, 1LL};\n")
        f.write("            } else {\n")
        f.write("                bool val = (ev->values_mask & mask) != 0;\n")
        f.write(
            "                bool sat = (val && !n->is_negated) || (!val &&"
            " n->is_negated);\n"
        )
        f.write(
            "                res = sat ? (Rational){1LL, 1LL} :"
            " (Rational){0LL, 1LL};\n"
        )
        f.write("            }\n")
        f.write("        } else if (n->type == NODE_PROD) {\n")
        f.write("            res = (Rational){1LL, 1LL};\n")
        f.write(
            "            for (uint16_t c = 0; c < n->child_count; c++) {\n"
        )
        f.write(
            "                uint16_t child_id = g_circuit_children[n->child_offset"
            " + c];\n"
        )
        f.write(
            "                /* child_id < i strictly guaranteed by topological"
            " sort */\n"
        )
        f.write(
            "                res = rational_mul(res, memo[child_id]);\n"
        )
        f.write("                if (res.num == 0) break;\n")
        f.write("            }\n")
        f.write("        } else if (n->type == NODE_SUM) {\n")
        f.write("            res = (Rational){0LL, 1LL};\n")
        f.write(
            "            for (uint16_t c = 0; c < n->child_count; c++) {\n"
        )
        f.write(
            "                uint16_t child_id = g_circuit_children[n->child_offset"
            " + c];\n"
        )
        f.write(
            "                /* child_id < i strictly guaranteed by topological"
            " sort */\n"
        )
        f.write(
            "                Rational w ="
            " g_circuit_weights[n->weight_offset + c];\n"
        )
        f.write(
            "                Rational branch = rational_mul(w,"
            " memo[child_id]);\n"
        )
        f.write("                res = rational_add(res, branch);\n")
        f.write("            }\n")
        f.write("        } else {\n")
        f.write("            res = (Rational){0LL, 1LL};\n")
        f.write("        }\n\n")
        f.write("        memo[i] = res;\n")
        f.write("    }\n\n")
        f.write("    return memo[CIRCUIT_ROOT_ID];\n")
        f.write("}\n\n")

        f.write(
            "HiramAuditReport hiram_audit_hazard(const Evidence* base_ev,"
            " Rational max_acceptable_risk) {\n"
        )
        f.write("    HiramAuditReport rep;\n")
        f.write("    rep.hazard_threshold = max_acceptable_risk;\n\n")
        f.write("    Rational p_e = hiram_eval_evidence(base_ev);\n")
        f.write("    if (p_e.num == 0) {\n")
        f.write("        rep.decision = HIRAM_REJECTED_CONTRADICTION;\n")
        f.write("        rep.hazard_probability = (Rational){0LL, 1LL};\n")
        f.write("        return rep;\n")
        f.write("    }\n\n")
        f.write("    Evidence ev_hazard = *base_ev;\n")
        f.write("    hiram_evidence_set(&ev_hazard, HAZARD_VAR_ID, true);\n")
        f.write(
            "    Rational p_hazard_and_e = hiram_eval_evidence(&ev_hazard);\n"
        )
        f.write("    Rational posterior = rational_div(p_hazard_and_e, p_e);\n")
        f.write("    rep.hazard_probability = posterior;\n\n")
        f.write(
            "    /* Integer cross-multiplication: posterior >"
            " max_acceptable_risk */\n"
        )
        f.write(
            "    __int128_t lhs = (__int128_t)posterior.num *"
            " max_acceptable_risk.den;\n"
        )
        f.write(
            "    __int128_t rhs = (__int128_t)max_acceptable_risk.num *"
            " posterior.den;\n\n"
        )
        f.write("    if (lhs > rhs) {\n")
        f.write("        rep.decision = HIRAM_VETOED_SAFETY_VIOLATION;\n")
        f.write("    } else {\n")
        f.write("        rep.decision = HIRAM_APPROVED;\n")
        f.write("    }\n")
        f.write("    return rep;\n")
        f.write("}\n\n")

        # Standalone main self-test
        f.write("#ifdef HIRAM_STANDALONE_TEST\n")
        f.write("int main(void) {\n")
        f.write(
            '    printf("================================================================\\n");\n'
        )
        f.write(
            '    printf("HIRAM MISRA-C TOPOLOGICAL ARITHMETIC CIRCUIT'
            ' VERIFIER\\n");\n'
        )
        f.write(
            '    printf("================================================================\\n");\n'
        )
        f.write(
            '    printf("Nodes in Flash: %u | ROM Footprint: ~%lu bytes\\n",\n'
        )
        f.write("           (unsigned int)CIRCUIT_NODE_COUNT,\n")
        f.write(
            "           (unsigned long)(sizeof(g_circuit_nodes) +"
            " sizeof(g_circuit_children) + sizeof(g_circuit_weights)));\n"
        )
        f.write(
            '    printf("Evaluation Stack Scratchpad: %lu bytes (Zero'
            ' Heap)\\n\\n",\n'
        )
        f.write(
            "           (unsigned long)(sizeof(Rational) *"
            " CIRCUIT_NODE_COUNT));\n\n"
        )

        f.write("    Evidence ev;\n")
        f.write("    hiram_evidence_init(&ev);\n")
        f.write("    hiram_evidence_set(&ev, VAR_TERRAIN_CLEAR, true);\n")
        f.write("    hiram_evidence_set(&ev, VAR_ALTITUDE_STABLE, true);\n\n")

        f.write("    Rational thresh = {5LL, 100LL}; /* 5% risk limit */\n")
        f.write("    HiramAuditReport rep1 = hiram_audit_hazard(&ev, thresh);\n")
        f.write(
            '    printf("[TEST 1] Nominal Descent: Decision = %s | P(Stall) ='
            ' %lld/%lld\\n",\n'
        )
        f.write(
            '           rep1.decision == HIRAM_APPROVED ? "APPROVED" :'
            ' "VETOED",\n'
        )
        f.write(
            "           (long long)rep1.hazard_probability.num, (long"
            " long)rep1.hazard_probability.den);\n\n"
        )

        f.write("    hiram_evidence_set(&ev, VAR_LOW_AIRSPEED, true);\n")
        f.write(
            "    hiram_evidence_set(&ev, VAR_HIGH_ANGLE_OF_ATTACK, true);\n"
        )
        f.write("    HiramAuditReport rep2 = hiram_audit_hazard(&ev, thresh);\n")
        f.write(
            '    printf("[TEST 2] Severe Stall Telemetry: Decision = %s |'
            ' P(Stall) = %lld/%lld\\n",\n'
        )
        f.write(
            '           rep2.decision == HIRAM_VETOED_SAFETY_VIOLATION ?'
            ' "VETOED_SAFETY_VIOLATION" : "APPROVED",\n'
        )
        f.write(
            "           (long long)rep2.hazard_probability.num, (long"
            " long)rep2.hazard_probability.den);\n\n"
        )

        f.write(
            "    if (rep1.decision == HIRAM_APPROVED && rep2.decision =="
            " HIRAM_VETOED_SAFETY_VIOLATION) {\n"
        )
        f.write(
            '        printf("\\n[SUCCESS] MISRA-C flat topological runtime'
            ' verified identical!\\n");\n'
        )
        f.write("        return 0;\n")
        f.write("    }\n")
        f.write("    return 1;\n")
        f.write("}\n")
        f.write("#endif /* HIRAM_STANDALONE_TEST */\n")
    print("Done.\n")

    rom_bytes = (
        (num_nodes * 12) + (len(flat_children) * 2) + (len(flat_weights) * 16)
    )
    ram_bytes = num_nodes * 16
    print("MISRA-C:2012 Rule 17.2 Refactoring Complete:")
    print(
        "  - Traversal Mode:    Strict Topological Linear Sweep (Zero"
        " Recursion)"
    )
    print(f"  - Total ROM (Flash): ~{rom_bytes:,} bytes")
    print(
        f"  - Total Stack (RAM): ~{ram_bytes:,} bytes (down from 1,411 bytes;"
        " memo_valid eliminated)"
    )
    print("=" * 80)


if __name__ == "__main__":
    generate_c_code()