"""
Compiler-driven causal DAG construction -- verification suite (Task 2.1).

Covers: topological ordering (parent precedence, determinism, cycle
detection), CPT row-sum validation, stride computation and flat indexing
against known table coordinates (for both node CPTs and the loss matrix),
bit-exact float.hex()/float.fromhex() round-tripping, loss-matrix shape
validation, and the tri-state dropout-index invariant for sensor nodes.

Run directly:          python tests/test_causal_dag.py
Verification command:  python -m unittest tests/test_causal_dag.py -v
"""

import json
import math
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.cart_model import CausalDAGBuilder, build_default_cart_model  # noqa: E402


def _default_compiled():
    return build_default_cart_model().compile()


class TopologicalSortTests(unittest.TestCase):
    def test_default_model_order_is_deterministic_and_parents_precede_children(self):
        order = build_default_cart_model().topological_sort()
        position = {name: i for i, name in enumerate(order)}

        self.assertEqual(set(order), {
            "TrackCondition", "TrueObstacle", "DecelCapability",
            "LidarObs", "TofObs", "WheelSlipObs",
        })

        expected_edges = [
            ("TrackCondition", "TrueObstacle"),
            ("TrackCondition", "DecelCapability"),
            ("TrueObstacle", "LidarObs"),
            ("TrueObstacle", "TofObs"),
            ("DecelCapability", "WheelSlipObs"),
        ]
        for parent, child in expected_edges:
            self.assertLess(position[parent], position[child], f"{parent} must precede {child}")

    def test_topological_sort_is_repeatable(self):
        builder = build_default_cart_model()
        self.assertEqual(builder.topological_sort(), builder.topological_sort())

    def test_topological_sort_insertion_order_tiebreak(self):
        # Three independent roots with no edges between them: with no
        # dependency to break the tie, insertion order must decide it.
        builder = CausalDAGBuilder()
        builder.add_node("Z", ["a", "b"], [], [0.5, 0.5])
        builder.add_node("A", ["a", "b"], [], [0.5, 0.5])
        builder.add_node("M", ["a", "b"], [], [0.5, 0.5])
        self.assertEqual(builder.topological_sort(), ["Z", "A", "M"])

    def test_unknown_parent_reference_raises_value_error(self):
        builder = CausalDAGBuilder()
        builder.add_node("Child", ["x", "y"], ["NoSuchParent"], [0.5, 0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            builder.topological_sort()

    def test_cycle_detection_raises_value_error(self):
        # A -> B -> C -> A. add_node() must not itself reject the forward
        # reference (C doesn't exist yet when A is added) -- only
        # topological_sort()/compile() may reject the completed cycle.
        builder = CausalDAGBuilder()
        builder.add_node("A", ["x", "y"], ["C"], [0.5, 0.5, 0.5, 0.5])
        builder.add_node("B", ["x", "y"], ["A"], [0.5, 0.5, 0.5, 0.5])
        builder.add_node("C", ["x", "y"], ["B"], [0.5, 0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            builder.topological_sort()

    def test_self_referential_parent_rejected_at_add_time(self):
        builder = CausalDAGBuilder()
        with self.assertRaises(ValueError):
            builder.add_node("Self", ["x", "y"], ["Self"], [0.5, 0.5])

    def test_cycle_with_an_acyclic_branch_still_detected(self):
        # A larger graph where only part of it is cyclic: cycle detection
        # must still fire, and the acyclic branch must not mask it.
        builder = CausalDAGBuilder()
        builder.add_node("Root", ["a", "b"], [], [0.5, 0.5])
        builder.add_node("Leaf", ["a", "b"], ["Root"], [0.5, 0.5, 0.5, 0.5])
        builder.add_node("X", ["a", "b"], ["Y"], [0.5, 0.5, 0.5, 0.5])
        builder.add_node("Y", ["a", "b"], ["X"], [0.5, 0.5, 0.5, 0.5])
        with self.assertRaises(ValueError):
            builder.topological_sort()


class CptRowSumTests(unittest.TestCase):
    def test_default_model_all_rows_sum_to_one(self):
        ir = _default_compiled()
        for name, node in ir["nodes"].items():
            states = node["states"]
            cpt = node["cpt"]
            n_rows = len(cpt) // len(states)
            for row in range(n_rows):
                row_vals = cpt[row * len(states) : (row + 1) * len(states)]
                self.assertTrue(
                    math.isclose(math.fsum(row_vals), 1.0, rel_tol=1e-9, abs_tol=1e-9),
                    f"{name} row {row} = {row_vals} sums to {math.fsum(row_vals)}",
                )

    def test_bad_row_sum_rejected_at_compile(self):
        builder = CausalDAGBuilder()
        builder.add_node("Bad", ["x", "y"], [], [0.5, 0.4])  # sums to 0.9
        with self.assertRaises(ValueError):
            builder.compile()

    def test_row_sum_tolerance_accepts_float_noise_rejects_real_error(self):
        builder_ok = CausalDAGBuilder()
        # classic float64 imprecision: 0.1 + 0.2 != 0.3 exactly, but the sum
        # of a full row here is within math.isclose's tolerance of 1.0
        builder_ok.add_node("AlmostOne", ["x", "y", "z"], [], [0.1, 0.2, 0.7])
        ir = builder_ok.compile()  # must not raise
        self.assertIn("AlmostOne", ir["nodes"])

        builder_bad = CausalDAGBuilder()
        builder_bad.add_node("RealError", ["x", "y"], [], [0.5, 0.6])  # sums to 1.1, not noise
        with self.assertRaises(ValueError):
            builder_bad.compile()

    def test_cpt_length_mismatch_rejected(self):
        builder = CausalDAGBuilder()
        builder.add_node("Parent", ["a", "b"], [], [0.5, 0.5])
        builder.add_node("Child", ["x", "y"], ["Parent"], [1.0, 0.0, 0.0])  # needs 4 entries, has 3
        with self.assertRaises(ValueError):
            builder.compile()


class StrideAndIndexingTests(unittest.TestCase):
    def test_root_node_has_no_parent_strides(self):
        builder = build_default_cart_model()
        strides, total = builder.compute_strides("TrackCondition")
        self.assertEqual(strides, {})
        self.assertEqual(total, 3)

    def test_single_parent_node_stride_matches_own_state_count(self):
        builder = build_default_cart_model()
        strides, total = builder.compute_strides("TrueObstacle")
        # shape (3 TrackCondition states, 2 TrueObstacle states); parent
        # stride = product of dims after it = 2 (its own state count)
        self.assertEqual(strides, {"TrackCondition": 2})
        self.assertEqual(total, 6)

    def test_decel_capability_strides(self):
        builder = build_default_cart_model()
        strides, total = builder.compute_strides("DecelCapability")
        self.assertEqual(strides, {"TrackCondition": 3})
        self.assertEqual(total, 9)

    def test_flat_cpt_index_matches_known_table_coordinates(self):
        builder = build_default_cart_model()
        # TrueObstacle cpt = [0.95,0.05, 0.90,0.10, 0.85,0.15]
        # row 0 = TrackCondition=DRY(0), row 1 = WET(1), row 2 = ICY(2)
        self.assertEqual(builder.flat_cpt_index("TrueObstacle", [0], 0), 0)  # DRY, CLEAR
        self.assertEqual(builder.flat_cpt_index("TrueObstacle", [0], 1), 1)  # DRY, BLOCKED
        self.assertEqual(builder.flat_cpt_index("TrueObstacle", [1], 0), 2)  # WET, CLEAR
        self.assertEqual(builder.flat_cpt_index("TrueObstacle", [2], 1), 5)  # ICY, BLOCKED (last cell)

        ir = _default_compiled()
        cpt = ir["nodes"]["TrueObstacle"]["cpt"]
        self.assertEqual(cpt[builder.flat_cpt_index("TrueObstacle", [1], 0)], 0.90)
        self.assertEqual(cpt[builder.flat_cpt_index("TrueObstacle", [2], 1)], 0.15)

    def test_flat_cpt_index_for_root_node_is_just_state_index(self):
        builder = build_default_cart_model()
        self.assertEqual(builder.flat_cpt_index("TrackCondition", [], 0), 0)
        self.assertEqual(builder.flat_cpt_index("TrackCondition", [], 2), 2)

    def test_flat_cpt_index_out_of_range_raises(self):
        builder = build_default_cart_model()
        with self.assertRaises(IndexError):
            builder.flat_cpt_index("TrueObstacle", [0], 5)

    def test_flat_cpt_index_wrong_arity_raises(self):
        builder = build_default_cart_model()
        with self.assertRaises(ValueError):
            builder.flat_cpt_index("TrueObstacle", [0, 0], 0)  # TrueObstacle has 1 parent, not 2

    def test_compute_strides_unknown_node_raises(self):
        builder = build_default_cart_model()
        with self.assertRaises(ValueError):
            builder.compute_strides("NoSuchNode")


class LossMatrixTests(unittest.TestCase):
    def test_loss_matrix_strides(self):
        builder = build_default_cart_model()
        action_stride, dep_strides, total = builder.compute_loss_strides()
        # shape (3 actions, 2 TrueObstacle states, 3 DecelCapability states)
        # action_stride = 2*3 = 6; TrueObstacle stride = 3; DecelCapability stride = 1
        self.assertEqual(action_stride, 6)
        self.assertEqual(dep_strides, {"TrueObstacle": 3, "DecelCapability": 1})
        self.assertEqual(total, 18)

    def test_loss_matrix_boundary_cells(self):
        builder = build_default_cart_model()
        ir = _default_compiled()
        matrix = ir["loss_matrix"]["matrix"]

        # first cell: ACCEL(0), TrueObstacle=CLEAR(0), DecelCapability=NOMINAL(0)
        first = builder.flat_loss_index(0, [0, 0])
        self.assertEqual(first, 0)
        self.assertEqual(matrix[first], 0.0)

        # last cell: EMERGENCY_BRAKE(2), TrueObstacle=BLOCKED(1), DecelCapability=CRITICAL(2)
        last = builder.flat_loss_index(2, [1, 2])
        self.assertEqual(last, 17)
        self.assertEqual(matrix[last], 50.0)

    def test_loss_matrix_known_interior_cells(self):
        builder = build_default_cart_model()
        ir = _default_compiled()
        matrix = ir["loss_matrix"]["matrix"]

        # ACCEL into a BLOCKED obstacle with CRITICAL brakes: worst accel outcome
        idx = builder.flat_loss_index(0, [1, 2])  # ACCEL, BLOCKED, CRITICAL
        self.assertEqual(matrix[idx], 2000.0)

        # EMERGENCY_BRAKE with a CLEAR track (unnecessary braking, flat nuisance cost)
        for decel_idx in range(3):
            idx = builder.flat_loss_index(2, [0, decel_idx])  # EMERGENCY_BRAKE, CLEAR, *
            self.assertEqual(matrix[idx], 25.0)

        # EMERGENCY_BRAKE, BLOCKED, NOMINAL: the cheapest "obstacle present" outcome
        idx = builder.flat_loss_index(2, [1, 0])
        self.assertEqual(matrix[idx], 5.0)

    def test_loss_matrix_dimension_mismatch_rejected(self):
        builder = build_default_cart_model()
        builder.set_loss_matrix(  # overwrites the default model's loss matrix
            actions=["A", "B"],
            state_dependencies=["TrueObstacle"],
            matrix=[1.0, 2.0],  # needs 2 actions * 2 states = 4 entries
        )
        with self.assertRaises(ValueError):
            builder.compile()

    def test_loss_matrix_unknown_dependency_rejected(self):
        builder = build_default_cart_model()
        builder.set_loss_matrix(
            actions=["A"],
            state_dependencies=["NoSuchNode"],
            matrix=[1.0, 2.0],
        )
        with self.assertRaises(ValueError):
            builder.compile()

    def test_model_without_loss_matrix_compiles_with_none(self):
        builder = CausalDAGBuilder()
        builder.add_node("Solo", ["a", "b"], [], [0.5, 0.5])
        ir = builder.compile()
        self.assertIsNone(ir["loss_matrix"])


class FloatHexRoundTripTests(unittest.TestCase):
    def test_cpt_hex_round_trips_bit_exact(self):
        ir = _default_compiled()
        for node in ir["nodes"].values():
            for value, hex_str in zip(node["cpt"], node["cpt_hex"]):
                self.assertEqual(float.fromhex(hex_str), value)
                # and the hex string itself round-trips to an identical hex string
                self.assertEqual(float.fromhex(hex_str).hex(), hex_str)

    def test_loss_matrix_hex_round_trips_bit_exact(self):
        ir = _default_compiled()
        lm = ir["loss_matrix"]
        for value, hex_str in zip(lm["matrix"], lm["matrix_hex"]):
            self.assertEqual(float.fromhex(hex_str), value)

    def test_hex_encoding_distinguishes_values_decimal_repr_would_conflate(self):
        # 0.1 + 0.2 != 0.3 in float64; their hex encodings must differ even
        # though naive decimal formatting could round both to "0.3".
        a = 0.1 + 0.2
        b = 0.3
        self.assertNotEqual(a, b)
        self.assertNotEqual(a.hex(), b.hex())

    def test_compiled_ir_is_json_round_trippable_and_hex_survives(self):
        ir = _default_compiled()
        serialized = json.dumps(ir)
        restored = json.loads(serialized)
        for node in restored["nodes"].values():
            for value, hex_str in zip(node["cpt"], node["cpt_hex"]):
                self.assertEqual(float.fromhex(hex_str), value)

    def test_digest_is_stable_across_recompiles_and_sensitive_to_changes(self):
        ir_a = build_default_cart_model().compile()
        ir_b = build_default_cart_model().compile()
        self.assertEqual(ir_a["digest"], ir_b["digest"])
        self.assertEqual(len(ir_a["digest"]), 64)  # sha256 hex length
        int(ir_a["digest"], 16)  # must be valid hex

        mutated = CausalDAGBuilder()
        mutated.add_node("TrackCondition", ["DRY", "WET", "ICY"], [], [0.70, 0.20, 0.10])
        mutated.add_node(
            "TrueObstacle", ["CLEAR", "BLOCKED"], ["TrackCondition"],
            [0.95, 0.05, 0.90, 0.10, 0.84, 0.16],  # last row tweaked (still sums to 1.0)
        )
        ir_c = mutated.compile()
        self.assertNotEqual(ir_a["digest"], ir_c["digest"])


class TriStateDropoutInvariantTests(unittest.TestCase):
    def test_sensor_nodes_flag_dropout_state_index(self):
        ir = _default_compiled()
        for sensor_node in ("LidarObs", "TofObs", "WheelSlipObs"):
            node = ir["nodes"][sensor_node]
            self.assertEqual(node["states"][2], "DROPOUT")
            self.assertEqual(node["dropout_state_index"], 2)
            self.assertTrue(node["is_sensor_variable"])

    def test_non_sensor_nodes_have_no_dropout_state(self):
        ir = _default_compiled()
        for latent_node in ("TrackCondition", "TrueObstacle", "DecelCapability"):
            node = ir["nodes"][latent_node]
            self.assertIsNone(node["dropout_state_index"])
            self.assertFalse(node["is_sensor_variable"])
            self.assertNotIn("DROPOUT", node["states"])


class CliGenerationTests(unittest.TestCase):
    def test_generated_file_matches_fresh_compile(self):
        output_path = REPO_ROOT / "data" / "models" / "causal_dag.json"
        self.assertTrue(output_path.exists(), f"{output_path} was not generated; run `python tools/cart_model.py`")

        on_disk = json.loads(output_path.read_text(encoding="utf-8"))
        fresh = _default_compiled()
        self.assertEqual(on_disk["digest"], fresh["digest"])
        self.assertEqual(on_disk["topological_order"], fresh["topological_order"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
