"""
Factor reduction & expected-loss inference engine -- verification suite
(Task 2.2).

Hardened per the Astra audit's 3 HOLD findings on commit 2708bd3e. Covers:
sealed ir_digest custody enforcement (both self-consistency and
baseline-match failure modes), exact-marginal correctness when sensors are
unobserved, the DROPOUT-state === unobserved equivalence contract
(exhaustively, across all 64 evidence combinations), directional/
qualitative correctness on deterministic corner evidence (full clear, full
obstacle, critical slip alone, partial dropout), bit-exact expected-loss
arithmetic against a hand-derived posterior, strict posterior bounds/
normalization validation, safety-first tie-breaking (including under a
non-finite expected loss, which must never resolve to ACCEL), and the
zero-likelihood fallback to the model's true physical prior (with forced
EMERGENCY_BRAKE selection and an explicit fallback_active flag) for
contradictory evidence.

Run directly:          python tests/test_inference_engine.py
Verification command:  python -m unittest tests/test_inference_engine.py -v
"""

import json
import math
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.inference_engine import BayesianInferenceEngine  # noqa: E402

MODEL_PATH = REPO_ROOT / "data" / "models" / "causal_dag.json"


def _load_model() -> dict:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _write_temp_model(model: dict) -> str:
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(model, tmp)
    tmp.close()
    return tmp.name


def _theoretical_prior_marginal(model: dict):
    """
    Independent reimplementation of P(TrueObstacle=o, DecelCapability=d),
    reading only raw CPTs/strides from the model file (not the engine under
    test), to serve as a ground-truth oracle for the marginalization tests.
    """
    nodes = model["nodes"]
    track_cpt = nodes["TrackCondition"]["cpt"]
    n_track = len(nodes["TrackCondition"]["states"])
    obstacle_cpt = nodes["TrueObstacle"]["cpt"]
    n_obstacle = len(nodes["TrueObstacle"]["states"])
    obstacle_stride = nodes["TrueObstacle"]["parent_strides"]["TrackCondition"]
    decel_cpt = nodes["DecelCapability"]["cpt"]
    n_decel = len(nodes["DecelCapability"]["states"])
    decel_stride = nodes["DecelCapability"]["parent_strides"]["TrackCondition"]

    marginal = [0.0] * (n_obstacle * n_decel)
    for t in range(n_track):
        for o in range(n_obstacle):
            p_o = obstacle_cpt[t * obstacle_stride + o]
            for d in range(n_decel):
                p_d = decel_cpt[t * decel_stride + d]
                marginal[o * n_decel + d] += track_cpt[t] * p_o * p_d
    return marginal


class IrDigestCustodyTests(unittest.TestCase):
    def test_loads_successfully_with_sealed_baseline_model(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        self.assertEqual(engine.EXPECTED_IR_DIGEST, "b39845824f903e9230c75110ffd822ccea36cd89a5ebb4eae1235ff4860b57e7")

    def test_missing_file_raises_value_error(self):
        with self.assertRaises(ValueError):
            BayesianInferenceEngine(str(REPO_ROOT / "data" / "models" / "does_not_exist.json"))

    def test_corrupted_json_raises_value_error(self):
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8")
        tmp.write("{ this is not valid json ")
        tmp.close()
        try:
            with self.assertRaises(ValueError):
                BayesianInferenceEngine(tmp.name)
        finally:
            pathlib.Path(tmp.name).unlink(missing_ok=True)

    def test_missing_ir_digest_field_raises_value_error(self):
        model = _load_model()
        del model["ir_digest"]
        path = _write_temp_model(model)
        try:
            with self.assertRaises(ValueError):
                BayesianInferenceEngine(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

    def test_content_mutated_stale_digest_left_untouched_raises_value_error(self):
        # The naive tamper: edit a probability, forget to update ir_digest.
        # Self-consistency check (recomputed vs stored) must catch this.
        model = _load_model()
        model["nodes"]["TrackCondition"]["cpt"][0] = 0.99  # was 0.70; ir_digest not updated
        path = _write_temp_model(model)
        try:
            with self.assertRaises(ValueError):
                BayesianInferenceEngine(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

    def test_content_mutated_digest_recomputed_but_not_sealed_baseline_raises_value_error(self):
        # The careful tamper: edit content AND correctly recompute+rewrite
        # ir_digest, so self-consistency alone would pass. The baseline
        # comparison against EXPECTED_IR_DIGEST must still catch this.
        from tools.inference_engine import _canonical_digest

        model = _load_model()
        model["nodes"]["TrackCondition"]["cpt"][0] = 0.99
        model["ir_digest"] = _canonical_digest(model)  # internally consistent, but wrong model
        self.assertNotEqual(model["ir_digest"], BayesianInferenceEngine.EXPECTED_IR_DIGEST)

        path = _write_temp_model(model)
        try:
            with self.assertRaises(ValueError):
                BayesianInferenceEngine(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)

    def test_wrong_but_internally_consistent_digest_on_a_trivial_model_is_rejected(self):
        # A minimal, entirely different (but internally self-sealed) model
        # must still be rejected: it is not *this* cart model.
        from tools.inference_engine import _canonical_digest

        trivial = {"schema_version": "1.0.0", "topological_order": [], "nodes": {}, "loss_matrix": None}
        trivial["digest"] = "0" * 64
        trivial["ir_digest"] = _canonical_digest(trivial)
        path = _write_temp_model(trivial)
        try:
            with self.assertRaises(ValueError):
                BayesianInferenceEngine(path)
        finally:
            pathlib.Path(path).unlink(missing_ok=True)


class MarginalizationTests(unittest.TestCase):
    def test_empty_evidence_matches_theoretical_prior_marginal(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        expected = _theoretical_prior_marginal(_load_model())
        posterior = engine.compute_posterior({})
        for got, want in zip(posterior, expected):
            self.assertAlmostEqual(got, want, delta=1e-12)
        self.assertAlmostEqual(math.fsum(posterior), 1.0, delta=1e-12)

    def test_all_none_evidence_matches_empty_evidence(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        all_none = engine.compute_posterior({"LidarObs": None, "TofObs": None, "WheelSlipObs": None})
        empty = engine.compute_posterior({})
        self.assertEqual(all_none, empty)

    def test_partially_unobserved_evidence_marginalizes_only_the_missing_sensor(self):
        # Observing only WheelSlipObs must match the joint computed by
        # explicitly summing LidarObs/TofObs over their 3 states each (the
        # brute-force definition of marginalization), confirming that
        # treating an unobserved sensor as a factor of 1.0 is equivalent
        # to actually summing over its states (since its CPT rows sum to 1).
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        model = _load_model()
        nodes = model["nodes"]

        wheel_val = 1
        analytic = engine.compute_posterior({"WheelSlipObs": wheel_val})

        # Brute-force: recompute the full joint over (T, O, D, LidarObs, TofObs)
        # and sum out LidarObs/TofObs explicitly.
        track_cpt = nodes["TrackCondition"]["cpt"]
        n_track = len(nodes["TrackCondition"]["states"])
        obstacle_cpt = nodes["TrueObstacle"]["cpt"]
        n_obstacle = len(nodes["TrueObstacle"]["states"])
        obstacle_stride = nodes["TrueObstacle"]["parent_strides"]["TrackCondition"]
        decel_cpt = nodes["DecelCapability"]["cpt"]
        n_decel = len(nodes["DecelCapability"]["states"])
        decel_stride = nodes["DecelCapability"]["parent_strides"]["TrackCondition"]
        lidar_cpt = nodes["LidarObs"]["cpt"]
        n_lidar = len(nodes["LidarObs"]["states"])
        lidar_stride = nodes["LidarObs"]["parent_strides"]["TrueObstacle"]
        tof_cpt = nodes["TofObs"]["cpt"]
        n_tof = len(nodes["TofObs"]["states"])
        tof_stride = nodes["TofObs"]["parent_strides"]["TrueObstacle"]
        wheel_cpt = nodes["WheelSlipObs"]["cpt"]
        wheel_stride = nodes["WheelSlipObs"]["parent_strides"]["DecelCapability"]

        brute = [0.0] * (n_obstacle * n_decel)
        for t in range(n_track):
            for o in range(n_obstacle):
                for d in range(n_decel):
                    base = track_cpt[t] * obstacle_cpt[t * obstacle_stride + o] * decel_cpt[t * decel_stride + d]
                    base *= wheel_cpt[d * wheel_stride + wheel_val]
                    lidar_sum = sum(lidar_cpt[o * lidar_stride + li] for li in range(n_lidar))
                    tof_sum = sum(tof_cpt[o * tof_stride + ti] for ti in range(n_tof))
                    brute[o * n_decel + d] += base * lidar_sum * tof_sum

        total = math.fsum(brute)
        brute_posterior = [v / total for v in brute]

        for got, want in zip(analytic, brute_posterior):
            self.assertAlmostEqual(got, want, delta=1e-12)


class DeterministicCornerEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianInferenceEngine(str(MODEL_PATH))

    @staticmethod
    def _obstacle_marginal(posterior, n_decel=3):
        clear = sum(posterior[0:n_decel])
        blocked = sum(posterior[n_decel : 2 * n_decel])
        return clear, blocked

    def test_full_clear_favors_clear_nominal_and_selects_accel(self):
        result = self.engine.step({"LidarObs": 0, "TofObs": 0, "WheelSlipObs": 0})
        clear, blocked = self._obstacle_marginal(result["posterior"])
        self.assertGreater(clear, 0.99)
        self.assertLess(blocked, 0.01)
        self.assertGreater(result["posterior"][0], 0.9)  # O=CLEAR, D=NOMINAL dominates
        self.assertEqual(result["selected_action"], "ACCEL")

    def test_full_obstacle_detected_favors_blocked_and_selects_emergency_brake(self):
        result = self.engine.step({"LidarObs": 1, "TofObs": 1, "WheelSlipObs": 0})
        clear, blocked = self._obstacle_marginal(result["posterior"])
        self.assertGreater(blocked, 0.99)
        self.assertLess(clear, 0.01)
        self.assertEqual(result["selected_action"], "EMERGENCY_BRAKE")

    def test_critical_slip_alone_with_obstacle_unobserved_shifts_risk_to_braking(self):
        # WheelSlipObs=HIGH(1) only; obstacle sensors unobserved (so the
        # obstacle posterior stays governed by the shared TrackCondition
        # prior, still mostly CLEAR) -- degraded traction alone must still
        # be enough to move the decision away from ACCEL.
        clear_result = self.engine.step({"LidarObs": 0, "TofObs": 0, "WheelSlipObs": 0})
        slip_result = self.engine.step({"WheelSlipObs": 1})

        clear_marginal, _ = self._obstacle_marginal(slip_result["posterior"])
        self.assertGreater(clear_marginal, 0.5, "high slip alone shouldn't itself imply an obstacle")

        # decel marginal (NOMINAL, DEGRADED, CRITICAL) must shift away from NOMINAL
        prior_result = self.engine.step({})

        def decel_marginal(posterior, n_decel=3):
            return [posterior[d] + posterior[n_decel + d] for d in range(n_decel)]

        prior_decel = decel_marginal(prior_result["posterior"])
        slip_decel = decel_marginal(slip_result["posterior"])
        self.assertLess(slip_decel[0], prior_decel[0])  # P(NOMINAL) decreased
        self.assertGreater(slip_decel[1] + slip_decel[2], prior_decel[1] + prior_decel[2])

        # and the decision must differ from the unambiguous "all clear" case
        self.assertNotEqual(slip_result["selected_action"], "ACCEL")
        self.assertNotEqual(slip_result["selected_action"], clear_result["selected_action"])

    def test_partial_dropout_is_strictly_between_full_clear_and_full_detected(self):
        full_clear = self.engine.step({"LidarObs": 0, "TofObs": 0, "WheelSlipObs": 0})
        full_detected = self.engine.step({"LidarObs": 1, "TofObs": 1, "WheelSlipObs": 0})
        partial = self.engine.step({"LidarObs": 1, "TofObs": None, "WheelSlipObs": 0})

        _, blocked_clear = self._obstacle_marginal(full_clear["posterior"])
        _, blocked_detected = self._obstacle_marginal(full_detected["posterior"])
        _, blocked_partial = self._obstacle_marginal(partial["posterior"])

        self.assertLess(blocked_clear, blocked_partial)
        self.assertLess(blocked_partial, blocked_detected)

        # and it must be a coherent probability distribution, not a NaN/garbage artifact
        self.assertAlmostEqual(math.fsum(partial["posterior"]), 1.0, delta=1e-12)
        self.assertTrue(all(0.0 <= p <= 1.0 for p in partial["posterior"]))


class DropoutEquivalenceTests(unittest.TestCase):
    """DROPOUT (state index 2) must be exactly equivalent to unobserved
    (None) under the default marginalize_dropout_state=True contract."""

    def test_exhaustive_64_combination_dropout_equals_none(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        sensor_values = (0, 1, 2, None)  # 4 values x 3 sensors = 64 combinations

        checked = 0
        for lidar in sensor_values:
            for tof in sensor_values:
                for wheel in sensor_values:
                    evidence = {"LidarObs": lidar, "TofObs": tof, "WheelSlipObs": wheel}
                    canonical = {k: (None if v == 2 else v) for k, v in evidence.items()}

                    with self.subTest(evidence=evidence):
                        posterior = engine.compute_posterior(evidence)
                        canonical_posterior = engine.compute_posterior(canonical)
                        self.assertEqual(posterior, canonical_posterior)

                        result = engine.step(evidence)
                        canonical_result = engine.step(canonical)
                        self.assertEqual(result["selected_action"], canonical_result["selected_action"])
                        self.assertEqual(result["expected_losses"], canonical_result["expected_losses"])
                    checked += 1

        self.assertEqual(checked, 64)

    def test_disabling_the_flag_makes_dropout_and_none_genuinely_differ(self):
        # Proves the equivalence above comes from the contract being active,
        # not from DROPOUT and None coincidentally always agreeing anyway.
        engine = BayesianInferenceEngine(str(MODEL_PATH), marginalize_dropout_state=False)
        posterior_none = engine.compute_posterior({"LidarObs": None, "TofObs": 1, "WheelSlipObs": 0})
        posterior_dropout = engine.compute_posterior({"LidarObs": 2, "TofObs": 1, "WheelSlipObs": 0})
        self.assertNotEqual(posterior_none, posterior_dropout)

    def test_flag_defaults_to_true(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        self.assertTrue(engine.marginalize_dropout_state)


class ExpectedLossArithmeticTests(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianInferenceEngine(str(MODEL_PATH))

    def test_hand_derived_posterior_matches_bit_exact(self):
        # A deterministic posterior concentrated entirely on
        # (TrueObstacle=CLEAR, DecelCapability=NOMINAL) -- index 0. Expected
        # loss for each action is then just that single loss-matrix cell:
        # ACCEL row0 = 0.0, COAST row0 = 2.0, EMERGENCY_BRAKE row0 = 25.0
        # (see build_default_cart_model's loss matrix).
        posterior = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        expected_losses, best_action, best_idx = self.engine.evaluate_expected_loss(posterior)

        self.assertEqual(expected_losses, [0.0, 2.0, 25.0])
        self.assertEqual(best_action, "ACCEL")
        self.assertEqual(best_idx, 0)

        # bit-exact IEEE 754 hex agreement, not just decimal closeness
        self.assertEqual(expected_losses[0].hex(), (0.0).hex())
        self.assertEqual(expected_losses[1].hex(), (2.0).hex())
        self.assertEqual(expected_losses[2].hex(), (25.0).hex())

    def test_hand_derived_posterior_on_blocked_critical_cell(self):
        # index 5 = (TrueObstacle=BLOCKED, DecelCapability=CRITICAL):
        # ACCEL=2000.0, COAST=800.0, EMERGENCY_BRAKE=50.0
        posterior = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        expected_losses, best_action, best_idx = self.engine.evaluate_expected_loss(posterior)
        self.assertEqual(expected_losses, [2000.0, 800.0, 50.0])
        self.assertEqual(best_action, "EMERGENCY_BRAKE")
        self.assertEqual(best_idx, 2)

    def test_uniform_posterior_expected_loss_matches_hand_sum(self):
        posterior = [1.0 / 6.0] * 6
        expected_losses, _, _ = self.engine.evaluate_expected_loss(posterior)

        loss_matrix = self.engine._loss_values
        for a in range(3):
            block = loss_matrix[a * 6 : a * 6 + 6]
            hand_sum = math.fsum(p * v for p, v in zip(posterior, block))
            self.assertEqual(expected_losses[a], hand_sum)

    def test_step_hex_fields_round_trip_bit_exact(self):
        result = self.engine.step({"LidarObs": 0, "TofObs": 0, "WheelSlipObs": 0})
        for value, hex_str in zip(result["posterior"], result["posterior_hex"]):
            self.assertEqual(float.fromhex(hex_str), value)
        for value, hex_str in zip(result["expected_losses"], result["expected_losses_hex"]):
            self.assertEqual(float.fromhex(hex_str), value)


class TieBreakingTests(unittest.TestCase):
    def test_tie_breaking_favors_emergency_brake(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        # Force an artificial loss matrix where every action costs exactly
        # the same for every state, so all three tie exactly -- exercising
        # evaluate_expected_loss's real arithmetic path (not a hand-picked
        # posterior trying to coincidentally tie against the real losses).
        original_loss_values = engine._loss_values
        try:
            engine._loss_values = [10.0] * 18
            posterior = [1.0 / 6.0] * 6
            expected_losses, best_action, best_idx = engine.evaluate_expected_loss(posterior)

            self.assertEqual(expected_losses, [10.0, 10.0, 10.0])
            self.assertEqual(best_action, "EMERGENCY_BRAKE")
            self.assertEqual(best_idx, 2)
        finally:
            engine._loss_values = original_loss_values

    def test_tie_breaking_prefers_coast_over_accel_when_brake_is_worse(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        original_loss_values = engine._loss_values
        try:
            # ACCEL and COAST tie at 5.0, EMERGENCY_BRAKE is strictly worse (50.0)
            engine._loss_values = [5.0] * 6 + [5.0] * 6 + [50.0] * 6
            posterior = [1.0 / 6.0] * 6
            expected_losses, best_action, best_idx = engine.evaluate_expected_loss(posterior)
            self.assertEqual(best_action, "COAST")
            self.assertEqual(best_idx, 1)
        finally:
            engine._loss_values = original_loss_values

    def test_select_best_action_index_direct(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        self.assertEqual(engine._select_best_action_index([100.0, 100.0, 100.0]), 2)  # EMERGENCY_BRAKE
        self.assertEqual(engine._select_best_action_index([1.0, 1.0, 2.0]), 1)  # ACCEL/COAST tie, COAST wins (safer)
        self.assertEqual(engine._select_best_action_index([1.0, 0.5, 0.5]), 2)  # COAST/EBRAKE tie, EBRAKE wins


class PosteriorValidationTests(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianInferenceEngine(str(MODEL_PATH))

    def test_nan_posterior_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([float("nan")] * 6)

    def test_inf_posterior_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([float("inf")] + [0.0] * 5)
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([float("-inf")] + [0.0] * 5)

    def test_negative_probability_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([-0.1, 0.2, 0.2, 0.2, 0.2, 0.3])

    def test_probability_above_one_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([1.1, -0.1, 0.0, 0.0, 0.0, 0.0])

    def test_unnormalized_posterior_rejected(self):
        # every entry individually valid ([0,1], finite) but the vector
        # doesn't sum to 1.0
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])  # sums to 1.5

    def test_slightly_off_normalization_within_tolerance_is_accepted(self):
        posterior = [1.0 / 6.0] * 6  # sums to bit-exact 1.0 via IEEE 754, well within 1e-12
        self.engine.evaluate_expected_loss(posterior)  # must not raise

    def test_wrong_length_posterior_rejected_by_evaluate_expected_loss(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([0.2, 0.2, 0.2, 0.2, 0.2])  # 5, not 6

    def test_nonfinite_expected_loss_forces_emergency_brake_never_accel(self):
        # A corrupted loss matrix (never possible with the sealed model,
        # whose values are validated finite at compile time -- this is
        # defense in depth) that makes ACCEL's expected loss NaN must not
        # be allowed to win a NaN-involving comparison by accident: the
        # engine must force EMERGENCY_BRAKE outright, and never ACCEL.
        original_loss_values = self.engine._loss_values
        try:
            corrupted = list(original_loss_values)
            corrupted[0] = float("nan")  # ACCEL's first cell
            self.engine._loss_values = corrupted

            posterior = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # concentrates all mass on that cell
            expected_losses, best_action, best_idx = self.engine.evaluate_expected_loss(posterior)

            self.assertTrue(math.isnan(expected_losses[0]))
            self.assertNotEqual(best_action, "ACCEL")
            self.assertEqual(best_action, "EMERGENCY_BRAKE")
            self.assertEqual(best_idx, self.engine._emergency_brake_index)
        finally:
            self.engine._loss_values = original_loss_values

    def test_infinite_expected_loss_also_forces_emergency_brake(self):
        original_loss_values = self.engine._loss_values
        try:
            corrupted = list(original_loss_values)
            corrupted[6] = float("inf")  # COAST's first cell
            self.engine._loss_values = corrupted

            posterior = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            _expected_losses, best_action, _best_idx = self.engine.evaluate_expected_loss(posterior)
            self.assertEqual(best_action, "EMERGENCY_BRAKE")
        finally:
            self.engine._loss_values = original_loss_values


class ZeroLikelihoodGuardTests(unittest.TestCase):
    def test_contradictory_evidence_falls_back_to_true_prior_without_crashing(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        original_lidar_cpt = engine._sensor_cpt["LidarObs"]
        try:
            # Force P(LidarObs=DETECTED | TrueObstacle=*) = 0.0 for every
            # TrueObstacle state, i.e. "this sensor physically cannot report
            # DETECTED" -- then observing DETECTED(1) is a genuine
            # contradiction under the (patched) model, driving the joint
            # likelihood to exactly 0.0 regardless of TrackCondition/
            # DecelCapability. (Not state 2/DROPOUT: with the default
            # dropout<->None contract, evidence=2 would be marginalized
            # out before ever reaching this CPT, and couldn't force a
            # contradiction -- see DropoutEquivalenceTests.)
            engine._sensor_cpt["LidarObs"] = [0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
            posterior, fallback_active = engine._compute_posterior_with_diagnostics({"LidarObs": 1})

            self.assertTrue(fallback_active)
            self.assertEqual(len(posterior), 6)
            self.assertTrue(all(math.isfinite(p) for p in posterior))
            self.assertAlmostEqual(math.fsum(posterior), 1.0, delta=1e-12)
            # falls back to the model's true physical prior, not an
            # arbitrary uniform guess
            self.assertEqual(posterior, engine._prior_marginal)
        finally:
            engine._sensor_cpt["LidarObs"] = original_lidar_cpt

    def test_fallback_forces_emergency_brake_and_flags_fallback_active(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        original_lidar_cpt = engine._sensor_cpt["LidarObs"]
        try:
            engine._sensor_cpt["LidarObs"] = [0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
            result = engine.step({"LidarObs": 1})

            self.assertTrue(result["fallback_active"])
            self.assertEqual(result["selected_action"], "EMERGENCY_BRAKE")
            self.assertEqual(result["selected_action_index"], engine._emergency_brake_index)
            self.assertEqual(result["posterior"], engine._prior_marginal)
        finally:
            engine._sensor_cpt["LidarObs"] = original_lidar_cpt

    def test_ordinary_evidence_never_sets_fallback_active(self):
        engine = BayesianInferenceEngine(str(MODEL_PATH))
        for evidence in ({}, {"LidarObs": 0}, {"LidarObs": 1, "TofObs": 1, "WheelSlipObs": 0}):
            with self.subTest(evidence=evidence):
                result = engine.step(evidence)
                self.assertFalse(result["fallback_active"])

    def test_real_model_has_no_naturally_zero_probability_evidence(self):
        # Documents *why* the guard above needs to monkeypatch to trigger:
        # every CPT entry in the real sealed model is strictly positive, so
        # no genuine evidence combination on the unmodified model can ever
        # drive the likelihood to exactly zero.
        model = _load_model()
        for node in model["nodes"].values():
            for p in node["cpt"]:
                self.assertGreater(p, 0.0, f"{node['name']} has a zero-probability CPT entry")


class EvidenceValidationTests(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianInferenceEngine(str(MODEL_PATH))

    def test_unknown_evidence_key_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.compute_posterior({"NotASensor": 0})

    def test_out_of_range_evidence_value_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.compute_posterior({"LidarObs": 3})
        with self.assertRaises(ValueError):
            self.engine.compute_posterior({"LidarObs": -1})

    def test_non_integer_evidence_value_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.compute_posterior({"LidarObs": 1.0})
        with self.assertRaises(ValueError):
            self.engine.compute_posterior({"LidarObs": "DETECTED"})
        with self.assertRaises(ValueError):
            self.engine.compute_posterior({"LidarObs": True})

    def test_wrong_length_posterior_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.evaluate_expected_loss([1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
