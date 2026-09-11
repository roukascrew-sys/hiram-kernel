"""
HIRAM Safety Kernel - Task 2.3 Verification Suite
Closed-loop trajectory qualification, exact kinematic contact resolution,
continuous evidence discretization, and moving dropout mitigation tests.
"""

import math
import unittest

from tools.inference_engine import BayesianInferenceEngine
from tools.trajectory_evaluator import (
    TrajectoryEvaluator,
    solve_step_kinematics_and_contact,
)


class TestKinematicContactResolution(unittest.TestCase):
    def test_exact_partial_step_stopping_displacement(self):
        """Auditor check: v0=0.001, a=-1, dt=0.02 must yield 0.5 um and zero velocity."""
        x_next, v_next, coll, c_dt, c_v = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.001, a_cmd=-1.0, dt=0.02, v_cruise=0.50, obstacle_x=None
        )
        self.assertAlmostEqual(x_next, 0.0000005, places=9)
        self.assertEqual(v_next, 0.0)
        self.assertFalse(coll)

    def test_original_auditor_braking_contact_probe(self):
        """Auditor check: v0=0.01, a=-1.0, obstacle at 25 um, dt=0.01."""
        x_next, v_next, coll, c_dt, c_v = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.01, a_cmd=-1.0, dt=0.01, v_cruise=0.50, obstacle_x=0.000025
        )
        self.assertTrue(coll)
        expected_v = math.sqrt(0.01**2 + 2.0 * (-1.0) * 0.000025)  # sqrt(0.000050)
        expected_dt = (expected_v - 0.01) / (-1.0)
        self.assertAlmostEqual(c_v, expected_v, places=8)
        self.assertAlmostEqual(c_dt, expected_dt, places=8)
        self.assertAlmostEqual(c_v, 0.0070710678, places=6)

    def test_contact_from_rest(self):
        """Astra Probe 2: Acceleration from rest must detect collision."""
        x_next, v_next, coll, c_dt, c_v = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.0, a_cmd=0.25, dt=0.01, v_cruise=0.50, obstacle_x=5e-6
        )
        self.assertTrue(coll)
        self.assertAlmostEqual(c_dt, 0.006324555, places=8)
        self.assertAlmostEqual(c_v, 0.001581139, places=8)

    def test_contact_while_already_cruising(self):
        """Astra Probe 1: Command +a when at cruise must clamp contact speed."""
        x_next, v_next, coll, c_dt, c_v = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.50, a_cmd=0.25, dt=0.01, v_cruise=0.50, obstacle_x=0.004
        )
        self.assertTrue(coll)
        self.assertAlmostEqual(c_dt, 0.008, places=8)
        self.assertAlmostEqual(c_v, 0.50, places=8)

    def test_contact_before_cruise(self):
        """Contact while accelerating before reaching cruise."""
        x_next, v_next, coll, c_dt, c_v = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.30, a_cmd=0.25, dt=0.01, v_cruise=0.50, obstacle_x=0.001
        )
        self.assertTrue(coll)
        expected_v = math.sqrt(0.30**2 + 2.0 * 0.25 * 0.001)
        expected_dt = (expected_v - 0.30) / 0.25
        self.assertAlmostEqual(c_v, expected_v, places=8)
        self.assertAlmostEqual(c_dt, expected_dt, places=8)

    def test_contact_after_crossing_cruise(self):
        """Contact during the constant-speed phase after crossing cruise speed."""
        x_next, v_next, coll, c_dt, c_v = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.499, a_cmd=0.25, dt=0.01, v_cruise=0.50, obstacle_x=0.003
        )
        self.assertTrue(coll)
        expected_v = 0.50
        # dt_cruise = (0.50 - 0.499) / 0.25 = 0.004 s
        # dx1 = (0.50^2 - 0.499^2) / (2 * 0.25) = 0.001998 m
        # dt_contact = 0.004 + (0.003 - 0.001998) / 0.50 = 0.006004 s
        expected_dt = 0.004 + (0.003 - 0.001998) / 0.50
        self.assertAlmostEqual(c_v, expected_v, places=8)
        self.assertAlmostEqual(c_dt, expected_dt, places=8)

    def test_cruise_speed_boundary_clamp(self):
        """Velocity never exceeds v_cruise under forward drive."""
        x_next, v_next, coll, _, _ = solve_step_kinematics_and_contact(
            x0=0.0, v0=0.499, a_cmd=0.25, dt=0.02, v_cruise=0.50, obstacle_x=None
        )
        self.assertEqual(v_next, 0.50)


class TestTrajectoryVerification(unittest.TestCase):
    def setUp(self):
        self.engine = BayesianInferenceEngine()
        self.evaluator = TrajectoryEvaluator(self.engine, dt=0.010, v_cruise=0.50)

    def test_continuous_distance_and_slip_discretization_thresholds(self):
        """Tests continuous state extraction strictly around threshold boundaries."""
        # 1. Obstacle distance threshold at 0.300 m
        e_above = self.evaluator.extract_evidence(
            x=0.0, v=0.2, a_current=0.0, obstacle_x=0.30001, track_condition="DRY", decel_capability="NOMINAL", t=0.0
        )
        self.assertEqual(e_above["LidarObs"], 0)
        self.assertEqual(e_above["TofObs"], 0)

        e_at = self.evaluator.extract_evidence(
            x=0.0, v=0.2, a_current=0.0, obstacle_x=0.30000, track_condition="DRY", decel_capability="NOMINAL", t=0.0
        )
        self.assertEqual(e_at["LidarObs"], 1)
        self.assertEqual(e_at["TofObs"], 1)

        e_below = self.evaluator.extract_evidence(
            x=0.0, v=0.2, a_current=0.0, obstacle_x=0.29999, track_condition="DRY", decel_capability="NOMINAL", t=0.0
        )
        self.assertEqual(e_below["LidarObs"], 1)
        self.assertEqual(e_below["TofObs"], 1)

        # 2. Wheel slip ratio thresholding
        e_slip_low = self.evaluator.extract_evidence(
            x=0.0, v=0.2, a_current=0.0, obstacle_x=1.0, track_condition="DRY", decel_capability="NOMINAL", t=0.0
        )
        self.assertEqual(e_slip_low["WheelSlipObs"], 0)

        e_slip_high = self.evaluator.extract_evidence(
            x=0.0, v=0.2, a_current=-1.0, obstacle_x=1.0, track_condition="ICY", decel_capability="DEGRADED", t=0.0
        )
        self.assertEqual(e_slip_high["WheelSlipObs"], 1)

    def test_nominal_clear_track_traversal(self):
        """Nominal run reaches target speed without false emergency braking."""
        res = self.evaluator.run_episode(initial_x=0.0, initial_v=0.0, max_duration=3.0, obstacle_x=None)
        self.assertFalse(res.collision)
        self.assertAlmostEqual(res.final_v, 0.50, places=2)
        self.assertNotIn("EMERGENCY_BRAKE", res.action_history)

    def test_popup_obstacle_safe_stop(self):
        """Static obstacle ahead triggers brake and halts cart with positive clearance."""
        res = self.evaluator.run_episode(
            initial_x=0.0, initial_v=0.50, max_duration=3.0, obstacle_x=0.60, track_condition="DRY"
        )
        self.assertFalse(res.collision)
        self.assertEqual(res.final_v, 0.0)
        self.assertLess(res.final_x, 0.60)
        self.assertGreater(0.60 - res.final_x, 0.0)

    def test_moving_blackout_dropout_trajectory(self):
        """Moving cart encountering a 500 ms sensor dropout stops safely before obstacle."""
        res = self.evaluator.run_episode(
            initial_x=0.0,
            initial_v=0.40,
            max_duration=4.0,
            obstacle_x=0.80,
            dropout_window=(0.20, 0.70),
            track_condition="DRY",
        )
        self.assertFalse(res.collision)
        self.assertEqual(res.final_v, 0.0)
        self.assertLess(res.final_x, 0.80)

    def test_shift6_moving_constrained_mitigation_and_impact_energy(self):
        """
        Moving start with degraded brakes (SHIFT_6) and constrained clearance.
        Evaluates early brake deadline (<= 20 ms) and >= 75% kinetic energy reduction
        measured strictly at the point of contact.
        """
        v_init = 0.35
        x_obs = 0.20  # Clearance 0.20 m < stopping distance 0.255 m ensures impact under degraded brakes
        cart_mass = 5.0

        # 1. Unmitigated baseline (zero braking capability)
        eval_unmitigated = TrajectoryEvaluator(self.engine, dt=0.010, v_cruise=0.50, cart_mass=cart_mass)
        res_baseline = eval_unmitigated.run_episode(
            initial_x=0.0,
            initial_v=v_init,
            max_duration=2.0,
            obstacle_x=x_obs,
            override_brake_decel=0.0,
        )
        self.assertTrue(res_baseline.collision)
        self.assertIsNotNone(res_baseline.contact_velocity)
        self.assertIsNotNone(res_baseline.impact_energy)
        baseline_energy = res_baseline.impact_energy

        # 2. Mitigated execution under degraded brakes (deceleration = 0.24 m/s²)
        eval_kernel = TrajectoryEvaluator(self.engine, dt=0.010, v_cruise=0.50, cart_mass=cart_mass)
        res_mitigated = eval_kernel.run_episode(
            initial_x=0.0,
            initial_v=v_init,
            max_duration=2.0,
            obstacle_x=x_obs,
            track_condition="WET",
            decel_capability="CRITICAL",
            override_brake_decel=0.24,
        )

        self.assertTrue(res_mitigated.collision)
        self.assertIsNotNone(res_mitigated.contact_velocity)
        self.assertIsNotNone(res_mitigated.impact_energy)

        # Early braking reaction time deadline: must brake within 20 ms of detection
        first_brake_step = next(
            i for i, a in enumerate(res_mitigated.action_history) if a == "EMERGENCY_BRAKE"
        )
        self.assertLessEqual(first_brake_step * eval_kernel.dt, 0.020)

        # Kinetic energy attenuation verification
        mitigated_energy = res_mitigated.impact_energy
        attenuation = (baseline_energy - mitigated_energy) / baseline_energy
        self.assertGreaterEqual(
            attenuation,
            0.75,
            f"Impact energy reduction {attenuation * 100:.2f}% did not meet 75% requirement."
        )


if __name__ == "__main__":
    unittest.main()