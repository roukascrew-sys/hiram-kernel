"""
Closed-loop integration & trajectory verification -- verification suite
(Task 2.3, hardened per the Astra audit's HOLD findings on commits
140b8a03 and 39ba0dd8).

Covers: exact analytical partial-step physics (zero-velocity crossing and
v_cruise clamping, both bit-precise, no overshoot for any dt), exact
*intra-step* collision kinematics (contact_time/contact_velocity/
impact_energy isolated via Torricelli's equation, rather than reporting
whatever velocity a full, obstacle-oblivious dt of integration happened to
settle to -- the specific defect this hardening pass fixes, including the
auditor's exact regression probe), deterministic continuous evidence
discretization (no randomness anywhere), nominal cruise-to-target-velocity
with genuinely zero false emergency stops (trivial now that evidence has no
noise), immediate EMERGENCY_BRAKE response to a pop-up obstacle with a
bit-exact cross-check against sim.plant.analytic_plant's independent
closed-form stopping-distance solution, a moving cart surviving a 500 ms
telemetry blackout without collision, a SHIFT_6 degraded-brake mitigation
scenario (operational wheel telemetry -- sensor-blackout qualification is
the moving-blackout tests' scope, not this one's) meeting both an explicit
<=20ms reaction-deadline requirement and a >=75% terminal-impact-kinetic-
energy reduction versus an unmitigated baseline (measured via
contact_velocity/impact_energy on both sides, not post-braking final_v),
forced zero-likelihood fallback producing a fail-safe stop, and focused
unit tests of the evidence generator and zero-order-hold control-loop
mechanics.

Run directly:          python tests/test_trajectory_verification.py
Verification command:  python -m unittest tests/test_trajectory_verification.py -v
"""

import dataclasses
import math
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sim.plant import PlantParameters, PlantState  # noqa: E402
from sim.plant.analytic_plant import stopping_position  # noqa: E402
from sim.scenarios.generator import DEFAULT_MASTER_SEED, generate_instance  # noqa: E402
from sim.scenarios.strata import StratumType  # noqa: E402
from tools.inference_engine import BayesianInferenceEngine  # noqa: E402
from tools.trajectory_evaluator import (  # noqa: E402
    TrajectoryEvaluator,
    TrajectoryProfile,
    _true_decel_index,
)

# One shared, digest-verified engine instance -- loading it repeatedly per
# test would be wasteful and the model is immutable read-only state anyway
# (individual tests that need a corrupted CPT restore it in a finally block).
_ENGINE = BayesianInferenceEngine()


def _far_obstacle(scenario):
    """A copy of `scenario` with the obstacle pushed out of reach, for
    scenarios that should behave as if there's effectively no obstacle."""
    return dataclasses.replace(scenario, obstacle_position=1000.0)


def _unmitigated_impact_result(evaluator: TrajectoryEvaluator, v0: float, profile: TrajectoryProfile):
    """
    Rolls out a full episode with evidence forced to always read CLEAR (no
    perception of the obstacle, no braking reaction at all), and returns the
    resulting EpisodeResult. Used as the "unmitigated" baseline for
    kinetic-energy-reduction comparisons: since this goes through the same
    run_episode() and _advance_physics_with_collision() as the mitigated
    run, both sides of the comparison get their contact_velocity/
    impact_energy from the same exact intra-step collision resolution --
    not a hand-rolled physics loop with its own (potentially different)
    end-of-step overshoot behavior.
    """
    return evaluator.run_episode(
        PlantState(position=0.0, velocity=v0, acceleration=0.0),
        profile,
        evidence_override=lambda t: {"LidarObs": 0, "TofObs": 0, "WheelSlipObs": 0},
    )


class ExactPhysicsIntegrationTests(unittest.TestCase):
    def setUp(self):
        scenario = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        self.evaluator = TrajectoryEvaluator(_ENGINE, scenario)

    def test_exact_zero_velocity_crossing_within_a_step(self):
        # v=0.001 m/s, decelerating at 1.0 m/s^2 over dt=0.02s would go
        # negative under naive Euler (0.001 - 0.02 = -0.019); the exact
        # partial-step integration instead stops exactly at
        # dt_stop = v/|a| = 0.001s, covering dx = v^2/(2|a|) = 5e-7 m.
        x, v = self.evaluator._advance_physics(0.0, 0.001, -1.0, 0.02)
        self.assertEqual(x, 5e-7)
        self.assertEqual(v, 0.0)

    def test_zero_crossing_displacement_matches_hand_derivation_for_other_values(self):
        x, v = self.evaluator._advance_physics(1.0, 0.5, -2.0, 1.0)
        # v + a*dt = 0.5 - 2.0 = -1.5 <= 0 -> stop branch
        # dx = -(0.5^2)/(2*-2.0) = -(0.25)/(-4.0) = 0.0625
        self.assertAlmostEqual(x, 1.0 + 0.0625, delta=1e-15)
        self.assertEqual(v, 0.0)

    def test_deceleration_not_reaching_zero_uses_the_plain_kinematic_branch(self):
        x, v = self.evaluator._advance_physics(0.0, 1.0, -0.1, 0.1)
        # v + a*dt = 1.0 - 0.01 = 0.99 > 0 -> plain branch
        self.assertAlmostEqual(v, 0.99, delta=1e-15)
        self.assertAlmostEqual(x, 1.0 * 0.1 + 0.5 * -0.1 * 0.1 * 0.1, delta=1e-15)

    def test_cruise_clamp_reaches_exactly_v_cruise_with_zero_overshoot_across_any_dt(self):
        self.evaluator.v_cruise = 0.50
        # Any dt large enough to cross the (0.5 - 0.499)/0.25 = 0.004s
        # threshold must land on *exactly* 0.50, never above it, regardless
        # of how much larger dt is than that threshold.
        for dt in (0.004, 0.01, 0.02, 0.1, 1.0, 5.0):
            with self.subTest(dt=dt):
                x, v = self.evaluator._advance_physics(0.0, 0.499, 0.25, dt)
                self.assertEqual(v, 0.50)
                self.assertGreater(x, 0.0)

    def test_cruise_clamp_does_not_fire_when_the_step_is_too_small_to_reach_it(self):
        self.evaluator.v_cruise = 0.50
        x, v = self.evaluator._advance_physics(0.0, 0.499, 0.25, 0.001)
        self.assertLess(v, 0.50)  # correctly not yet at cruise -- no premature clamp

    def test_acceleration_never_overshoots_v_cruise(self):
        self.evaluator.v_cruise = 0.50
        for dt in (0.001, 0.004, 0.01, 0.1, 1.0, 10.0):
            _, v = self.evaluator._advance_physics(0.0, 0.499, 0.25, dt)
            self.assertLessEqual(v, 0.50)

    def test_zero_acceleration_is_trivial_constant_velocity(self):
        x, v = self.evaluator._advance_physics(2.0, 0.3, 0.0, 0.5)
        self.assertEqual(v, 0.3)
        self.assertAlmostEqual(x, 2.0 + 0.15, delta=1e-15)


class IntraStepCollisionKinematicsTests(unittest.TestCase):
    """
    Regression coverage for the defect identified in audit review of commit
    39ba0dd8: the original collision check compared *end-of-step* position
    against obstacle_x after a full, obstacle-oblivious dt of integration,
    so a step that stopped the cart exactly at its kinematic limit past the
    obstacle reported the already-settled v=0.0 as the "impact velocity"
    instead of the strictly higher velocity the cart actually had at the
    moment it passed the obstacle, earlier in that same step.
    """

    def setUp(self):
        scenario = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        self.evaluator = TrajectoryEvaluator(_ENGINE, scenario)

    def test_auditor_probe_direct_method_call(self):
        # v0=0.01 m/s, a=-1.0 m/s^2, dt=0.010s: the full obstacle-oblivious
        # step would travel v^2/(2|a|) = 50 um before settling to v=0.0 --
        # but the obstacle is only 25 um away, half that distance, so true
        # contact happens partway through the step at a strictly positive
        # velocity: v_contact = sqrt(v0^2 + 2*a*d) = sqrt(0.0001 - 0.00005)
        # = sqrt(0.00005) ~= 0.0070710678 m/s.
        x, v, contact = self.evaluator._advance_physics_with_collision(
            0.0, 0.01, -1.0, 0.010, 0.0, 0.000025
        )
        self.assertIsNotNone(contact)
        self.assertEqual(x, 0.000025)
        self.assertEqual(v, contact.contact_velocity)
        self.assertTrue(math.isclose(contact.contact_velocity, 0.0070710678, rel_tol=1e-5))
        self.assertGreater(contact.impact_energy, 0.0)
        self.assertGreater(contact.contact_time, 0.0)
        self.assertLess(contact.contact_time, 0.010)  # contact is strictly within the step

    def test_auditor_probe_via_full_episode_result(self):
        # The same probe, but exercised end to end through run_episode(),
        # asserting on the exact field names the episode summary must
        # expose: result.collision, result.contact_velocity,
        # result.impact_energy.
        scenario = dataclasses.replace(
            generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0),
            obstacle_position=0.000025,
            braking_deceleration=1.0,
        )
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=0.0, dt=0.010, evidence_period_s=0.010, max_duration_s=1.0)
        result = evaluator.run_episode(
            PlantState(position=0.0, velocity=0.01, acceleration=0.0),
            profile,
            evidence_override=lambda t: {"LidarObs": 1, "TofObs": 1, "WheelSlipObs": 0},
        )

        self.assertTrue(result.collision is True)
        self.assertTrue(math.isclose(result.contact_velocity, 0.0070710678, rel_tol=1e-5))
        self.assertGreater(result.impact_energy, 0.0)

    def test_no_collision_case_leaves_contact_fields_none(self):
        x, v, contact = self.evaluator._advance_physics_with_collision(
            0.0, 0.01, -1.0, 0.010, 0.0, 1.0  # obstacle far away -- cart stops well short of it
        )
        self.assertIsNone(contact)

    def test_independent_torricelli_cross_check_across_multiple_combinations(self):
        # Compares the method's contact_velocity against a Torricelli
        # prediction computed independently in this test (not by calling
        # into the same formula inside the evaluator), across a spread of
        # (v0, a, d_clear) combinations. Each combination's own stopping
        # distance (v0^2/(2|a|)) exceeds d_clear, so the obstacle would
        # eventually be reached if given long enough -- dt is set to more
        # than twice each case's own full-stop time so the single-step,
        # obstacle-oblivious displacement actually reaches that far within
        # this one call (a too-small dt would correctly report *no*
        # collision this step, which is a different, valid code path
        # covered by test_no_collision_case_leaves_contact_fields_none).
        cases = [
            (0.01, -1.0, 0.000025),
            (0.5, -0.2, 0.3),
            (0.3, -0.177, 0.1),
            (1.0, -0.5, 0.5),
            (0.05, -10.0, 0.0001),
            (0.2, -0.05, 0.15),
        ]
        for v0, a, d_clear in cases:
            t_stop = v0 / abs(a)
            self.assertGreater(v0 * v0 / (2.0 * abs(a)), d_clear)  # test setup sanity: obstacle is reachable
            dt = 2.0 * t_stop + 1.0
            with self.subTest(v0=v0, a=a, d_clear=d_clear, dt=dt):
                expected_v_contact = math.sqrt(max(0.0, v0 * v0 + 2.0 * a * d_clear))
                x, v, contact = self.evaluator._advance_physics_with_collision(0.0, v0, a, dt, 0.0, d_clear)
                self.assertIsNotNone(contact, f"expected a collision for {(v0, a, d_clear, dt)}")
                self.assertAlmostEqual(contact.contact_velocity, expected_v_contact, delta=1e-9)
                self.assertAlmostEqual(v, expected_v_contact, delta=1e-9)
                self.assertAlmostEqual(x, d_clear, delta=1e-9)

                expected_impact_energy = 0.5 * self.evaluator.mass * expected_v_contact ** 2
                self.assertAlmostEqual(contact.impact_energy, expected_impact_energy, delta=1e-12)

    def test_impact_energy_uses_the_evaluators_mass(self):
        evaluator = TrajectoryEvaluator(_ENGINE, self.evaluator.scenario, mass=2.5)
        _, _, contact = evaluator._advance_physics_with_collision(0.0, 0.01, -1.0, 0.010, 0.0, 0.000025)
        self.assertAlmostEqual(contact.impact_energy, 0.5 * 2.5 * contact.contact_velocity ** 2, delta=1e-15)


class DeterministicEvidenceDiscretizationTests(unittest.TestCase):
    def setUp(self):
        scenario = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        self.evaluator = TrajectoryEvaluator(_ENGINE, scenario)

    def test_no_obstacle_reads_clear_on_both_range_sensors(self):
        ev = self.evaluator.extract_evidence(0.0, 0.0, None, "DRY", "NOMINAL", 0.0, None)
        self.assertEqual(ev["LidarObs"], 0)
        self.assertEqual(ev["TofObs"], 0)

    def test_distance_threshold_boundary_is_deterministic(self):
        just_outside = self.evaluator.extract_evidence(0.0, 0.0, 0.301, "DRY", "NOMINAL", 0.0, None)
        exactly_at = self.evaluator.extract_evidence(0.0, 0.0, 0.300, "DRY", "NOMINAL", 0.0, None)
        just_inside = self.evaluator.extract_evidence(0.0, 0.0, 0.299, "DRY", "NOMINAL", 0.0, None)
        self.assertEqual(just_outside["LidarObs"], 0)
        self.assertEqual(exactly_at["LidarObs"], 1)  # <=0.300 is DETECTED, per spec
        self.assertEqual(just_inside["LidarObs"], 1)
        # deterministic: repeated calls with identical inputs give identical output
        self.assertEqual(
            self.evaluator.extract_evidence(0.0, 0.0, 0.300, "DRY", "NOMINAL", 0.0, None),
            exactly_at,
        )

    def test_dropout_window_blanks_only_range_sensors_not_wheel_slip(self):
        # Per the exact specification: a dropout window forces LidarObs/
        # TofObs to None, but WheelSlipObs is not affected by it (modeling
        # a range-sensor telemetry outage that doesn't also take down an
        # independent wheel-encoder channel).
        ev = self.evaluator.extract_evidence(0.0, 0.5, 0.1, "ICY", "CRITICAL", 0.5, (0.0, 1.0))
        self.assertIsNone(ev["LidarObs"])
        self.assertIsNone(ev["TofObs"])
        self.assertIn(ev["WheelSlipObs"], (0, 1))

    def test_dropout_window_boundary_is_inclusive_both_ends(self):
        window = (0.2, 0.7)
        at_start = self.evaluator.extract_evidence(0.0, 0.0, None, "DRY", "NOMINAL", 0.2, window)
        at_end = self.evaluator.extract_evidence(0.0, 0.0, None, "DRY", "NOMINAL", 0.7, window)
        just_after = self.evaluator.extract_evidence(0.0, 0.0, None, "DRY", "NOMINAL", 0.700001, window)
        self.assertIsNone(at_start["LidarObs"])
        self.assertIsNone(at_end["LidarObs"])
        self.assertEqual(just_after["LidarObs"], 0)

    def test_no_randomness_anywhere_in_evidence_generation(self):
        args = (0.1, 0.3, 0.35, "WET", "DEGRADED", 1.0, None)
        results = {tuple(self.evaluator.extract_evidence(*args).items()) for _ in range(1000)}
        self.assertEqual(len(results), 1)

    def test_wheel_slip_zero_when_not_braking_or_stationary(self):
        self.assertEqual(self.evaluator.compute_wheel_slip(0.3, 0.25, "ICY", "CRITICAL"), 0.0)  # accelerating
        self.assertEqual(self.evaluator.compute_wheel_slip(0.0, -1.0, "ICY", "CRITICAL"), 0.0)  # stationary

    def test_wheel_slip_increases_with_worse_traction_and_harder_braking(self):
        nominal_dry = self.evaluator.compute_wheel_slip(0.3, -0.5, "DRY", "NOMINAL")
        critical_icy = self.evaluator.compute_wheel_slip(0.3, -0.5, "ICY", "CRITICAL")
        self.assertLess(nominal_dry, critical_icy)

        light_brake = self.evaluator.compute_wheel_slip(0.3, -0.1, "DRY", "NOMINAL")
        hard_brake = self.evaluator.compute_wheel_slip(0.3, -1.0, "DRY", "NOMINAL")
        self.assertLess(light_brake, hard_brake)


class NominalCruiseTests(unittest.TestCase):
    def test_reaches_target_velocity_with_zero_false_emergency_stops(self):
        # Evidence is now a fully deterministic function of physical state
        # (no CPT sampling, no noise), so a genuinely obstacle-free run
        # produces zero false emergency stops as a hard guarantee, not an
        # empirical/seed-dependent property.
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=None, max_duration_s=5.0)
        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)

        self.assertFalse(result.collision)
        self.assertEqual(result.false_emergency_stop_count, 0)
        self.assertEqual(result.final_v, profile.v_cruise)
        self.assertTrue(all(a == "ACCEL" for a in result.action_history))

    def test_deterministic_across_repeated_runs(self):
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        profile = TrajectoryProfile(obstacle_appears_at_s=None, max_duration_s=3.0)
        results = []
        for _ in range(3):
            evaluator = TrajectoryEvaluator(_ENGINE, scenario)
            results.append(evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile))
        self.assertTrue(all(r.final_x == results[0].final_x for r in results))
        self.assertTrue(all(r.action_history == results[0].action_history for r in results))


class PopUpObstacleTests(unittest.TestCase):
    def test_pop_up_obstacle_triggers_emergency_brake_with_positive_clearance(self):
        base = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        scenario = dataclasses.replace(base, obstacle_position=0.6, braking_deceleration=0.8)
        profile = TrajectoryProfile(obstacle_appears_at_s=1.0, max_duration_s=10.0, evidence_period_s=0.1)
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)

        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)

        self.assertFalse(result.collision)
        self.assertTrue(result.emergency_brake_engaged)
        self.assertGreater(result.clearance_m, 0.0)
        self.assertGreaterEqual(result.clearance_m, profile.clearance_margin_m)
        self.assertEqual(result.final_v, 0.0)

    def test_achieved_stopping_position_matches_independent_analytic_plant_oracle(self):
        # With exact partial-step physics, this now agrees with
        # sim.plant.analytic_plant's independent closed-form solution to
        # floating-point precision, not just approximately.
        base = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        scenario = dataclasses.replace(base, obstacle_position=0.6, braking_deceleration=0.8)
        profile = TrajectoryProfile(obstacle_appears_at_s=1.0, max_duration_s=10.0, evidence_period_s=0.1)
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)

        brake_record = next(r for r in result.records if r.action == "EMERGENCY_BRAKE")
        pre_brake_index = result.records.index(brake_record) - 1
        x0 = result.records[pre_brake_index].x if pre_brake_index >= 0 else 0.0
        v0 = result.records[pre_brake_index].v if pre_brake_index >= 0 else 0.0

        state = PlantState(position=x0, velocity=v0, acceleration=0.0)
        params = PlantParameters(actuator_delay=0.0, braking_deceleration=scenario.braking_deceleration)
        predicted_stop_x = stopping_position(state, params, command_time=0.0)

        self.assertAlmostEqual(result.final_x, predicted_stop_x, delta=1e-9)


class MovingBlackoutTests(unittest.TestCase):
    def test_moving_cart_survives_a_500ms_dropout_beginning_mid_flight(self):
        # v0 = 0.40 m/s already moving toward an obstacle at 0.80 m; a
        # 500 ms dropout begins at t=0.20s (while moving, not at episode
        # start). Must not collide, and must not do anything erratic across
        # the blackout interval.
        base = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        scenario = dataclasses.replace(base, obstacle_position=0.80, dropout_start_s=0.20, dropout_duration_s=0.50)
        profile = TrajectoryProfile(obstacle_appears_at_s=0.0, max_duration_s=10.0, evidence_period_s=0.01)
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)

        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.40, acceleration=0.0), profile)

        self.assertFalse(result.collision)
        self.assertGreater(result.clearance_m, 0.0)

        # Each record's `t` is the *post-tick* time, but the evidence
        # attached to it was gathered one dt earlier (perception happens at
        # the start of a tick, physics advances it to t_next) -- so the
        # window of records whose evidence was actually gathered inside
        # [dropout_start_s, dropout_end] is shifted forward by one dt.
        dropout_end = scenario.dropout_start_s + scenario.dropout_duration_s
        window_lo = scenario.dropout_start_s + profile.dt
        window_hi = dropout_end + profile.dt
        blackout_evidence = [r.evidence for r in result.records if window_lo <= r.t <= window_hi]
        self.assertTrue(blackout_evidence, "test setup error: no records fell inside the blackout window")
        self.assertTrue(all(e["LidarObs"] is None and e["TofObs"] is None for e in blackout_evidence))

        # no erratic thrash: at most one action transition inside the blackout
        blackout_actions = [r.action for r in result.records if window_lo <= r.t <= window_hi]
        transitions = sum(1 for p, c in zip(blackout_actions, blackout_actions[1:]) if p != c)
        self.assertLessEqual(transitions, 1)

    def test_dropout_does_not_prevent_eventual_safe_stop_before_obstacle(self):
        base = generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0)
        scenario = dataclasses.replace(base, obstacle_position=0.80, dropout_start_s=0.20, dropout_duration_s=0.50)
        profile = TrajectoryProfile(obstacle_appears_at_s=0.0, max_duration_s=10.0, evidence_period_s=0.01)
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)

        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.40, acceleration=0.0), profile)

        self.assertTrue(result.emergency_brake_engaged)
        self.assertEqual(result.final_v, 0.0)


class Shift6ConstrainedClearanceTests(unittest.TestCase):
    """
    Scope: SHIFT_6 degraded-brake mitigation with fully operational wheel
    telemetry (dropout_duration_s=0.0 -- these scenarios test whether
    degraded braking capability alone still gets a meaningfully mitigated
    outcome). Sensor-blackout qualification (whether a telemetry outage
    itself is handled safely) is intentionally out of scope here; that is
    MovingBlackoutTests' job, so it isn't duplicated/conflated with this
    class's degraded-brake-specific assertions.
    """

    def test_early_emergency_brake_deadline_and_impact_energy_reduction(self):
        # v0 = 0.35 m/s, SHIFT_6 degraded brakes, and the obstacle placed
        # exactly at the detection range (0.30 m) so it is detectable from
        # the very first perception update -- a maximally "constrained
        # clearance" scenario for these degraded brakes (stopping distance
        # from cruise speed alone exceeds what 0.30 m of warning provides,
        # so this scenario does still end in contact even when mitigated;
        # what mitigation buys is a drastically reduced contact velocity).
        shift6 = generate_instance(DEFAULT_MASTER_SEED, StratumType.SHIFT_6_SIMULTANEOUS_FAULTS, 0)
        self.assertLess(shift6.braking_deceleration, 0.2)  # confirms genuinely degraded brakes
        scenario = dataclasses.replace(shift6, obstacle_position=0.30, dropout_duration_s=0.0)
        profile = TrajectoryProfile(obstacle_appears_at_s=0.0, max_duration_s=10.0, evidence_period_s=0.01)
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)

        v0 = 0.35
        result = evaluator.run_episode(PlantState(position=0.0, velocity=v0, acceleration=0.0), profile)

        # The obstacle is detectable from t=0 (d_clear = 0.30 - 0.0 <= 0.300),
        # so the reaction deadline is measured from episode start.
        self.assertTrue(result.emergency_brake_engaged)
        self.assertIsNotNone(result.first_emergency_brake_t)
        self.assertLessEqual(result.first_emergency_brake_t, 0.020)  # <= 20 ms

        # Terminal impact energy, computed strictly from contact_velocity/
        # impact_energy (the exact intra-step contact kinematics), never
        # from a post-braking clamped final_v -- on both the mitigated side
        # and the unmitigated baseline, so the comparison is apples to
        # apples through the same collision-resolution code path.
        self.assertTrue(result.collision)
        self.assertIsNotNone(result.contact_velocity)
        self.assertIsNotNone(result.impact_energy)

        unmitigated = _unmitigated_impact_result(evaluator, v0, profile)
        self.assertTrue(unmitigated.collision)
        self.assertIsNotNone(unmitigated.contact_velocity)
        self.assertGreater(unmitigated.contact_velocity, 0.0)

        reduction = 1.0 - (result.impact_energy / unmitigated.impact_energy)
        self.assertGreaterEqual(
            reduction,
            0.75,
            f"KE reduction only {reduction:.1%} "
            f"(mitigated contact_velocity={result.contact_velocity}, "
            f"unmitigated contact_velocity={unmitigated.contact_velocity})",
        )

    def test_combined_degraded_traction_and_dropout_still_avoids_collision(self):
        # With SHIFT_6's own built-in dropout window active (dropout_start=0,
        # duration=0.5s) and the obstacle close enough that the cart is
        # detected well before reaching v_cruise (an obstacle placed far
        # enough away that the cart reaches full cruise speed first would
        # make collision physically inevitable for this stratum's degraded
        # braking regardless of reaction time: stopping from v_cruise=0.50
        # needs ~0.706 m, but the fixed 0.300 m detection range can never
        # provide that much warning -- that is a real property of the exact
        # specified constants, not something mitigation can overcome), the
        # system must still engage emergency braking and avoid collision.
        shift6 = generate_instance(DEFAULT_MASTER_SEED, StratumType.SHIFT_6_SIMULTANEOUS_FAULTS, 0)
        scenario = dataclasses.replace(shift6, obstacle_position=0.5)
        profile = TrajectoryProfile(obstacle_appears_at_s=0.0, max_duration_s=15.0, evidence_period_s=0.01)
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)

        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)
        self.assertFalse(result.collision)
        self.assertGreater(result.clearance_m, 0.0)
        self.assertTrue(result.emergency_brake_engaged)


class FallbackFailSafeTests(unittest.TestCase):
    def test_forced_contradiction_triggers_fallback_and_a_fail_safe_stop(self):
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=None, max_duration_s=10.0, evidence_period_s=0.1)

        original_lidar_cpt = _ENGINE._sensor_cpt["LidarObs"]
        try:
            # Force P(LidarObs=DETECTED | TrueObstacle=*) = 0.0 for both
            # TrueObstacle states, then feed exactly that "impossible"
            # evidence throughout via evidence_override -- corrupted
            # trajectory telemetry the model itself considers contradictory
            # every single step.
            _ENGINE._sensor_cpt["LidarObs"] = [0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
            result = evaluator.run_episode(
                PlantState(position=0.0, velocity=0.3, acceleration=0.0),
                profile,
                evidence_override=lambda t: {"LidarObs": 1, "TofObs": 0, "WheelSlipObs": 0},
            )
        finally:
            _ENGINE._sensor_cpt["LidarObs"] = original_lidar_cpt

        self.assertFalse(result.collision)
        self.assertEqual(result.fallback_event_count, len(result.action_history))
        self.assertTrue(all(a == "EMERGENCY_BRAKE" for a in result.action_history))
        self.assertEqual(result.final_v, 0.0)
        self.assertTrue(result.settled)

    def test_ordinary_episode_never_triggers_fallback(self):
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=None, max_duration_s=3.0)
        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)
        self.assertEqual(result.fallback_event_count, 0)


class ControlLoopMechanicsTests(unittest.TestCase):
    def test_run_episode_returns_a_structured_episode_summary(self):
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=None, max_duration_s=1.0, evidence_period_s=0.5)
        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)

        self.assertIsInstance(result.final_x, float)
        self.assertIsInstance(result.final_v, float)
        self.assertIsInstance(result.collision, bool)
        self.assertIsInstance(result.action_history, list)
        self.assertGreater(len(result.records), 0)
        self.assertTrue(all(math.isfinite(r.x) and math.isfinite(r.v) for r in result.records))

    def test_zero_order_hold_keeps_action_constant_between_evidence_updates(self):
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=None, dt=0.01, evidence_period_s=0.5, max_duration_s=0.5)
        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)

        # 0.5s / 0.01s dt = 50 physics ticks, but only 1 evidence update
        # (at t=0) within [0, 0.5); the action must be identical across all.
        self.assertEqual(len(result.action_history), 1)
        self.assertEqual(len(set(r.action for r in result.records)), 1)

    def test_evidence_period_none_defaults_to_dt(self):
        scenario = _far_obstacle(generate_instance(DEFAULT_MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0))
        evaluator = TrajectoryEvaluator(_ENGINE, scenario)
        profile = TrajectoryProfile(obstacle_appears_at_s=None, dt=0.1, evidence_period_s=None, max_duration_s=0.5)
        result = evaluator.run_episode(PlantState(position=0.0, velocity=0.0, acceleration=0.0), profile)
        self.assertEqual(len(result.action_history), 5)


class TrueDecelIndexTests(unittest.TestCase):
    def test_nominal_band(self):
        self.assertEqual(_true_decel_index(1.0), 0)
        self.assertEqual(_true_decel_index(0.4), 0)

    def test_degraded_band(self):
        self.assertEqual(_true_decel_index(0.39999), 1)
        self.assertEqual(_true_decel_index(0.2), 1)

    def test_critical_band(self):
        self.assertEqual(_true_decel_index(0.19999), 2)
        self.assertEqual(_true_decel_index(0.05), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
