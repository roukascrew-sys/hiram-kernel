"""
Dual reference plant oracle -- verification and contract regression suite.
"""

import math
import pathlib
import random
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sim.plant import PlantParameters, PlantState  # noqa: E402
from sim.plant import analytic_plant, numerical_plant  # noqa: E402
from sim.plant.analytic_plant import PlantOracleContractError, PlantOracleConvergenceError  # noqa: E402

TOLERANCE_M = 5.0e-5  # 0.05 mm terminal-position agreement
CONVERGENCE_TOLERANCE_M = 1.0e-5  # 10 micron step-halving convergence bound
SEED = 20260910

V0_RANGE = (0.0, 0.5)
X0_RANGE = (-1.0, 1.0)
A_CMD_RANGE = (-0.25, 0.25)
TAU_RANGE = (0.0, 0.10)
DECEL_RANGE = (0.2, 1.0)
OBSTACLE_RANGE = (0.5, 3.0)

V0_CORNERS = (0.0, 0.01, 0.5)
A_CMD_CORNERS = (-0.25, 0.0, 0.25)
TAU_CORNERS = (0.0, 0.01, 0.10)
B_CORNERS = (0.2, 0.5, 1.0)


def _random_profile(rng: random.Random):
    state = PlantState(
        position=rng.uniform(*X0_RANGE),
        velocity=rng.uniform(*V0_RANGE),
        acceleration=0.0,
    )
    params = PlantParameters(
        actuator_delay=rng.uniform(*TAU_RANGE),
        braking_deceleration=rng.uniform(*DECEL_RANGE),
        commanded_acceleration=rng.uniform(*A_CMD_RANGE),
        obstacle_position=rng.uniform(*OBSTACLE_RANGE),
    )
    return state, params


def is_clearance_violation(stop_pos: float, obstacle_pos: float, clearance_margin: float = 0.10) -> bool:
    """True if resting position leaves less than clearance_margin meters."""
    return stop_pos > (obstacle_pos - clearance_margin)


def is_physical_collision(stop_pos: float, obstacle_pos: float) -> bool:
    """True if cart physically makes contact with obstacle."""
    return stop_pos >= obstacle_pos


def _mutant_ignore_delay(v0, x0, a_cmd, tau, b):
    v_d, x_d = v0, x0
    if v_d <= 0.0:
        return x_d
    t_brake = v_d / b
    return x_d + v_d * t_brake - 0.5 * b * t_brake * t_brake


def _mutant_inverted_brake_sign(v0, x0, a_cmd, tau, b):
    v_d = v0 + a_cmd * tau
    x_d = x0 + v0 * tau + 0.5 * a_cmd * tau * tau
    if v_d <= 0.0:
        return x_d
    t_brake = v_d / b
    return x_d + v_d * t_brake + 0.5 * b * t_brake * t_brake


def _mutant_millisecond_scaling(v0, x0, a_cmd, tau, b):
    tau_bug = tau * 1000.0
    v_d = v0 + a_cmd * tau_bug
    x_d = x0 + v0 * tau_bug + 0.5 * a_cmd * tau_bug * tau_bug
    if v_d <= 0.0:
        return x_d
    t_brake = v_d / b
    return x_d + v_d * t_brake - 0.5 * b * t_brake * t_brake


def _mutant_omit_a_cmd(v0, x0, a_cmd, tau, b):
    v_d = v0
    x_d = x0 + v0 * tau
    if v_d <= 0.0:
        return x_d
    t_brake = v_d / b
    return x_d + v_d * t_brake - 0.5 * b * t_brake * t_brake


MUTANTS = {
    "ignore_delay": _mutant_ignore_delay,
    "inverted_brake_sign": _mutant_inverted_brake_sign,
    "millisecond_scaling": _mutant_millisecond_scaling,
    "omit_a_cmd": _mutant_omit_a_cmd,
}

MUTATION_PROFILES = [
    (0.2, 0.0, 0.1, 0.02, 0.3),
    (0.3, 0.0, -0.1, 0.05, 0.6),
    (0.4, 0.0, 0.2, 0.08, 0.9),
]


class PlantOracleTests(unittest.TestCase):

    def test_deterministic_corner_vectors(self):
        for v0 in V0_CORNERS:
            for a_cmd in A_CMD_CORNERS:
                for tau in TAU_CORNERS:
                    for b in B_CORNERS:
                        with self.subTest(v0=v0, a_cmd=a_cmd, tau=tau, b=b):
                            state = PlantState(position=0.0, velocity=v0, acceleration=0.0)
                            params = PlantParameters(
                                actuator_delay=tau, braking_deceleration=b, commanded_acceleration=a_cmd
                            )

                            analytic_final = analytic_plant.stopping_position(state, params)
                            rk4_trace = numerical_plant.simulate(state, params, dt=0.0001, margin=0.01)
                            rk4_final = rk4_trace[-1][1].position

                            diff = abs(analytic_final - rk4_final)
                            self.assertLessEqual(diff, TOLERANCE_M)
                            self.assertEqual(rk4_trace[-1][1].velocity, 0.0)
                            self.assertEqual(rk4_trace[-1][1].acceleration, 0.0)

    def test_randomized_sweep_10000_profiles(self):
        rng = random.Random(SEED)
        max_diff = 0.0

        for trial in range(10_000):
            state, params = _random_profile(rng)
            analytic_final_x = analytic_plant.stopping_position(state, params)
            rk4_trace = numerical_plant.simulate(state, params, dt=0.0001, margin=0.005)
            rk4_final_x = rk4_trace[-1][1].position

            diff = abs(analytic_final_x - rk4_final_x)
            max_diff = max(max_diff, diff)
            self.assertLessEqual(diff, TOLERANCE_M)

        print(f"\n  max terminal disagreement over 10,000 trials: {max_diff * 1000:.6f} mm")

    def test_rk4_step_halving_convergence(self):
        rng = random.Random(SEED + 1)
        max_diff = 0.0
        for trial in range(1000):
            state, params = _random_profile(rng)
            x_coarse = numerical_plant.simulate(state, params, dt=0.0002, margin=0.005)[-1][1].position
            x_fine = numerical_plant.simulate(state, params, dt=0.0001, margin=0.005)[-1][1].position

            diff = abs(x_coarse - x_fine)
            max_diff = max(max_diff, diff)
            self.assertLessEqual(diff, CONVERGENCE_TOLERANCE_M)
        print(f"\n  max step-halving disagreement over 1,000 trials: {max_diff * 1e6:.4f} um")

    def test_disentangled_collision_and_clearance_predicates(self):
        obstacle = 3.0
        margin = 0.10

        # Safe short of clearance line (e.g. 2.80 m)
        self.assertFalse(is_clearance_violation(2.80, obstacle, margin))
        self.assertFalse(is_physical_collision(2.80, obstacle))

        # Exactly at clearance line (2.90 m) -> compliant with margin, no collision
        self.assertFalse(is_clearance_violation(2.90, obstacle, margin))
        self.assertFalse(is_physical_collision(2.90, obstacle))

        # Clearance violation without physical contact (e.g. 2.95 m)
        self.assertTrue(is_clearance_violation(2.95, obstacle, margin))
        self.assertFalse(is_physical_collision(2.95, obstacle))

        # Direct boundary contact (3.00 m)
        self.assertTrue(is_clearance_violation(3.00, obstacle, margin))
        self.assertTrue(is_physical_collision(3.00, obstacle))

        # Obstacle penetration (3.05 m)
        self.assertTrue(is_clearance_violation(3.05, obstacle, margin))
        self.assertTrue(is_physical_collision(3.05, obstacle))

    def test_nonzero_command_time_cross_oracle_agreement(self):
        rng = random.Random(SEED + 4)
        for _ in range(500):
            state, params = _random_profile(rng)
            cmd_time = rng.uniform(0.01, 0.5)

            analytic_final = analytic_plant.stopping_position(state, params, command_time=cmd_time)
            rk4_trace = numerical_plant.simulate(state, params, dt=0.0001, command_time=cmd_time, margin=0.005)
            rk4_final = rk4_trace[-1][1].position

            self.assertLessEqual(abs(analytic_final - rk4_final), TOLERANCE_M)

    def test_right_continuous_boundary_acceleration_agreement(self):
        state = PlantState(position=0.0, velocity=0.4, acceleration=0.0)
        params = PlantParameters(actuator_delay=0.05, braking_deceleration=0.6, commanded_acceleration=0.2)
        cmd_time = 0.10
        delay_end = cmd_time + params.actuator_delay

        # Exactly at command_time: acceleration switches to a_cmd
        s_cmd = analytic_plant.state_at(cmd_time, state, params, command_time=cmd_time)
        self.assertEqual(s_cmd.acceleration, params.commanded_acceleration)

        # Exactly at delay_end: acceleration switches to -b
        s_brake = analytic_plant.state_at(delay_end, state, params, command_time=cmd_time)
        self.assertEqual(s_brake.acceleration, -params.braking_deceleration)

    def test_contract_validation_rejects_malformed_inputs(self):
        valid_state = PlantState(position=0.0, velocity=0.3, acceleration=0.0)
        valid_params = PlantParameters(actuator_delay=0.05, braking_deceleration=0.5)

        # Negative initial velocity
        bad_v = PlantState(position=0.0, velocity=-0.1, acceleration=0.0)
        with self.assertRaises(PlantOracleContractError):
            analytic_plant.stopping_position(bad_v, valid_params)
        with self.assertRaises(PlantOracleContractError):
            numerical_plant.simulate(bad_v, valid_params)

        # Non-positive braking deceleration
        bad_b = PlantParameters(actuator_delay=0.05, braking_deceleration=0.0)
        with self.assertRaises(PlantOracleContractError):
            analytic_plant.stopping_position(valid_state, bad_b)
        with self.assertRaises(PlantOracleContractError):
            numerical_plant.simulate(valid_state, bad_b)

        # Negative delay
        bad_tau = PlantParameters(actuator_delay=-0.01, braking_deceleration=0.5)
        with self.assertRaises(PlantOracleContractError):
            analytic_plant.stopping_position(valid_state, bad_tau)
        with self.assertRaises(PlantOracleContractError):
            numerical_plant.simulate(valid_state, bad_tau)

        # Non-finite values
        bad_nan = PlantParameters(actuator_delay=float("nan"), braking_deceleration=0.5)
        with self.assertRaises(PlantOracleContractError):
            analytic_plant.stopping_position(valid_state, bad_nan)
        with self.assertRaises(PlantOracleContractError):
            numerical_plant.simulate(valid_state, bad_nan)

    def test_mutation_suite_catches_injected_defects(self):
        caught = {name: False for name in MUTANTS}
        for v0, x0, a_cmd, tau, b in MUTATION_PROFILES:
            state = PlantState(position=x0, velocity=v0, acceleration=0.0)
            params = PlantParameters(actuator_delay=tau, braking_deceleration=b, commanded_acceleration=a_cmd)
            ground_truth = numerical_plant.simulate(state, params, dt=0.0001, margin=0.01)[-1][1].position

            for name, fn in MUTANTS.items():
                mutant_result = fn(v0, x0, a_cmd, tau, b)
                if abs(mutant_result - ground_truth) > TOLERANCE_M:
                    caught[name] = True

        for name, was_caught in caught.items():
            self.assertTrue(was_caught, f"mutation suite FAILED to catch: {name}")


if __name__ == "__main__":
    unittest.main(verbosity=2)