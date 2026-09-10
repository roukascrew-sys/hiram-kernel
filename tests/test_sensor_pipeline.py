"""
Unit tests for sensor pipeline: cumulative ticks, zero stationary drift,
unbiased tracking, alternating edge jitter, and closed-loop dropout execution.
"""

import math
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sim.runner import run_episode
from sim.scenarios.generator import generate_instance
from sim.scenarios.strata import StratumType
from sim.sensors.pipeline import SensorPipeline

MASTER_SEED = "HIRAM-BENCHMARK-2026-TEST"


class TestSensorPipeline(unittest.TestCase):

    def test_stationary_cart_has_zero_encoder_drift(self):
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 101)
        pipe = SensorPipeline(MASTER_SEED, sc)

        for frame in range(10_000):
            obs = pipe.sample(t=frame * 0.01, current_x=0.0, current_v=0.0, dt_sample=0.01)

        self.assertEqual(pipe.cumulative_ticks, 0)
        self.assertEqual(obs.v_measured, 0.0)

    def test_edge_jitter_chatter_does_not_rectify_forward_motion(self):
        """
        Auditor check: 1 nanometer actual travel fluctuating over an edge
        must NOT rectify into 0.025 m of phantom motion.
        """
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 104)
        pipe = SensorPipeline(MASTER_SEED, sc)

        # Position hovers at 0.0004999 m (just below 0.5 mm tick edge) with 1 nm fluctuation
        edge_x = 0.0004999
        for frame in range(1_000):
            x_hover = edge_x + (1e-9 if frame % 2 == 0 else -1e-9)
            obs = pipe.sample(t=frame * 0.01, current_x=x_hover, current_v=0.0, dt_sample=0.01)

        self.assertLessEqual(pipe.cumulative_ticks, 1)

    def test_deceleration_displacement_accuracy(self):
        """
        Auditor check: cart decelerating from 0.25 m/s to 0 over 1 second travels 0.125 m.
        Encoder must measure travel accurately based on plant position state.
        """
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 105)
        pipe = SensorPipeline(MASTER_SEED, sc)

        total_time = 1.0
        dt = 0.01
        steps = int(total_time / dt)
        v0 = 0.25
        a = -0.25

        for step in range(steps + 1):
            t = step * dt
            v = max(0.0, v0 + a * t)
            x = v0 * t + 0.5 * a * (t ** 2)
            pipe.sample(t=t, current_x=x, current_v=v, dt_sample=dt)

        # 0.125 m travel / 0.0005 m tick = exactly 250 ticks
        self.assertAlmostEqual(pipe.last_physical_x, 0.125, delta=1e-5)
        self.assertGreaterEqual(pipe.cumulative_ticks, 248)
        self.assertLessEqual(pipe.cumulative_ticks, 252)

    def test_cumulative_encoder_tick_accumulation(self):
        """Auditor check: persistent tick accumulation over 1 second at constant 0.10 m/s."""
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 3)
        pipe = SensorPipeline(MASTER_SEED, sc)

        for step in range(1, 101):
            t = step * 0.01
            x = 0.10 * t
            obs = pipe.sample(t=t, current_x=x, current_v=0.10, dt_sample=0.01)

        self.assertAlmostEqual(pipe.last_physical_x, 0.10, delta=1e-5)
        self.assertGreaterEqual(pipe.cumulative_ticks, 198)
        self.assertLessEqual(pipe.cumulative_ticks, 202)

    def test_v_med_is_empirically_reachable(self):
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 5)
        pipe = SensorPipeline(MASTER_SEED, sc)

        v_med_hits = 0
        total_steps = 200
        for step in range(1, total_steps + 1):
            x = step * 0.01 * 0.25
            obs = pipe.sample(t=step * 0.01, current_x=x, current_v=0.25, dt_sample=0.01)
            if obs.v_med:
                v_med_hits += 1

        self.assertGreater(v_med_hits, 150)

    def test_shift4_full_500ms_dropout_episode_execution(self):
        """Auditor check: full 500 ms dropout timeline without truncation."""
        sc = generate_instance(MASTER_SEED, StratumType.SHIFT_4_OBS_DROPOUT_500MS, 0)
        result = run_episode(sc, MASTER_SEED, dt=0.01)

        self.assertEqual(result.dropout_steps_executed, 50)
        self.assertGreaterEqual(result.total_duration_s, 0.50)

        for frame in result.evidence_frames[:50]:
            self.assertIsNone(frame.d_critical)
            self.assertIsNone(frame.d_marginal)
            self.assertIsNone(frame.sensor_disagree)
            self.assertTrue(frame.telemetry_stale)

        post_dropout_frame = result.evidence_frames[50]
        self.assertIsNotNone(post_dropout_frame.d_critical)
        self.assertFalse(post_dropout_frame.telemetry_stale)

    def test_shift6_simultaneous_fault_episode_handling(self):
        """Auditor check: SHIFT_6 combines degraded brakes and full 500 ms dropout."""
        sc = generate_instance(MASTER_SEED, StratumType.SHIFT_6_SIMULTANEOUS_FAULTS, 0)
        result = run_episode(sc, MASTER_SEED, dt=0.01)

        self.assertEqual(result.dropout_steps_executed, 50)
        self.assertLess(sc.braking_deceleration, 0.20)
        self.assertEqual(sc.sensor_noise_scale, 2.0)
        self.assertGreaterEqual(sc.sensor_bias, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)