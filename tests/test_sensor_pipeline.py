"""
Unit tests for sensor pipeline: cumulative ticks, zero stationary drift,
unbiased low-speed tracking, timestamp-zero interval integrity, and tri-state dropout evidence.
"""

import math
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios.generator import generate_instance
from sim.scenarios.strata import StratumType
from sim.sensors.pipeline import SensorPipeline

MASTER_SEED = "HIRAM-BENCHMARK-2026-TEST"


class TestSensorPipeline(unittest.TestCase):

    def test_stationary_cart_has_zero_encoder_drift(self):
        """Auditor check: stationary cart over 100 seconds must accumulate exactly zero displacement."""
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 101)
        pipe = SensorPipeline(MASTER_SEED, sc)

        for frame in range(10_000):
            obs = pipe.sample(t=frame * 0.01, current_x=0.0, current_v=0.0, dt_sample=0.01)

        self.assertEqual(pipe.cumulative_true_pos_m, 0.0)
        self.assertEqual(pipe.cumulative_ticks, 0)
        self.assertEqual(obs.v_measured, 0.0)

    def test_low_speed_unbiased_accumulation(self):
        """
        Auditor check: cart at 1 um/s over 100 seconds travels 0.0001 m.
        Must not suffer half-wave rectification drift (accumulating ~0.2 m).
        """
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 103)
        pipe = SensorPipeline(MASTER_SEED, sc)

        v_low = 1e-6  # 1 um/s
        total_time_s = 100.0
        dt = 0.01
        steps = int(total_time_s / dt)

        for frame in range(steps):
            pipe.sample(t=frame * dt, current_x=frame * dt * v_low, current_v=v_low, dt_sample=dt)

        # Expected true displacement is exactly 0.0001 m (0.1 mm)
        self.assertAlmostEqual(pipe.cumulative_true_pos_m, 0.0001, delta=1e-6)
        # 0.1 mm is less than TICK_PITCH (0.5 mm), so ticks must be 0
        self.assertEqual(pipe.cumulative_ticks, 0)

    def test_encoder_initialization_at_timestamp_zero(self):
        """Auditor check: samples at t=0 and t=0.1s must accumulate expected 0.025m travel at 0.25 m/s."""
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 102)
        pipe = SensorPipeline(MASTER_SEED, sc)

        pipe.sample(t=0.0, current_x=0.0, current_v=0.25, dt_sample=0.01)
        self.assertEqual(pipe.cumulative_true_pos_m, 0.0)

        obs_1 = pipe.sample(t=0.10, current_x=0.025, current_v=0.25, dt_sample=0.01)
        self.assertAlmostEqual(pipe.cumulative_true_pos_m, 0.025, delta=1e-6)
        self.assertGreaterEqual(pipe.cumulative_ticks, 48)
        self.assertLessEqual(pipe.cumulative_ticks, 52)
        self.assertTrue(obs_1.v_med)

    def test_v_med_is_empirically_reachable(self):
        sc = generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 5)
        pipe = SensorPipeline(MASTER_SEED, sc)

        v_med_hits = 0
        total_steps = 200
        for step in range(1, total_steps + 1):
            obs = pipe.sample(t=step * 0.01, current_x=0.0, current_v=0.25, dt_sample=0.01)
            if obs.v_med:
                v_med_hits += 1

        self.assertGreater(v_med_hits, 150)

    def test_dropout_tri_state_evidence_masking_and_recovery(self):
        sc = generate_instance(MASTER_SEED, StratumType.SHIFT_4_OBS_DROPOUT_500MS, 1)
        pipe = SensorPipeline(MASTER_SEED, sc)

        obs_drop = pipe.sample(t=0.25, current_x=0.0, current_v=0.2)
        self.assertIsNone(obs_drop.d_tof)
        self.assertIsNone(obs_drop.d_critical)
        self.assertIsNone(obs_drop.d_marginal)
        self.assertIsNone(obs_drop.sensor_disagree)
        self.assertTrue(obs_drop.telemetry_stale)

        obs_rec = pipe.sample(t=0.55, current_x=0.1, current_v=0.2)
        self.assertIsNotNone(obs_rec.d_tof)
        self.assertIsNotNone(obs_rec.d_critical)
        self.assertFalse(obs_rec.telemetry_stale)


if __name__ == "__main__":
    unittest.main(verbosity=2)