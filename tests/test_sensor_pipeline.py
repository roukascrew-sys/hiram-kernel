"""
Synthetic dual-sensor pipeline -- verification suite (Task 1.3).

Covers: empirical noise-variance conformance (ToF, ultrasonic, encoder) over
10,000 samples, exact boolean-evidence boundary triggers at the specified
thresholds (including the brief's explicit 0.299/0.301 m and 0.079/0.081 m
probe points), non-negative clamping, per-pipeline stream determinism and
independence, mutual exclusivity of the discretized evidence bands, and the
from_scenario() convenience constructor's seed derivation.

Run directly:          python tests/test_sensor_pipeline.py
Verification command:  python -m unittest tests/test_sensor_pipeline.py -v
"""

import hashlib
import pathlib
import random
import statistics
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sim.sensors import (  # noqa: E402
    D_CRITICAL_THRESHOLD_M,
    D_MARGINAL_THRESHOLD_M,
    ENCODER_NOISE_SIGMA,
    SENSOR_DISAGREE_THRESHOLD_M,
    TELEMETRY_STALE_THRESHOLD_S,
    TOF_NOISE_SIGMA_BASE,
    ULTRASONIC_NOISE_SIGMA_BASE,
    V_HIGH_THRESHOLD_MPS,
    V_MED_THRESHOLD_MPS,
    DiscreteEvidence,
    SensorPipeline,
    SensorReading,
    derive_sensor_seed,
)

N_SAMPLES = 10_000
VARIANCE_TOLERANCE = 0.05  # 5%

# Ground-truth values far enough from 0 that max(0.0, ...) clamping never
# fires across N_SAMPLES draws even at the widest sigma tested here (so the
# empirical std isn't truncated/biased by clamping).
D_TRUE_FOR_VARIANCE = 5.0
V_TRUE_FOR_VARIANCE = 1.0


class NoiseVarianceTests(unittest.TestCase):
    def _empirical_std(self, values):
        return statistics.pstdev(values)

    def test_tof_noise_std_matches_expected_within_5_percent(self):
        pipeline = SensorPipeline(seed=1, sensor_bias=0.0, sensor_noise_scale=1.0)
        residuals = [
            pipeline.sample(D_TRUE_FOR_VARIANCE, V_TRUE_FOR_VARIANCE).d_tof - D_TRUE_FOR_VARIANCE
            for _ in range(N_SAMPLES)
        ]
        empirical = self._empirical_std(residuals)
        expected = TOF_NOISE_SIGMA_BASE
        self.assertLess(abs(empirical - expected) / expected, VARIANCE_TOLERANCE, (empirical, expected))

    def test_ultrasonic_noise_std_matches_expected_within_5_percent(self):
        pipeline = SensorPipeline(seed=2, sensor_bias=0.0, sensor_noise_scale=1.0)
        residuals = [
            pipeline.sample(D_TRUE_FOR_VARIANCE, V_TRUE_FOR_VARIANCE).d_ultra - D_TRUE_FOR_VARIANCE
            for _ in range(N_SAMPLES)
        ]
        empirical = self._empirical_std(residuals)
        expected = ULTRASONIC_NOISE_SIGMA_BASE
        self.assertLess(abs(empirical - expected) / expected, VARIANCE_TOLERANCE, (empirical, expected))

    def test_encoder_noise_std_matches_expected_within_5_percent(self):
        pipeline = SensorPipeline(seed=3, sensor_bias=0.0, sensor_noise_scale=1.0)
        residuals = [
            pipeline.sample(D_TRUE_FOR_VARIANCE, V_TRUE_FOR_VARIANCE).v_meas - V_TRUE_FOR_VARIANCE
            for _ in range(N_SAMPLES)
        ]
        empirical = self._empirical_std(residuals)
        expected = ENCODER_NOISE_SIGMA
        self.assertLess(abs(empirical - expected) / expected, VARIANCE_TOLERANCE, (empirical, expected))

    def test_noise_scale_multiplies_range_sensor_sigma_but_not_encoder(self):
        scale = 2.5
        pipeline = SensorPipeline(seed=4, sensor_bias=0.0, sensor_noise_scale=scale)
        tof_residuals = []
        ultra_residuals = []
        encoder_residuals = []
        for _ in range(N_SAMPLES):
            r = pipeline.sample(D_TRUE_FOR_VARIANCE, V_TRUE_FOR_VARIANCE)
            tof_residuals.append(r.d_tof - D_TRUE_FOR_VARIANCE)
            ultra_residuals.append(r.d_ultra - D_TRUE_FOR_VARIANCE)
            encoder_residuals.append(r.v_meas - V_TRUE_FOR_VARIANCE)

        self.assertLess(
            abs(self._empirical_std(tof_residuals) - TOF_NOISE_SIGMA_BASE * scale) / (TOF_NOISE_SIGMA_BASE * scale),
            VARIANCE_TOLERANCE,
        )
        self.assertLess(
            abs(self._empirical_std(ultra_residuals) - ULTRASONIC_NOISE_SIGMA_BASE * scale)
            / (ULTRASONIC_NOISE_SIGMA_BASE * scale),
            VARIANCE_TOLERANCE,
        )
        # encoder sigma is fixed at 0.01 regardless of sensor_noise_scale
        self.assertLess(
            abs(self._empirical_std(encoder_residuals) - ENCODER_NOISE_SIGMA) / ENCODER_NOISE_SIGMA,
            VARIANCE_TOLERANCE,
        )

    def test_sensor_bias_shifts_mean_not_variance(self):
        bias = 0.15
        pipeline = SensorPipeline(seed=5, sensor_bias=bias, sensor_noise_scale=1.0)
        tof_values = [pipeline.sample(D_TRUE_FOR_VARIANCE, V_TRUE_FOR_VARIANCE).d_tof for _ in range(N_SAMPLES)]
        mean = statistics.fmean(tof_values)
        self.assertLess(abs(mean - (D_TRUE_FOR_VARIANCE + bias)), 0.01)  # mean within ~1cm of true+bias
        std = self._empirical_std([v - mean for v in tof_values])
        self.assertLess(abs(std - TOF_NOISE_SIGMA_BASE) / TOF_NOISE_SIGMA_BASE, VARIANCE_TOLERANCE)


class BoundaryDiscretizationTests(unittest.TestCase):
    @staticmethod
    def _reading(d_tof=1.0, d_ultra=1.0, v_meas=0.0, delta_t_sample=0.0):
        return SensorReading(d_tof=d_tof, d_ultra=d_ultra, v_meas=v_meas, delta_t_sample=delta_t_sample)

    def test_d_critical_boundary_0_299_vs_0_301(self):
        below = self._reading(d_tof=0.299, d_ultra=0.299)
        above = self._reading(d_tof=0.301, d_ultra=0.301)
        self.assertTrue(SensorPipeline.discretize(below).d_critical)
        self.assertFalse(SensorPipeline.discretize(below).d_marginal)
        self.assertFalse(SensorPipeline.discretize(above).d_critical)
        self.assertTrue(SensorPipeline.discretize(above).d_marginal)

    def test_d_critical_exact_threshold_is_not_critical(self):
        # spec is min(...) < 0.30 (strict); exactly 0.30 must be marginal, not critical
        at_threshold = self._reading(d_tof=D_CRITICAL_THRESHOLD_M, d_ultra=D_CRITICAL_THRESHOLD_M)
        ev = SensorPipeline.discretize(at_threshold)
        self.assertFalse(ev.d_critical)
        self.assertTrue(ev.d_marginal)

    def test_d_marginal_upper_boundary_0_649_vs_0_651(self):
        below = self._reading(d_tof=0.649, d_ultra=0.649)
        above = self._reading(d_tof=0.651, d_ultra=0.651)
        self.assertTrue(SensorPipeline.discretize(below).d_marginal)
        self.assertFalse(SensorPipeline.discretize(above).d_marginal)
        self.assertFalse(SensorPipeline.discretize(above).d_critical)

    def test_v_high_boundary_0_349_vs_0_351(self):
        below = self._reading(v_meas=0.349)
        above = self._reading(v_meas=0.351)
        self.assertFalse(SensorPipeline.discretize(below).v_high)
        self.assertTrue(SensorPipeline.discretize(below).v_med)
        self.assertTrue(SensorPipeline.discretize(above).v_high)
        self.assertFalse(SensorPipeline.discretize(above).v_med)

    def test_v_high_exact_threshold_is_high_not_med(self):
        # spec is v_meas >= 0.35 (inclusive); exactly 0.35 must be v_high
        at_threshold = self._reading(v_meas=V_HIGH_THRESHOLD_MPS)
        ev = SensorPipeline.discretize(at_threshold)
        self.assertTrue(ev.v_high)
        self.assertFalse(ev.v_med)

    def test_v_med_lower_boundary_0_149_vs_0_151(self):
        below = self._reading(v_meas=0.149)
        above = self._reading(v_meas=0.151)
        self.assertFalse(SensorPipeline.discretize(below).v_med)
        self.assertFalse(SensorPipeline.discretize(below).v_high)
        self.assertTrue(SensorPipeline.discretize(above).v_med)

    def test_sensor_disagree_boundary_0_079_vs_0_081(self):
        agree_ish = self._reading(d_tof=1.000, d_ultra=1.079)
        disagree = self._reading(d_tof=1.000, d_ultra=1.081)
        self.assertFalse(SensorPipeline.discretize(agree_ish).sensor_disagree)
        self.assertTrue(SensorPipeline.discretize(disagree).sensor_disagree)

    def test_sensor_disagree_exact_threshold_is_not_disagree(self):
        # spec is |d_tof - d_ultra| > 0.08 (strict); exactly 0.08 is not a
        # disagreement. Built from 0.0 and the threshold constant directly
        # (not `1.0 + SENSOR_DISAGREE_THRESHOLD_M`) so the subtraction
        # inside discretize() is exact -- 0.08 has no exact float64
        # representation, and 1.0 + 0.08 - 1.0 rounds to a hair above 0.08.
        at_threshold = self._reading(d_tof=0.0, d_ultra=SENSOR_DISAGREE_THRESHOLD_M)
        self.assertFalse(SensorPipeline.discretize(at_threshold).sensor_disagree)

    def test_sensor_disagree_is_symmetric(self):
        a = self._reading(d_tof=1.000, d_ultra=1.10)
        b = self._reading(d_tof=1.10, d_ultra=1.000)
        self.assertTrue(SensorPipeline.discretize(a).sensor_disagree)
        self.assertEqual(SensorPipeline.discretize(a).sensor_disagree, SensorPipeline.discretize(b).sensor_disagree)

    def test_telemetry_stale_boundary_0_049_vs_0_051(self):
        fresh = self._reading(delta_t_sample=0.049)
        stale = self._reading(delta_t_sample=0.051)
        self.assertFalse(SensorPipeline.discretize(fresh).telemetry_stale)
        self.assertTrue(SensorPipeline.discretize(stale).telemetry_stale)

    def test_telemetry_stale_exact_threshold_is_not_stale(self):
        # spec is delta_t > 0.05 (strict); exactly 50ms is not stale
        at_threshold = self._reading(delta_t_sample=TELEMETRY_STALE_THRESHOLD_S)
        self.assertFalse(SensorPipeline.discretize(at_threshold).telemetry_stale)

    def test_d_critical_uses_min_of_both_sensors(self):
        # one sensor critical, the other not -> still critical (min semantics)
        mixed = self._reading(d_tof=0.20, d_ultra=0.90)
        self.assertTrue(SensorPipeline.discretize(mixed).d_critical)


class ClampingTests(unittest.TestCase):
    def test_range_readings_never_negative_under_strong_negative_bias(self):
        pipeline = SensorPipeline(seed=11, sensor_bias=-100.0, sensor_noise_scale=1.0)
        for _ in range(2000):
            r = pipeline.sample(d_true=0.0, v_true=0.0)
            self.assertGreaterEqual(r.d_tof, 0.0)
            self.assertGreaterEqual(r.d_ultra, 0.0)

    def test_velocity_reading_never_negative_near_zero_true_velocity(self):
        pipeline = SensorPipeline(seed=12, sensor_bias=0.0, sensor_noise_scale=1.0)
        for _ in range(2000):
            r = pipeline.sample(d_true=5.0, v_true=0.0)
            self.assertGreaterEqual(r.v_meas, 0.0)

    def test_clamping_actually_engages_for_at_least_some_draws(self):
        # sanity: the clamp tests above aren't vacuous -- with v_true=0.0 and
        # sigma=0.01, roughly half of 2000 unclamped draws would be negative.
        pipeline = SensorPipeline(seed=13, sensor_bias=0.0, sensor_noise_scale=1.0)
        exactly_zero = sum(1 for _ in range(2000) if pipeline.sample(5.0, 0.0).v_meas == 0.0)
        self.assertGreater(exactly_zero, 500)


class DeterminismAndStreamTests(unittest.TestCase):
    def test_same_seed_reproduces_identical_reading_sequence(self):
        pipeline_a = SensorPipeline(seed=77, sensor_bias=0.01, sensor_noise_scale=1.2)
        pipeline_b = SensorPipeline(seed=77, sensor_bias=0.01, sensor_noise_scale=1.2)
        readings_a = [pipeline_a.sample(1.0, 0.2) for _ in range(10)]
        readings_b = [pipeline_b.sample(1.0, 0.2) for _ in range(10)]
        self.assertEqual(readings_a, readings_b)

    def test_consecutive_samples_on_one_pipeline_differ(self):
        pipeline = SensorPipeline(seed=88, sensor_bias=0.0, sensor_noise_scale=1.0)
        r1 = pipeline.sample(1.0, 0.2)
        r2 = pipeline.sample(1.0, 0.2)
        self.assertNotEqual(r1, r2)  # the stream actually advances

    def test_different_seed_diverges(self):
        r1 = SensorPipeline(seed=1, sensor_bias=0.0, sensor_noise_scale=1.0).sample(1.0, 0.2)
        r2 = SensorPipeline(seed=2, sensor_bias=0.0, sensor_noise_scale=1.0).sample(1.0, 0.2)
        self.assertNotEqual(r1, r2)

    def test_derive_sensor_seed_is_deterministic_and_distinct_from_episode_seed(self):
        episode_seed = 123456789
        s1 = derive_sensor_seed(episode_seed)
        s2 = derive_sensor_seed(episode_seed)
        self.assertEqual(s1, s2)
        self.assertNotEqual(s1, episode_seed)
        expected = int(hashlib.sha256(f"{episode_seed}:sensors".encode()).hexdigest()[:8], 16)
        self.assertEqual(s1, expected)


class FromScenarioTests(unittest.TestCase):
    class _FakeScenario:
        def __init__(self, seed, sensor_bias, sensor_noise_scale):
            self.seed = seed
            self.sensor_bias = sensor_bias
            self.sensor_noise_scale = sensor_noise_scale

    def test_from_scenario_uses_derived_seed_and_episode_sensor_params(self):
        scenario = self._FakeScenario(seed=42, sensor_bias=0.03, sensor_noise_scale=1.4)
        pipeline = SensorPipeline.from_scenario(scenario)
        self.assertEqual(pipeline.sensor_bias, 0.03)
        self.assertEqual(pipeline.sensor_noise_scale, 1.4)

        expected_seed_pipeline = SensorPipeline(
            seed=derive_sensor_seed(42), sensor_bias=0.03, sensor_noise_scale=1.4
        )
        self.assertEqual(pipeline.sample(1.0, 0.2), expected_seed_pipeline.sample(1.0, 0.2))

    def test_from_scenario_explicit_seed_override(self):
        scenario = self._FakeScenario(seed=42, sensor_bias=0.0, sensor_noise_scale=1.0)
        pipeline = SensorPipeline.from_scenario(scenario, seed=999)
        other = SensorPipeline(seed=999, sensor_bias=0.0, sensor_noise_scale=1.0)
        self.assertEqual(pipeline.sample(1.0, 0.2), other.sample(1.0, 0.2))

    def test_real_scenario_instance_integration(self):
        # integration smoke test against the actual generator, not the fake
        from sim.scenarios import StratumType, generate_episode

        instance = generate_episode(StratumType.NOMINAL_3_SHARED_BIAS, 0)
        pipeline = SensorPipeline.from_scenario(instance)
        reading, evidence = pipeline.read(d_true=instance.obstacle_position - instance.x0, v_true=instance.v0)
        self.assertIsInstance(reading, SensorReading)
        self.assertIsInstance(evidence, DiscreteEvidence)


class DiscretizationConsistencyTests(unittest.TestCase):
    def test_bands_are_mutually_exclusive_across_random_readings(self):
        rng = random.Random(2026)
        for _ in range(5000):
            reading = SensorReading(
                d_tof=rng.uniform(-0.1, 2.0),
                d_ultra=rng.uniform(-0.1, 2.0),
                v_meas=rng.uniform(-0.1, 1.0),
                delta_t_sample=rng.uniform(0.0, 0.2),
            )
            ev = SensorPipeline.discretize(reading)
            self.assertFalse(ev.d_critical and ev.d_marginal)
            self.assertFalse(ev.v_high and ev.v_med)

    def test_discretize_is_a_pure_function_of_the_reading(self):
        reading = SensorReading(d_tof=0.5, d_ultra=0.55, v_meas=0.2, delta_t_sample=0.01)
        self.assertEqual(SensorPipeline.discretize(reading), SensorPipeline.discretize(reading))


if __name__ == "__main__":
    unittest.main(verbosity=2)
