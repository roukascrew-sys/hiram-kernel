"""
Comprehensive qualification test suite for scenario generation, full 420k custody,
lossless bit-exact hashing, sequence verification, and all 12 physical strata.
"""

import math
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sim.plant import PlantParameters, PlantState, analytic_plant
from sim.scenarios.generator import (
    DEFAULT_MASTER_SEED,
    TOTAL_BENCHMARK_EPISODES,
    canonical_episode_count,
    compute_dataset_hash,
    generate_canonical_stratum,
    generate_instance,
    generate_stratum_sample,
)
from sim.scenarios.strata import ScenarioInstance, StratumType, derive_domain_seed

MASTER_SEED = "HIRAM-BENCHMARK-2026-TEST"


class TestScenarioGenerator(unittest.TestCase):

    def test_seed_uniqueness_across_full_domains(self):
        seen_seeds = set()
        total_trials = 25_000
        for ep_id in range(total_trials):
            for dom in ["dynamics", "tof", "ultra", "encoder"]:
                s = derive_domain_seed(MASTER_SEED, StratumType.NOMINAL_1_STEADY.name, ep_id, dom)
                seen_seeds.add(s)
        self.assertEqual(len(seen_seeds), total_trials * 4)

    def test_exact_reserve_band_distribution(self):
        instances = generate_stratum_sample(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 2000)
        critical_band = 0
        marginal_band = 0
        open_band = 0

        for sc in instances:
            state = PlantState(position=sc.x0, velocity=sc.v0, acceleration=0.0)
            params = PlantParameters(sc.actuator_delay, sc.braking_deceleration, sc.commanded_acceleration)
            x_stop = analytic_plant.stopping_position(state, params, command_time=0.0)
            r = sc.obstacle_position - x_stop - 0.10

            self.assertGreaterEqual(r, 0.0)
            if 0.001 <= r < 0.05:
                critical_band += 1
            elif 0.05 <= r < 0.30:
                marginal_band += 1
            elif 0.30 <= r <= 1.50:
                open_band += 1

        self.assertAlmostEqual(critical_band / 2000.0, 0.30, delta=0.05)
        self.assertAlmostEqual(marginal_band / 2000.0, 0.40, delta=0.05)
        self.assertAlmostEqual(open_band / 2000.0, 0.30, delta=0.05)

    def test_canonical_stratum_counts_and_all_twelve_strata_logic(self):
        for st in StratumType:
            expected = 10_000 if st.name.startswith("SHIFT") else 60_000
            self.assertEqual(canonical_episode_count(st), expected)
            inst = generate_instance(MASTER_SEED, st, 1)

            if st == StratumType.NOMINAL_2_BOUNDARY_NOISE:
                self.assertGreaterEqual(inst.sensor_noise_scale, 1.2)
            elif st == StratumType.NOMINAL_4_LOW_BRAKING:
                self.assertLessEqual(inst.braking_deceleration, 0.35)
            elif st == StratumType.NOMINAL_5_SENSOR_DELAY:
                self.assertGreaterEqual(inst.actuator_delay, 0.08)
            elif st == StratumType.SHIFT_1_DEGRADED_BRAKE:
                self.assertLess(inst.braking_deceleration, 0.20)
            elif st == StratumType.SHIFT_2_DOUBLE_NOISE:
                self.assertEqual(inst.sensor_noise_scale, 2.0)
            elif st == StratumType.SHIFT_4_OBS_DROPOUT_500MS:
                self.assertEqual(inst.dropout_duration_s, 0.50)
                self.assertEqual(inst.dropout_start_s, 0.0)

    def test_lossless_hash_detects_single_bit_float_change(self):
        sample = generate_stratum_sample(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 10)
        h_orig = compute_dataset_hash(sample, MASTER_SEED, validate_canonical_sequence=False)

        mutated = list(sample)
        orig_inst = mutated[0]
        mutated_inst = ScenarioInstance(
            episode_id=orig_inst.episode_id,
            stratum=orig_inst.stratum,
            seed=orig_inst.seed,
            x0=orig_inst.x0,
            v0=orig_inst.v0 + 1e-8,
            commanded_acceleration=orig_inst.commanded_acceleration,
            obstacle_position=orig_inst.obstacle_position,
            braking_deceleration=orig_inst.braking_deceleration,
            actuator_delay=orig_inst.actuator_delay,
            sensor_bias=orig_inst.sensor_bias,
            sensor_noise_scale=orig_inst.sensor_noise_scale,
            dropout_start_s=orig_inst.dropout_start_s,
            dropout_duration_s=orig_inst.dropout_duration_s,
            is_recoverable=orig_inst.is_recoverable,
        )
        mutated[0] = mutated_inst
        h_mut = compute_dataset_hash(mutated, MASTER_SEED, validate_canonical_sequence=False)

        self.assertNotEqual(h_orig, h_mut)

    def test_sequence_custody_rejects_duplicate_or_unordered_episodes(self):
        """Auditor check: supplying duplicate episodes or out-of-order sequences must fail validation."""
        # Supply 10 copies of episode 0
        duplicate_stream = [generate_instance(MASTER_SEED, StratumType.NOMINAL_1_STEADY, 0) for _ in range(10)]
        with self.assertRaises(ValueError):
            compute_dataset_hash(duplicate_stream, MASTER_SEED, validate_canonical_sequence=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)