"""
Scenario generator & sealed seed custody -- verification suite (Task 1.2).

Covers: strata/episode-count contracts, the exact mandated seed-derivation
formula (and its known 32-bit collision behavior at full scale), per-episode
reproducibility and cross-stratum seed independence, the physical
recoverability guarantee for all 360,000 nominal-stratum episodes (verified
against the independent sim.plant.analytic_plant oracle), the corresponding
non-guarantee for the 60,000 shift-stratum episodes (with explicit
out-of-envelope checks for SHIFT_1 and SHIFT_4), the defense-in-depth
resample/fallback path that backs the guarantee, per-stratum
parameter-envelope conformance, the streaming generate_stratum_sample entry
point, and manifest-digest sealing.

Run directly:          python tests/test_scenario_generator.py
Verification command:  python -m unittest tests/test_scenario_generator.py -v
"""

import dataclasses
import hashlib
import pathlib
import random
import sys
import types
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import sim.scenarios.generator as generator  # noqa: E402
from sim.scenarios import (  # noqa: E402
    ALL_STRATA,
    EPISODES_PER_NOMINAL_STRATUM,
    EPISODES_PER_SHIFT_STRATUM,
    NOMINAL_STRATA,
    SHIFT_STRATA,
    TOTAL_EPISODES,
    TOTAL_NOMINAL_EPISODES,
    TOTAL_SHIFT_EPISODES,
    ScenarioInstance,
    StratumType,
    episodes_for,
    generate_episode,
    generate_stratum,
    generate_stratum_sample,
    iter_all,
    iter_stratum,
    manifest_digest,
)
from sim.scenarios.generator import CLEARANCE_MARGIN_M, OBSTACLE_NOMINAL, TAU_NOMINAL, V0_NOMINAL


class StrataContractTests(unittest.TestCase):
    def test_strata_counts_and_totals(self):
        self.assertEqual(len(NOMINAL_STRATA), 6)
        self.assertEqual(len(SHIFT_STRATA), 6)
        self.assertEqual(len(ALL_STRATA), 12)
        self.assertEqual(len(set(ALL_STRATA)), 12)  # all distinct

        for stratum in NOMINAL_STRATA:
            self.assertEqual(episodes_for(stratum), EPISODES_PER_NOMINAL_STRATUM)
        for stratum in SHIFT_STRATA:
            self.assertEqual(episodes_for(stratum), EPISODES_PER_SHIFT_STRATUM)

        self.assertEqual(EPISODES_PER_NOMINAL_STRATUM, 60_000)
        self.assertEqual(EPISODES_PER_SHIFT_STRATUM, 10_000)
        self.assertEqual(TOTAL_NOMINAL_EPISODES, 360_000)
        self.assertEqual(TOTAL_SHIFT_EPISODES, 60_000)
        self.assertEqual(TOTAL_EPISODES, 420_000)

    def test_scenario_instance_schema_matches_contract_exactly(self):
        fields = {f.name for f in dataclasses.fields(ScenarioInstance)}
        self.assertEqual(
            fields,
            {
                "episode_id",
                "stratum",
                "seed",
                "x0",
                "v0",
                "obstacle_position",
                "braking_deceleration",
                "actuator_delay",
                "sensor_bias",
                "sensor_noise_scale",
                "is_recoverable",
            },
        )

    def test_scenario_instance_is_frozen(self):
        inst = generate_episode(StratumType.NOMINAL_1_STEADY, 0)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            inst.v0 = 999.0  # type: ignore[misc]


class SeedFormulaTests(unittest.TestCase):
    def test_derive_seed_matches_exact_mandated_formula(self):
        for stratum, episode_id, master_seed in [
            (StratumType.NOMINAL_1_STEADY, 0, generator.DEFAULT_MASTER_SEED),
            (StratumType.SHIFT_6_SIMULTANEOUS_FAULTS, 12345, 7),
            (StratumType.NOMINAL_4_LOW_BRAKING, 59999, 0),
        ]:
            with self.subTest(stratum=stratum, episode_id=episode_id, master_seed=master_seed):
                expected = int(
                    hashlib.sha256(f"{master_seed}:{stratum.name}:{episode_id}".encode()).hexdigest()[:8], 16
                )
                self.assertEqual(generator.derive_seed(master_seed, stratum, episode_id), expected)

    def test_seed_collision_rate_matches_birthday_bound_at_full_scale(self):
        """The mandated formula truncates SHA-256 to 8 hex chars (32 bits),
        not the full digest -- with 420,000 total episodes, the birthday
        bound puts the *expected* collision count at roughly
        420000^2 / (2 * 2**32) =~ 20.5. This is a real, documented property
        of the exact formula this generator must use, not a bug: this test
        measures it directly rather than asserting (incorrectly) that
        collisions never happen."""
        seeds = [
            generator.derive_seed(generator.DEFAULT_MASTER_SEED, stratum, episode_id)
            for stratum in ALL_STRATA
            for episode_id in range(episodes_for(stratum))
        ]
        n = len(seeds)
        self.assertEqual(n, TOTAL_EPISODES)
        n_unique = len(set(seeds))
        collisions = n - n_unique

        expected_collisions = n * n / (2 * 2**32)
        # Generous band around the theoretical estimate -- this is a random
        # process, not an exact count, but it should land in the right
        # neighborhood (order of magnitude), not zero and not thousands.
        self.assertGreater(collisions, 0, "expected some collisions at this scale with a 32-bit truncated hash")
        self.assertLess(collisions, expected_collisions * 5)


class DeterminismAndCustodyTests(unittest.TestCase):
    def test_100_percent_deterministic_reproduction_across_repeated_runs(self):
        """Regenerating the same episodes from the same master_seed, in
        three independent calls, must be byte-for-byte identical every time
        -- the brief's explicit determinism requirement."""
        probes = [
            (StratumType.NOMINAL_1_STEADY, 0),
            (StratumType.NOMINAL_6_ADVERSE_COMPOUND, 59_999),
            (StratumType.SHIFT_3_COMMON_BIAS_EXTREME, 4_321),
            (StratumType.SHIFT_4_OBS_DROPOUT_500MS, 8_888),
            (StratumType.SHIFT_6_SIMULTANEOUS_FAULTS, 9_999),
        ]
        for stratum, episode_id in probes:
            with self.subTest(stratum=stratum, episode_id=episode_id):
                runs = [generate_episode(stratum, episode_id) for _ in range(3)]
                self.assertTrue(all(r == runs[0] for r in runs), runs)

        # and at batch granularity via the streaming entry point
        batch_runs = [list(generate_stratum_sample(StratumType.NOMINAL_2_BOUNDARY_NOISE, 200)) for _ in range(3)]
        self.assertTrue(all(b == batch_runs[0] for b in batch_runs))

    def test_generate_stratum_matches_generate_episode_and_iter_stratum(self):
        stratum = StratumType.NOMINAL_2_BOUNDARY_NOISE
        batch = generate_stratum(stratum)
        self.assertEqual(len(batch), EPISODES_PER_NOMINAL_STRATUM)

        for episode_id in (0, 1, 25_000, EPISODES_PER_NOMINAL_STRATUM - 1):
            self.assertEqual(batch[episode_id], generate_episode(stratum, episode_id))

        self.assertEqual(list(iter_stratum(stratum)), batch)

    def test_generate_stratum_sample_is_a_lazy_streaming_generator(self):
        """Do NOT write large batch arrays directly to disk: the primary
        chunked entry point must actually be a generator (so a caller can
        consume/write it incrementally), and must match the corresponding
        prefix of the full stratum."""
        gen = generate_stratum_sample(StratumType.NOMINAL_3_SHARED_BIAS, 50)
        self.assertIsInstance(gen, types.GeneratorType)

        sample = list(gen)
        full_prefix = generate_stratum(StratumType.NOMINAL_3_SHARED_BIAS)[:50]
        self.assertEqual(sample, full_prefix)

        # count is clamped to the stratum's actual size, not an error
        clamped = list(generate_stratum_sample(StratumType.SHIFT_1_DEGRADED_BRAKE, EPISODES_PER_SHIFT_STRATUM + 500))
        self.assertEqual(len(clamped), EPISODES_PER_SHIFT_STRATUM)

        with self.assertRaises(ValueError):
            list(generate_stratum_sample(StratumType.NOMINAL_1_STEADY, -1))

    def test_seed_uniqueness_across_distinct_cross_stratum_sample(self):
        # Sample *distinct* (stratum, episode_id) pairs (rng.sample, not
        # rng.choice + randrange with replacement) so this actually tests
        # derive_seed's collision-resistance at a scale small enough that
        # the birthday bound (see SeedFormulaTests) predicts ~0 collisions
        # -- distinct from the full-population test, which expects some.
        rng = random.Random(7)
        pairs = []
        per_stratum = 500
        for stratum in ALL_STRATA:
            n = episodes_for(stratum)
            for episode_id in rng.sample(range(n), min(per_stratum, n)):
                pairs.append((stratum, episode_id))

        seeds = {generate_episode(stratum, episode_id).seed for stratum, episode_id in pairs}
        self.assertEqual(len(seeds), len(pairs), "expected no seed collisions across a distinct cross-stratum sample")

    def test_different_master_seed_changes_parameters(self):
        a = generate_episode(StratumType.NOMINAL_1_STEADY, 0, master_seed=generator.DEFAULT_MASTER_SEED)
        b = generate_episode(StratumType.NOMINAL_1_STEADY, 0, master_seed=generator.DEFAULT_MASTER_SEED + 1)
        self.assertNotEqual(a.seed, b.seed)
        self.assertNotEqual(
            (a.v0, a.actuator_delay, a.braking_deceleration), (b.v0, b.actuator_delay, b.braking_deceleration)
        )
        # both are still valid, recoverable nominal episodes
        self.assertTrue(a.is_recoverable)
        self.assertTrue(b.is_recoverable)

    def test_seed_and_parameters_independent_of_stratum_processing_order(self):
        solo = generate_episode(StratumType.NOMINAL_3_SHARED_BIAS, 0)
        via_iter = next(iter_stratum(StratumType.NOMINAL_3_SHARED_BIAS))
        self.assertEqual(solo, via_iter)

        # A custom-ordered, arbitrary-subset run must reproduce the same
        # per-episode result regardless of what else is being generated
        # around it or in what order -- no shared/incrementing stream.
        subset = list(iter_all(strata=(StratumType.SHIFT_2_DOUBLE_NOISE, StratumType.NOMINAL_3_SHARED_BIAS)))
        self.assertEqual(len(subset), EPISODES_PER_SHIFT_STRATUM + EPISODES_PER_NOMINAL_STRATUM)
        nominal3_via_subset = subset[EPISODES_PER_SHIFT_STRATUM]  # first episode after SHIFT_2's block
        self.assertEqual(nominal3_via_subset, solo)

    def test_x0_always_zero(self):
        for stratum in ALL_STRATA:
            for episode_id in (0, episodes_for(stratum) - 1):
                self.assertEqual(generate_episode(stratum, episode_id).x0, 0.0)

    def test_generate_episode_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            generate_episode(StratumType.NOMINAL_1_STEADY, -1)
        with self.assertRaises(ValueError):
            generate_episode(StratumType.NOMINAL_1_STEADY, EPISODES_PER_NOMINAL_STRATUM)
        with self.assertRaises(ValueError):
            generate_episode(StratumType.SHIFT_1_DEGRADED_BRAKE, EPISODES_PER_SHIFT_STRATUM)


class RecoverabilityGuaranteeTests(unittest.TestCase):
    def test_1000_nominal_instances_strictly_recoverable(self):
        """The brief's explicit verification requirement: sample 1,000
        instances per nominal stratum via the streaming entry point and
        assert every single one is strictly is_recoverable == True."""
        for stratum in NOMINAL_STRATA:
            with self.subTest(stratum=stratum):
                sample = list(generate_stratum_sample(stratum, 1000))
                self.assertEqual(len(sample), 1000)
                self.assertTrue(all(inst.is_recoverable is True for inst in sample))

    def test_all_nominal_episodes_recoverable_full_population(self):
        """The stronger, exhaustive version: all 360,000 primary nominal
        episodes (across all six nominal strata) must be physically
        recoverable at t=0 per sim.plant.analytic_plant, under maximum
        (sampled) actuator delay and full emergency braking."""
        checked = 0
        for stratum in NOMINAL_STRATA:
            for inst in iter_stratum(stratum):
                self.assertTrue(
                    inst.is_recoverable,
                    f"{stratum.name} episode {inst.episode_id} is not recoverable: {inst}",
                )
                checked += 1
        self.assertEqual(checked, TOTAL_NOMINAL_EPISODES)

    def test_recoverability_clearance_formula_matches_spec(self):
        """clearance = x_obs - x_stop >= 0.10 m, evaluated directly against
        sim.plant.analytic_plant.stopping_position (command_time=0.0),
        independent of generator._is_recoverable's own implementation."""
        from sim.plant import PlantParameters, PlantState
        from sim.plant.analytic_plant import stopping_position

        rng = random.Random(99)
        for _ in range(200):
            v0 = rng.uniform(*V0_NOMINAL)
            tau = rng.uniform(*TAU_NOMINAL)
            b = rng.uniform(0.05, 1.0)
            obstacle = rng.uniform(0.3, 5.0)

            state = PlantState(position=0.0, velocity=v0, acceleration=0.0)
            params = PlantParameters(actuator_delay=tau, braking_deceleration=b, obstacle_position=obstacle)
            x_stop = stopping_position(state, params, command_time=0.0)
            clearance = obstacle - x_stop
            expected = clearance >= CLEARANCE_MARGIN_M

            self.assertEqual(generator._is_recoverable(0.0, v0, obstacle, b, tau), expected)

    def test_nominal_worst_case_margin_holds_by_construction(self):
        """Documents and locks in *why* the guarantee above never needs its
        resample/fallback path: worst-case nominal kinematics (max v0, max
        actuator_delay, min braking_deceleration -- each nominal stratum's
        sampling sub-bands are within [V0_NOMINAL, TAU_NOMINAL] and
        braking_deceleration >= 0.2) stop with room to spare inside the
        nearest edge of OBSTACLE_NOMINAL, even after the clearance margin."""
        v0_max = V0_NOMINAL[1]
        tau_max = TAU_NOMINAL[1]
        b_min = 0.2  # sim/plant's declared floor; also every nominal stratum's floor
        worst_case_stop = v0_max * tau_max + (v0_max**2) / (2.0 * b_min)
        self.assertLess(worst_case_stop + CLEARANCE_MARGIN_M, OBSTACLE_NOMINAL[0])

    def test_shift_strata_are_not_universally_recoverable(self):
        """Sanity check that the shift envelopes actually bite: if every
        shift episode came back recoverable, the 'distribution shift' would
        be cosmetic. Degraded-brake strata in particular must show some
        unrecoverable episodes."""
        degraded_strata = (StratumType.SHIFT_1_DEGRADED_BRAKE, StratumType.SHIFT_6_SIMULTANEOUS_FAULTS)
        for stratum in degraded_strata:
            with self.subTest(stratum=stratum):
                batch = generate_stratum(stratum)
                n_unrecoverable = sum(1 for inst in batch if not inst.is_recoverable)
                self.assertGreater(n_unrecoverable, 0, f"{stratum.name} should include some unrecoverable episodes")

        # and shift strata are not filtered/resampled the way nominal ones are
        all_shift = [inst for s in SHIFT_STRATA for inst in generate_stratum(s)]
        self.assertEqual(len(all_shift), TOTAL_SHIFT_EPISODES)
        self.assertTrue(any(not inst.is_recoverable for inst in all_shift))

    def test_recoverability_fallback_path_actually_guarantees_recoverable(self):
        """Directly exercises the defense-in-depth path (rather than only
        observing that it never fires): force every sample attempt for a
        nominal stratum to be unrecoverable, and confirm _generate_one still
        returns is_recoverable=True by deterministically widening the
        obstacle, instead of ever emitting a broken 'nominal' episode."""

        def _always_unrecoverable(_rng):
            return dict(
                v0=0.5,
                obstacle_position=0.5,  # far too close for v0=0.5 to ever stop short of, at this b/tau
                braking_deceleration=0.2,
                actuator_delay=0.10,
                sensor_bias=0.0,
                sensor_noise_scale=1.0,
            )

        stratum = StratumType.NOMINAL_1_STEADY
        original_sampler = generator._SAMPLERS[stratum]
        generator._SAMPLERS[stratum] = _always_unrecoverable
        try:
            self.assertFalse(
                generator._is_recoverable(0.0, 0.5, 0.5, 0.2, 0.10),
                "test setup error: the forced sample should itself be unrecoverable",
            )
            inst = generator._generate_one(stratum, 0, generator.DEFAULT_MASTER_SEED)
        finally:
            generator._SAMPLERS[stratum] = original_sampler

        self.assertTrue(inst.is_recoverable)
        self.assertGreater(inst.obstacle_position, 0.5)


class OutOfEnvelopeTests(unittest.TestCase):
    """The brief's explicit requirement: check Shift 1 and Shift 4
    out-of-envelope bounds -- i.e. confirm each stratum's defining stressor
    actually pushes its sampled parameter outside the nominal envelope,
    every single episode, not just on average."""

    def test_shift_1_degraded_brake_is_out_of_nominal_envelope(self):
        nominal_b_floor = 0.2  # every nominal stratum's braking_deceleration floor
        batch = generate_stratum(StratumType.SHIFT_1_DEGRADED_BRAKE)
        self.assertTrue(all(inst.braking_deceleration < nominal_b_floor for inst in batch))
        self.assertTrue(all(0.05 <= inst.braking_deceleration < 0.2 for inst in batch))

    def test_shift_4_extreme_noise_scale_is_out_of_nominal_envelope(self):
        # highest any nominal stratum's sensor_noise_scale ever reaches
        nominal_noise_ceiling = 1.5  # NOMINAL_2_BOUNDARY_NOISE's upper bound
        batch = generate_stratum(StratumType.SHIFT_4_OBS_DROPOUT_500MS)
        self.assertTrue(all(inst.sensor_noise_scale > nominal_noise_ceiling for inst in batch))
        self.assertTrue(all(5.0 <= inst.sensor_noise_scale < 10.0 for inst in batch))
        # and its kinematics stay nominal-ish (isolated stressor)
        self.assertTrue(all(0.4 <= inst.braking_deceleration <= 1.0 for inst in batch))


class SamplingEnvelopeTests(unittest.TestCase):
    def test_parameter_bounds_per_stratum(self):
        bounds = {
            StratumType.NOMINAL_1_STEADY: dict(b=(0.4, 1.0), tau=(0.0, 0.05)),
            StratumType.NOMINAL_2_BOUNDARY_NOISE: dict(b=(0.4, 1.0), tau=(0.0, 0.05)),
            StratumType.NOMINAL_3_SHARED_BIAS: dict(b=(0.4, 1.0), tau=(0.0, 0.05)),
            StratumType.NOMINAL_4_LOW_BRAKING: dict(b=(0.2, 0.4), tau=(0.0, 0.05)),
            StratumType.NOMINAL_5_SENSOR_DELAY: dict(b=(0.4, 1.0), tau=(0.06, 0.10)),
            StratumType.NOMINAL_6_ADVERSE_COMPOUND: dict(b=(0.2, 0.5), tau=(0.05, 0.10)),
            StratumType.SHIFT_1_DEGRADED_BRAKE: dict(b=(0.05, 0.2), tau=TAU_NOMINAL),
            StratumType.SHIFT_2_DOUBLE_NOISE: dict(b=(0.4, 1.0), tau=TAU_NOMINAL),
            StratumType.SHIFT_3_COMMON_BIAS_EXTREME: dict(b=(0.4, 1.0), tau=TAU_NOMINAL),
            StratumType.SHIFT_4_OBS_DROPOUT_500MS: dict(b=(0.4, 1.0), tau=TAU_NOMINAL),
            StratumType.SHIFT_6_SIMULTANEOUS_FAULTS: dict(b=(0.05, 0.2), tau=(0.06, 0.10)),
        }
        for stratum, spec in bounds.items():
            with self.subTest(stratum=stratum):
                for inst in generate_stratum(stratum):
                    self.assertTrue(spec["b"][0] <= inst.braking_deceleration <= spec["b"][1])
                    self.assertTrue(spec["tau"][0] <= inst.actuator_delay <= spec["tau"][1])
                    self.assertTrue(V0_NOMINAL[0] <= inst.v0 <= V0_NOMINAL[1])
                    self.assertTrue(
                        OBSTACLE_NOMINAL[0] <= inst.obstacle_position <= OBSTACLE_NOMINAL[1]
                        or stratum not in NOMINAL_STRATA
                    )

    def test_shift_5_surface_distribution_is_a_mixture(self):
        batch = generate_stratum(StratumType.SHIFT_5_SURFACE_DISTRIBUTION)
        degraded = [inst for inst in batch if inst.braking_deceleration < 0.4]
        nominal_band = [inst for inst in batch if inst.braking_deceleration >= 0.4]
        self.assertGreater(len(degraded), 0)
        self.assertGreater(len(nominal_band), 0)
        # mixture weight is 30% degraded; check it landed in a sane range for 10,000 draws
        fraction_degraded = len(degraded) / len(batch)
        self.assertGreater(fraction_degraded, 0.20)
        self.assertLess(fraction_degraded, 0.40)


class ManifestSealTests(unittest.TestCase):
    def test_manifest_digest_reproducible(self):
        d1 = manifest_digest(generate_stratum(StratumType.NOMINAL_1_STEADY))
        d2 = manifest_digest(generate_stratum(StratumType.NOMINAL_1_STEADY))
        self.assertEqual(d1, d2)
        self.assertEqual(len(d1), 64)  # sha256 hex digest length
        int(d1, 16)  # must be valid hex

    def test_manifest_digest_sensitive_to_master_seed(self):
        d1 = manifest_digest(generate_stratum(StratumType.NOMINAL_1_STEADY, master_seed=1))
        d2 = manifest_digest(generate_stratum(StratumType.NOMINAL_1_STEADY, master_seed=2))
        self.assertNotEqual(d1, d2)

    def test_manifest_digest_sensitive_to_tampering(self):
        batch = generate_stratum(StratumType.NOMINAL_2_BOUNDARY_NOISE)
        original_digest = manifest_digest(batch)

        tampered = list(batch)
        tampered[500] = dataclasses.replace(tampered[500], v0=tampered[500].v0 + 1e-6)
        tampered_digest = manifest_digest(tampered)

        self.assertNotEqual(original_digest, tampered_digest)

    def test_manifest_digest_sensitive_to_order(self):
        batch = generate_stratum(StratumType.NOMINAL_1_STEADY)[:100]
        forward = manifest_digest(batch)
        backward = manifest_digest(list(reversed(batch)))
        self.assertNotEqual(forward, backward)


if __name__ == "__main__":
    unittest.main(verbosity=2)
