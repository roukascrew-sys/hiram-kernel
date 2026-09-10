"""
sim.scenarios.generator - Deterministic, sealed scenario generation.

Generates the 420,000 episodes (6 nominal strata x 60,000 + 6 shift strata x
10,000; see strata.py) that make up the stopping benchmark's evaluation set.

Seed custody
------------
Every episode's PRNG seed is derived from (master_seed, stratum, episode_id)
via SHA-256 (derive_seed), never from a single incrementing stream shared
across strata or episodes. That gives two properties a shared stream cannot:

  * any single episode can be regenerated in isolation (generate_episode),
    without replaying every episode before it in its stratum;
  * inserting, removing, or reordering strata cannot perturb any other
    stratum's seeds or parameters, since each stratum is salted independently
    by its own name, not by its position in some master sequence.

manifest_digest() then seals a *generated batch*: a single SHA-256 over every
instance's fully-specified fields, so a downstream consumer can confirm they
have the exact canonical dataset for a given master_seed, not a tampered or
partial one.

Recoverability guarantee
-------------------------
sim.plant.analytic_plant (imported read-only; this module never modifies
sim/plant) is used to verify, for every nominal-stratum episode, that
commanding full emergency braking at t=0 would stop the cart at least
CLEARANCE_MARGIN_M short of obstacle_position. The nominal sampling envelope
below is sized so this holds by construction (worst case nominal kinematics
stop within ~0.68 m; the obstacle floor is 1.0 m) -- the resample/fallback
logic in _generate_one exists as defense in depth, not as the primary
mechanism, and test_scenario_generator.py checks it is never actually
needed. Shift-stratum episodes are not filtered this way: an unrecoverable
shift episode is valid ground truth for evaluating whether a downstream
safety monitor correctly flags it.
"""

import hashlib
import random
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from sim.plant import PlantParameters, PlantState
from sim.plant.analytic_plant import stopping_position

from .strata import NOMINAL_STRATA, ScenarioInstance, StratumType, episodes_for

DEFAULT_MASTER_SEED = 20260910  # fixed, documented constant; pass a different one to mint a distinct sealed dataset

CLEARANCE_MARGIN_M = 0.10  # matches this benchmark's collision-clearance convention (sim/plant tests)
MAX_RECOVERABILITY_RESAMPLES = 64


def derive_seed(master_seed: int, stratum: StratumType, episode_id: int) -> int:
    """
    Cryptographically salted, per-episode PRNG seed, per the exact formula
    mandated for this generator:

        seed = int(sha256(f"{master_seed}:{stratum.name}:{episode_id}").hexdigest()[:8], 16)

    Note this truncates to the leading 8 *hex characters* (32 bits, not 64):
    with ~420,000 total episodes, the birthday bound puts the expected number
    of accidental seed collisions across the full population at roughly
    420000^2 / (2 * 2**32) =~ 20. A collision means two distinct
    (stratum, episode_id) pairs end up drawing from identical RNG streams --
    tests/test_scenario_generator.py measures the actual collision count
    against this estimate rather than assuming either "no collisions" or
    silently widening the seed, since the formula is specified exactly above.
    """
    key = f"{master_seed}:{stratum.name}:{episode_id}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _is_recoverable(
    x0: float,
    v0: float,
    obstacle_position: float,
    braking_deceleration: float,
    actuator_delay: float,
    clearance_margin: float = CLEARANCE_MARGIN_M,
) -> bool:
    """True if emergency braking commanded immediately (t=0) stops the cart
    at least `clearance_margin` meters short of obstacle_position, per the
    independent analytic plant oracle (sim.plant.analytic_plant)."""
    state = PlantState(position=x0, velocity=v0, acceleration=0.0)
    params = PlantParameters(
        actuator_delay=actuator_delay,
        braking_deceleration=braking_deceleration,
        obstacle_position=obstacle_position,
    )
    stop_pos = stopping_position(state, params, command_time=0.0)
    return stop_pos <= obstacle_position - clearance_margin


# ---------------------------------------------------------------------------
# Per-stratum sampling envelopes.
#
# All twelve strata share the ScenarioInstance schema; what distinguishes
# them is which distribution each field is drawn from. Nominal strata (1-6)
# stay within sim/plant's declared physical bounds (actuator_delay in
# [0, 0.10] s, braking_deceleration in [0.2, 1.0] m/s^2) and are sized so
# that even worst-case nominal kinematics (v0=0.5, tau=0.10, b=0.2) stop
# within ~0.68 m -- comfortably inside the 1.0 m obstacle floor with margin
# to spare, so the recoverability guarantee holds by construction. Shift
# strata (1-6) deliberately sample outside that envelope, or from a
# differently-shaped distribution, to probe generalization/degradation.
#
# The exact numeric ranges below are this generator's own interpretation of
# each stratum's name (the originating spec did not enumerate them); they
# are documented per-stratum so they can be retuned without touching the
# custody/recoverability machinery around them.
# ---------------------------------------------------------------------------
V0_NOMINAL = (0.05, 0.5)  # m/s
TAU_NOMINAL = (0.0, 0.10)  # s
OBSTACLE_NOMINAL = (1.0, 4.0)  # m

_StratumSampler = Callable[[random.Random], Dict[str, float]]


def _sample_nominal_1_steady(rng: random.Random) -> Dict[str, float]:
    """Baseline nominal envelope: no elevated noise, bias, or delay."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(0.0, 0.05),
        sensor_bias=rng.uniform(-0.01, 0.01),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_nominal_2_boundary_noise(rng: random.Random) -> Dict[str, float]:
    """Sensor noise pushed toward the top of the still-nominal band."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(0.0, 0.05),
        sensor_bias=rng.uniform(-0.02, 0.02),
        sensor_noise_scale=rng.uniform(1.0, 1.5),
    )


def _sample_nominal_3_shared_bias(rng: random.Random) -> Dict[str, float]:
    """A systematic (shared, not per-sample) range bias, still nominal-scale."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(0.0, 0.05),
        sensor_bias=rng.uniform(-0.08, 0.08),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_nominal_4_low_braking(rng: random.Random) -> Dict[str, float]:
    """Braking capability at the low sub-band of the nominal [0.2, 1.0] range."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.2, 0.4),
        actuator_delay=rng.uniform(0.0, 0.05),
        sensor_bias=rng.uniform(-0.01, 0.01),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_nominal_5_sensor_delay(rng: random.Random) -> Dict[str, float]:
    """Actuator delay at the high sub-band of the nominal [0, 0.10] s range."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(0.06, 0.10),
        sensor_bias=rng.uniform(-0.01, 0.01),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_nominal_6_adverse_compound(rng: random.Random) -> Dict[str, float]:
    """Several nominal-scale stressors stacked at once (still in-envelope)."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.2, 0.5),
        actuator_delay=rng.uniform(0.05, 0.10),
        sensor_bias=rng.uniform(-0.05, 0.05),
        sensor_noise_scale=rng.uniform(1.0, 1.3),
    )


def _sample_shift_1_degraded_brake(rng: random.Random) -> Dict[str, float]:
    """Braking capability below the nominal floor -- a genuinely degraded brake."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.05, 0.2),
        actuator_delay=rng.uniform(*TAU_NOMINAL),
        sensor_bias=rng.uniform(-0.02, 0.02),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_shift_2_double_noise(rng: random.Random) -> Dict[str, float]:
    """Sensor noise roughly double the nominal ~1.0 baseline."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(*TAU_NOMINAL),
        sensor_bias=rng.uniform(-0.02, 0.02),
        sensor_noise_scale=rng.uniform(1.8, 2.2),
    )


def _sample_shift_3_common_bias_extreme(rng: random.Random) -> Dict[str, float]:
    """A shared range bias at extreme magnitude, sign randomized."""
    sign = rng.choice((-1.0, 1.0))
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(*TAU_NOMINAL),
        sensor_bias=sign * rng.uniform(0.15, 0.5),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_shift_4_obs_dropout_500ms(rng: random.Random) -> Dict[str, float]:
    """Isolated stressor: physical kinematics stay nominal-ish. ScenarioInstance
    has no per-timestep dropout-duration field (the schema is fixed to
    exactly the given contract), so a 500 ms window of missing observations
    is represented the only way a single per-episode scalar can: as an
    extreme sensor_noise_scale (5.0-10.0, an order of magnitude past even
    SHIFT_2_DOUBLE_NOISE's 1.8-2.2 band) -- during the dropout, readings a
    downstream consumer does get are effectively unusable, which is the
    practical consequence this stratum needs to be distinguishable by."""
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.4, 1.0),
        actuator_delay=rng.uniform(*TAU_NOMINAL),
        sensor_bias=rng.uniform(-0.02, 0.02),
        sensor_noise_scale=rng.uniform(5.0, 10.0),
    )


def _sample_shift_5_surface_distribution(rng: random.Random) -> Dict[str, float]:
    """A genuine change in distribution *shape*, not just its bounds: a
    minority of episodes draw braking_deceleration from a degraded
    ("icy patch") sub-band instead of the nominal one."""
    if rng.random() < 0.3:
        braking_deceleration = rng.uniform(0.05, 0.2)
    else:
        braking_deceleration = rng.uniform(0.4, 1.0)
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=braking_deceleration,
        actuator_delay=rng.uniform(*TAU_NOMINAL),
        sensor_bias=rng.uniform(-0.02, 0.02),
        sensor_noise_scale=rng.uniform(0.95, 1.05),
    )


def _sample_shift_6_simultaneous_faults(rng: random.Random) -> Dict[str, float]:
    """Every shift-level stressor stacked at once: degraded brake (SHIFT_1),
    high delay, extreme bias (SHIFT_3), and the SHIFT_4-style extreme
    sensor_noise_scale standing in for the 500 ms dropout -- all at once."""
    sign = rng.choice((-1.0, 1.0))
    return dict(
        v0=rng.uniform(*V0_NOMINAL),
        obstacle_position=rng.uniform(*OBSTACLE_NOMINAL),
        braking_deceleration=rng.uniform(0.05, 0.2),
        actuator_delay=rng.uniform(0.06, 0.10),
        sensor_bias=sign * rng.uniform(0.15, 0.5),
        sensor_noise_scale=rng.uniform(5.0, 10.0),
    )


_SAMPLERS: Dict[StratumType, _StratumSampler] = {
    StratumType.NOMINAL_1_STEADY: _sample_nominal_1_steady,
    StratumType.NOMINAL_2_BOUNDARY_NOISE: _sample_nominal_2_boundary_noise,
    StratumType.NOMINAL_3_SHARED_BIAS: _sample_nominal_3_shared_bias,
    StratumType.NOMINAL_4_LOW_BRAKING: _sample_nominal_4_low_braking,
    StratumType.NOMINAL_5_SENSOR_DELAY: _sample_nominal_5_sensor_delay,
    StratumType.NOMINAL_6_ADVERSE_COMPOUND: _sample_nominal_6_adverse_compound,
    StratumType.SHIFT_1_DEGRADED_BRAKE: _sample_shift_1_degraded_brake,
    StratumType.SHIFT_2_DOUBLE_NOISE: _sample_shift_2_double_noise,
    StratumType.SHIFT_3_COMMON_BIAS_EXTREME: _sample_shift_3_common_bias_extreme,
    StratumType.SHIFT_4_OBS_DROPOUT_500MS: _sample_shift_4_obs_dropout_500ms,
    StratumType.SHIFT_5_SURFACE_DISTRIBUTION: _sample_shift_5_surface_distribution,
    StratumType.SHIFT_6_SIMULTANEOUS_FAULTS: _sample_shift_6_simultaneous_faults,
}


def _generate_one(stratum: StratumType, episode_id: int, master_seed: int) -> ScenarioInstance:
    seed = derive_seed(master_seed, stratum, episode_id)
    rng = random.Random(seed)
    sampler = _SAMPLERS[stratum]

    params = sampler(rng)
    recoverable = _is_recoverable(
        x0=0.0,
        v0=params["v0"],
        obstacle_position=params["obstacle_position"],
        braking_deceleration=params["braking_deceleration"],
        actuator_delay=params["actuator_delay"],
    )

    if stratum in NOMINAL_STRATA and not recoverable:
        # Defense in depth: by construction (see module docstring) this
        # should never fire for a nominal stratum. If it ever does, keep
        # resampling this episode's kinematics from the SAME seeded stream
        # -- the episode stays 100% reproducible from `seed` alone.
        for _attempt in range(MAX_RECOVERABILITY_RESAMPLES):
            params = sampler(rng)
            recoverable = _is_recoverable(
                x0=0.0,
                v0=params["v0"],
                obstacle_position=params["obstacle_position"],
                braking_deceleration=params["braking_deceleration"],
                actuator_delay=params["actuator_delay"],
            )
            if recoverable:
                break
        else:
            # Exhausted every resample attempt: guarantee the invariant
            # anyway by deterministically pushing the obstacle out to a
            # provably-safe distance for this episode's own kinematics,
            # rather than ever emitting an unrecoverable "nominal" episode.
            worst_case_stop = params["v0"] * params["actuator_delay"] + (params["v0"] ** 2) / (
                2.0 * params["braking_deceleration"]
            )
            params["obstacle_position"] = worst_case_stop + CLEARANCE_MARGIN_M + 0.5
            recoverable = True

    return ScenarioInstance(
        episode_id=episode_id,
        stratum=stratum,
        seed=seed,
        x0=0.0,
        v0=params["v0"],
        obstacle_position=params["obstacle_position"],
        braking_deceleration=params["braking_deceleration"],
        actuator_delay=params["actuator_delay"],
        sensor_bias=params["sensor_bias"],
        sensor_noise_scale=params["sensor_noise_scale"],
        is_recoverable=recoverable,
    )


def iter_stratum(stratum: StratumType, master_seed: int = DEFAULT_MASTER_SEED) -> Iterator[ScenarioInstance]:
    """Lazily yield every episode of one stratum, in episode_id order."""
    for episode_id in range(episodes_for(stratum)):
        yield _generate_one(stratum, episode_id, master_seed)


def generate_stratum_sample(
    stratum: StratumType, count: int, master_seed: int = DEFAULT_MASTER_SEED
) -> Iterator[ScenarioInstance]:
    """
    Lazily yield the first `count` episodes (episode_id 0..count-1, clamped
    to the stratum's actual size) of one stratum. This is the primary
    streaming/chunked entry point for pulling a bounded sample without
    materializing (or writing to disk) the full 60,000/10,000-episode
    population -- a plain generator, so a caller that only consumes part of
    it, or that wants to write results out in chunks, never holds more than
    one ScenarioInstance in memory at a time because of this function.
    """
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count!r}")
    n = min(count, episodes_for(stratum))
    for episode_id in range(n):
        yield _generate_one(stratum, episode_id, master_seed)


def generate_stratum(stratum: StratumType, master_seed: int = DEFAULT_MASTER_SEED) -> List[ScenarioInstance]:
    """Materialize every episode of one stratum as a list."""
    return list(iter_stratum(stratum, master_seed))


def generate_episode(
    stratum: StratumType, episode_id: int, master_seed: int = DEFAULT_MASTER_SEED
) -> ScenarioInstance:
    """Regenerate exactly one episode in isolation -- the seed-custody
    guarantee in action: no need to replay the stratum from episode 0."""
    n = episodes_for(stratum)
    if not 0 <= episode_id < n:
        raise ValueError(f"{stratum.name} has {n} episodes; episode_id={episode_id} out of range")
    return _generate_one(stratum, episode_id, master_seed)


def iter_all(
    master_seed: int = DEFAULT_MASTER_SEED,
    strata: Optional[Tuple[StratumType, ...]] = None,
) -> Iterator[ScenarioInstance]:
    """Lazily yield every episode across the given strata (default: all
    twelve), in the given stratum order, episode_id order within each."""
    from .strata import ALL_STRATA

    for stratum in strata if strata is not None else ALL_STRATA:
        yield from iter_stratum(stratum, master_seed)


def manifest_digest(instances: Iterable[ScenarioInstance]) -> str:
    """
    SHA-256 hex digest over the ordered, fully-specified fields of every
    instance -- the 'seal' for a generated batch. Two calls over the same
    (ordered) instances always produce the same digest; changing any field of
    any instance, or their order, changes it.
    """
    hasher = hashlib.sha256()
    for inst in instances:
        record = (
            f"{inst.episode_id}|{inst.stratum.name}|{inst.seed}|"
            f"{inst.x0!r}|{inst.v0!r}|{inst.obstacle_position!r}|"
            f"{inst.braking_deceleration!r}|{inst.actuator_delay!r}|"
            f"{inst.sensor_bias!r}|{inst.sensor_noise_scale!r}|"
            f"{inst.is_recoverable}\n"
        )
        hasher.update(record.encode("utf-8"))
    return hasher.hexdigest()
