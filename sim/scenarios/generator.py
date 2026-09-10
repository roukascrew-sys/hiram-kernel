"""
Boundary-dense scenario generator with guaranteed nominal recoverability,
full 12-strata physical logic, strict sequence custody validation, and verified dataset sealing.
"""

import hashlib
import json
import math
import random
from typing import Dict, Generator, Iterable, List, Optional

from sim.plant import PlantParameters, PlantState
from sim.plant import analytic_plant
from .strata import ScenarioInstance, StratumType, derive_domain_seed

DEFAULT_MASTER_SEED = "HIRAM-BENCHMARK-2026-SEALED-V1"
SCHEMA_VERSION = "1.0.0-PROD"
GENERATOR_REVISION = "rev-m0-final"
NOMINAL_EPISODES_PER_STRATUM = 60_000
SHIFT_EPISODES_PER_STRATUM = 10_000
TOTAL_BENCHMARK_EPISODES = 6 * NOMINAL_EPISODES_PER_STRATUM + 6 * SHIFT_EPISODES_PER_STRATUM  # 420,000


def canonical_episode_count(stratum: StratumType) -> int:
    return SHIFT_EPISODES_PER_STRATUM if stratum.name.startswith("SHIFT") else NOMINAL_EPISODES_PER_STRATUM


def derive_seed(master_seed: str, stratum_name: str, episode_id: int, domain: str = "dynamics") -> int:
    return derive_domain_seed(master_seed, stratum_name, episode_id, domain)


def generate_instance(master_seed: str, stratum: StratumType, episode_id: int) -> ScenarioInstance:
    """Generates a single reproducible scenario instance implementing explicit strata physics."""
    dynamics_seed = derive_domain_seed(master_seed, stratum.name, episode_id, "dynamics")
    rng = random.Random(dynamics_seed)

    x0 = 0.0
    v0 = rng.uniform(0.0, 0.50)
    a_cmd = rng.uniform(-0.25, 0.25)
    tau = rng.uniform(0.0, 0.10)
    b = rng.uniform(0.20, 1.0)
    bias = rng.uniform(-0.03, 0.03)
    noise_scale = 1.0
    dropout_start = 0.0
    dropout_duration = 0.0

    # Explicit 12-Stratum Parameterization
    if stratum == StratumType.NOMINAL_1_STEADY:
        pass
    elif stratum == StratumType.NOMINAL_2_BOUNDARY_NOISE:
        noise_scale = rng.uniform(1.2, 1.5)
    elif stratum == StratumType.NOMINAL_3_SHARED_BIAS:
        bias = rng.uniform(-0.05, 0.05)
    elif stratum == StratumType.NOMINAL_4_LOW_BRAKING:
        b = rng.uniform(0.20, 0.35)
    elif stratum == StratumType.NOMINAL_5_SENSOR_DELAY:
        tau = rng.uniform(0.08, 0.10)
    elif stratum == StratumType.NOMINAL_6_ADVERSE_COMPOUND:
        b = rng.uniform(0.20, 0.35)
        tau = rng.uniform(0.07, 0.10)
        bias = rng.uniform(-0.04, 0.04)
        noise_scale = rng.uniform(1.2, 1.4)
    elif stratum == StratumType.SHIFT_1_DEGRADED_BRAKE:
        b = rng.uniform(0.15, 0.199)
    elif stratum == StratumType.SHIFT_2_DOUBLE_NOISE:
        noise_scale = 2.0
    elif stratum == StratumType.SHIFT_3_COMMON_BIAS_EXTREME:
        bias = rng.uniform(0.05, 0.10)
    elif stratum == StratumType.SHIFT_4_OBS_DROPOUT_500MS:
        dropout_start = 0.0
        dropout_duration = 0.50
    elif stratum == StratumType.SHIFT_5_SURFACE_DISTRIBUTION:
        b = rng.uniform(0.12, 0.18) if rng.random() < 0.5 else rng.uniform(0.18, 0.22)
    elif stratum == StratumType.SHIFT_6_SIMULTANEOUS_FAULTS:
        b = rng.uniform(0.15, 0.199)
        noise_scale = 2.0
        bias = rng.uniform(0.05, 0.10)
        dropout_start = 0.0
        dropout_duration = 0.50

    state = PlantState(position=x0, velocity=v0, acceleration=0.0)
    params = PlantParameters(
        actuator_delay=tau,
        braking_deceleration=b,
        commanded_acceleration=a_cmd,
        obstacle_position=100.0,
    )

    x_stop = analytic_plant.stopping_position(state, params, command_time=0.0)

    # Reserve margin allocation
    is_shift = stratum.name.startswith("SHIFT")
    if is_shift:
        # Dropout strata guarantee minimum obstacle clearance to prevent early truncation
        if stratum in (StratumType.SHIFT_4_OBS_DROPOUT_500MS, StratumType.SHIFT_6_SIMULTANEOUS_FAULTS):
            reserve_draw = rng.uniform(0.10, 1.0)
        else:
            reserve_draw = rng.uniform(-0.05, 1.0)
    else:
        u = rng.random()
        if u < 0.30:
            reserve_draw = rng.uniform(0.001, 0.05)  # 30% Critical boundary stress
        elif u < 0.70:
            reserve_draw = rng.uniform(0.05, 0.30)   # 40% Marginal band
        else:
            reserve_draw = rng.uniform(0.30, 1.50)   # 30% Open clearance

    obstacle_position = x_stop + 0.10 + reserve_draw
    clearance_margin = obstacle_position - x_stop
    is_recoverable = clearance_margin >= 0.10

    return ScenarioInstance(
        episode_id=episode_id,
        stratum=stratum,
        seed=dynamics_seed,
        x0=x0,
        v0=v0,
        commanded_acceleration=a_cmd,
        obstacle_position=obstacle_position,
        braking_deceleration=b,
        actuator_delay=tau,
        sensor_bias=bias,
        sensor_noise_scale=noise_scale,
        dropout_start_s=dropout_start,
        dropout_duration_s=dropout_duration,
        is_recoverable=is_recoverable,
    )


generate_episode = generate_instance


def generate_canonical_stratum(master_seed: str, stratum: StratumType) -> Generator[ScenarioInstance, None, None]:
    count = canonical_episode_count(stratum)
    for ep_id in range(count):
        yield generate_instance(master_seed, stratum, ep_id)


def generate_stratum_sample(master_seed: str, stratum: StratumType, count: int) -> List[ScenarioInstance]:
    return [generate_instance(master_seed, stratum, ep_id) for ep_id in range(count)]


def compute_dataset_hash(
    dataset_stream: Iterable[ScenarioInstance],
    master_seed: str,
    validate_canonical_sequence: bool = True,
) -> str:
    """
    Cryptographically digests canonical rows in strict order.
    Enforces exact 12 strata, declared counts, and consecutive indexing.
    """
    hasher = hashlib.sha256()
    header = (
        f"HIRAM-CUSTODY|V:{SCHEMA_VERSION}|REV:{GENERATOR_REVISION}|"
        f"SEED:{master_seed}|COUNT:{TOTAL_BENCHMARK_EPISODES}".encode("utf-8")
    )
    hasher.update(header)

    strata_list = list(StratumType)
    current_stratum_idx = 0
    current_expected_id = 0
    total_count = 0

    for instance in dataset_stream:
        if validate_canonical_sequence:
            expected_stratum = strata_list[current_stratum_idx]
            if instance.stratum != expected_stratum:
                raise ValueError(
                    f"Sequence error at row {total_count}: expected {expected_stratum.name}, got {instance.stratum.name}"
                )
            if instance.episode_id != current_expected_id:
                raise ValueError(
                    f"Sequence error in {expected_stratum.name}: expected id {current_expected_id}, got {instance.episode_id}"
                )

            current_expected_id += 1
            if current_expected_id == canonical_episode_count(expected_stratum):
                current_stratum_idx += 1
                current_expected_id = 0

        hasher.update(instance.canonical_repr().encode("utf-8"))
        total_count += 1

    if validate_canonical_sequence and total_count != TOTAL_BENCHMARK_EPISODES:
        raise ValueError(
            f"Dataset incomplete: expected {TOTAL_BENCHMARK_EPISODES} episodes, processed {total_count}."
        )

    return hasher.hexdigest()


def manifest_digest(master_seed: str = DEFAULT_MASTER_SEED) -> str:
    """Computes the full verified custody hash over all 420,000 canonical episodes."""
    def stream_all():
        for st in StratumType:
            for inst in generate_canonical_stratum(master_seed, st):
                yield inst
    return compute_dataset_hash(stream_all(), master_seed, validate_canonical_sequence=True)


def build_manifest_metadata(master_seed: str = DEFAULT_MASTER_SEED) -> Dict:
    """Builds the complete custody manifest dictionary for archival verification."""
    digest = manifest_digest(master_seed)
    counts = {st.name: canonical_episode_count(st) for st in StratumType}
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_revision": GENERATOR_REVISION,
        "master_seed": master_seed,
        "total_episodes": TOTAL_BENCHMARK_EPISODES,
        "stratum_counts": counts,
        "sha256_canonical_digest": digest,
    }