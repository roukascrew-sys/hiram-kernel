"""
sim.scenarios.strata - Type contracts for the stopping-benchmark scenario generator.

Defines the twelve evaluation strata (six nominal, six distribution-shift)
and the immutable per-episode parameter record sim.scenarios.generator
produces for each one. Contains only data contracts and the fixed episode
counts the brief mandates -- no sampling logic, no randomness, no dependency
on sim.plant (that lives in generator.py, which uses sim.plant.analytic_plant
read-only, never modifying it).
"""

from dataclasses import dataclass
from enum import Enum, auto


class StratumType(Enum):
    # Nominal strata: parameters stay within this benchmark's declared
    # physical envelope (see generator.py's sampling ranges). Every episode
    # in these six strata is guaranteed physically recoverable at t=0.
    NOMINAL_1_STEADY = auto()
    NOMINAL_2_BOUNDARY_NOISE = auto()
    NOMINAL_3_SHARED_BIAS = auto()
    NOMINAL_4_LOW_BRAKING = auto()
    NOMINAL_5_SENSOR_DELAY = auto()
    NOMINAL_6_ADVERSE_COMPOUND = auto()

    # Shift strata: parameters are deliberately drawn from outside the
    # nominal envelope (or from a differently-shaped distribution) to probe
    # generalization and graceful degradation. NOT filtered for
    # recoverability -- an unrecoverable shift episode is valid ground truth.
    SHIFT_1_DEGRADED_BRAKE = auto()
    SHIFT_2_DOUBLE_NOISE = auto()
    SHIFT_3_COMMON_BIAS_EXTREME = auto()
    SHIFT_4_OBS_DROPOUT_500MS = auto()
    SHIFT_5_SURFACE_DISTRIBUTION = auto()
    SHIFT_6_SIMULTANEOUS_FAULTS = auto()


NOMINAL_STRATA = (
    StratumType.NOMINAL_1_STEADY,
    StratumType.NOMINAL_2_BOUNDARY_NOISE,
    StratumType.NOMINAL_3_SHARED_BIAS,
    StratumType.NOMINAL_4_LOW_BRAKING,
    StratumType.NOMINAL_5_SENSOR_DELAY,
    StratumType.NOMINAL_6_ADVERSE_COMPOUND,
)

SHIFT_STRATA = (
    StratumType.SHIFT_1_DEGRADED_BRAKE,
    StratumType.SHIFT_2_DOUBLE_NOISE,
    StratumType.SHIFT_3_COMMON_BIAS_EXTREME,
    StratumType.SHIFT_4_OBS_DROPOUT_500MS,
    StratumType.SHIFT_5_SURFACE_DISTRIBUTION,
    StratumType.SHIFT_6_SIMULTANEOUS_FAULTS,
)

ALL_STRATA = NOMINAL_STRATA + SHIFT_STRATA

EPISODES_PER_NOMINAL_STRATUM = 60_000
EPISODES_PER_SHIFT_STRATUM = 10_000

TOTAL_NOMINAL_EPISODES = EPISODES_PER_NOMINAL_STRATUM * len(NOMINAL_STRATA)  # 360,000
TOTAL_SHIFT_EPISODES = EPISODES_PER_SHIFT_STRATUM * len(SHIFT_STRATA)  # 60,000
TOTAL_EPISODES = TOTAL_NOMINAL_EPISODES + TOTAL_SHIFT_EPISODES  # 420,000


def episodes_for(stratum: StratumType) -> int:
    """Episode count mandated for a given stratum: 60,000 for any nominal
    stratum, 10,000 for any shift stratum."""
    return EPISODES_PER_NOMINAL_STRATUM if stratum in NOMINAL_STRATA else EPISODES_PER_SHIFT_STRATUM


@dataclass(frozen=True)
class ScenarioInstance:
    """One fully-specified, reproducible episode's physical parameters."""

    episode_id: int  # Index within its stratum, 0-based
    stratum: StratumType
    seed: int  # Per-episode PRNG seed, cryptographically salted (see generator.derive_seed)
    x0: float  # Initial position (0.0 m)
    v0: float  # Initial velocity (m/s)
    obstacle_position: float  # Obstacle coordinate (m), true physical value
    braking_deceleration: float  # b (m/s^2), true physical value
    actuator_delay: float  # tau (s)
    sensor_bias: float  # shared range bias (m), applied by the observation model, not the physics
    sensor_noise_scale: float  # multiplier on sensor sigma
    is_recoverable: bool  # Verified by sim.plant.analytic_plant: can braking at t=0 stop short of obstacle_position?
