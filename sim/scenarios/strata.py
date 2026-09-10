"""
Strata definitions, scenario data contracts, and bit-exact cryptographic seed derivation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import hashlib
from typing import Optional


class StratumType(Enum):
    NOMINAL_1_STEADY = auto()
    NOMINAL_2_BOUNDARY_NOISE = auto()
    NOMINAL_3_SHARED_BIAS = auto()
    NOMINAL_4_LOW_BRAKING = auto()
    NOMINAL_5_SENSOR_DELAY = auto()
    NOMINAL_6_ADVERSE_COMPOUND = auto()
    SHIFT_1_DEGRADED_BRAKE = auto()
    SHIFT_2_DOUBLE_NOISE = auto()
    SHIFT_3_COMMON_BIAS_EXTREME = auto()
    SHIFT_4_OBS_DROPOUT_500MS = auto()
    SHIFT_5_SURFACE_DISTRIBUTION = auto()
    SHIFT_6_SIMULTANEOUS_FAULTS = auto()


@dataclass(frozen=True)
class ScenarioInstance:
    episode_id: int
    stratum: StratumType
    seed: int
    x0: float
    v0: float
    commanded_acceleration: float
    obstacle_position: float
    braking_deceleration: float
    actuator_delay: float
    sensor_bias: float
    sensor_noise_scale: float
    dropout_start_s: float
    dropout_duration_s: float
    is_recoverable: bool

    def canonical_repr(self) -> str:
        """
        Bit-exact canonical serialization using IEEE 754 float hexadecimal format.
        Guarantees single-bit tamper detection without decimal quantization leaks.
        """
        return (
            f"{self.episode_id}|{self.stratum.name}|{self.seed}|"
            f"{self.x0.hex()}|{self.v0.hex()}|{self.commanded_acceleration.hex()}|"
            f"{self.obstacle_position.hex()}|{self.braking_deceleration.hex()}|"
            f"{self.actuator_delay.hex()}|{self.sensor_bias.hex()}|"
            f"{self.sensor_noise_scale.hex()}|{self.dropout_start_s.hex()}|"
            f"{self.dropout_duration_s.hex()}|{int(self.is_recoverable)}"
        )


def derive_domain_seed(master_seed: str, stratum_name: str, episode_id: int, domain: str) -> int:
    """
    Derives an isolated 128-bit integer seed for a specific domain/channel.
    Prevents cross-channel correlation and eliminates birthday-paradox collisions.
    """
    raw_key = f"{master_seed}:{stratum_name}:{episode_id}:{domain}".encode("utf-8")
    digest = hashlib.sha256(raw_key).digest()
    return int.from_bytes(digest[:16], byteorder="big")