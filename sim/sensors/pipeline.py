"""
Dual-sensor observation pipeline with unbiased cumulative encoder physics,
tri-state range literals, and literal observation dropouts.
"""

from dataclasses import dataclass
import math
import random
from typing import Optional

from sim.scenarios.strata import ScenarioInstance, derive_domain_seed

TICK_PITCH = 0.0005  # 0.5 mm per tick (high-resolution optical wheel encoder)
TOF_NOISE_SIGMA = 0.02
ULTRA_NOISE_SIGMA = 0.04
ENCODER_NOISE_SIGMA = 0.01
D_CRITICAL_THRESHOLD_M = 0.30
D_MARGINAL_THRESHOLD_M = 0.65
V_HIGH_THRESHOLD_MPS = 0.35
V_MED_THRESHOLD_MPS = 0.15
SENSOR_DISAGREE_THRESHOLD_M = 0.08
TELEMETRY_STALE_THRESHOLD_S = 0.05


@dataclass(frozen=True)
class SensorObservation:
    timestamp: float
    d_tof: Optional[float]
    d_ultra: Optional[float]
    v_measured: float
    d_critical: Optional[bool]     # None when unobserved/dropout
    d_marginal: Optional[bool]     # None when unobserved/dropout
    v_high: bool
    v_med: bool
    sensor_disagree: Optional[bool] # None when unobserved/dropout
    telemetry_stale: bool


class SensorPipeline:
    def __init__(self, master_seed: str, scenario: ScenarioInstance):
        self.scenario = scenario
        self.last_valid_timestamp: float = 0.0

        # Stateful encoder tracking:
        # True physical displacement is accumulated uncorrupted.
        # Optical detector phase jitter is applied directly at readout without clamping rectification.
        self.cumulative_true_pos_m: float = 0.0
        self.last_encoder_time: Optional[float] = None
        self.cumulative_ticks: int = 0

        tof_seed = derive_domain_seed(master_seed, scenario.stratum.name, scenario.episode_id, "tof")
        ultra_seed = derive_domain_seed(master_seed, scenario.stratum.name, scenario.episode_id, "ultra")
        enc_seed = derive_domain_seed(master_seed, scenario.stratum.name, scenario.episode_id, "encoder")

        self.rng_tof = random.Random(tof_seed)
        self.rng_ultra = random.Random(ultra_seed)
        self.rng_enc = random.Random(enc_seed)

    def sample(
        self, t: float, current_x: float, current_v: float, dt_sample: float = 0.01
    ) -> SensorObservation:
        """Evaluates sensor readings and discretization at timestamp t."""
        d_true = max(0.0, self.scenario.obstacle_position - current_x)

        # Check for literal telemetry dropout
        in_dropout = False
        if self.scenario.dropout_duration_s > 0.0:
            d_start = self.scenario.dropout_start_s
            d_end = d_start + self.scenario.dropout_duration_s
            if d_start <= t < d_end:
                in_dropout = True

        if in_dropout:
            d_tof = None
            d_ultra = None
            d_critical = None
            d_marginal = None
            sensor_disagree = None
            telemetry_stale = True
        else:
            scale = self.scenario.sensor_noise_scale
            noise_tof = self.rng_tof.gauss(0.0, TOF_NOISE_SIGMA * scale)
            noise_ultra = self.rng_ultra.gauss(0.0, ULTRA_NOISE_SIGMA * scale)

            raw_tof = d_true + self.scenario.sensor_bias + noise_tof
            raw_ultra = d_true + self.scenario.sensor_bias + noise_ultra

            d_tof = max(0.0, raw_tof)
            d_ultra = max(0.0, raw_ultra)
            self.last_valid_timestamp = t
            telemetry_stale = (t - self.last_valid_timestamp) > TELEMETRY_STALE_THRESHOLD_S

            d_min = min(d_tof, d_ultra)
            d_critical = d_min < D_CRITICAL_THRESHOLD_M
            d_marginal = D_CRITICAL_THRESHOLD_M <= d_min < D_MARGINAL_THRESHOLD_M
            sensor_disagree = abs(d_tof - d_ultra) > SENSOR_DISAGREE_THRESHOLD_M

        # Stateful incremental encoder physics (unbiased noise model)
        if self.last_encoder_time is None:
            dt_enc = 0.0
            self.last_encoder_time = t
        else:
            dt_enc = max(0.0, t - self.last_encoder_time)
            self.last_encoder_time = t

        # Accumulate exact continuous physical wheel motion (no rectification bias)
        if current_v > 0.0 and dt_enc > 0.0:
            self.cumulative_true_pos_m += current_v * dt_enc

        # Sensor phase jitter applies to optical edge detection without mutating accumulated distance
        optical_jitter = self.rng_enc.gauss(0.0, 0.00002) if current_v > 0.0 else 0.0
        effective_readout_pos = max(0.0, self.cumulative_true_pos_m + optical_jitter)

        current_ticks = math.floor(effective_readout_pos / TICK_PITCH)
        delta_ticks = max(0, current_ticks - self.cumulative_ticks)
        self.cumulative_ticks = current_ticks

        eff_dt = dt_enc if dt_enc > 0.0 else dt_sample
        v_measured = max(0.0, (delta_ticks * TICK_PITCH) / eff_dt)

        v_high = v_measured >= V_HIGH_THRESHOLD_MPS
        v_med = V_MED_THRESHOLD_MPS <= v_measured < V_HIGH_THRESHOLD_MPS

        return SensorObservation(
            timestamp=t,
            d_tof=d_tof,
            d_ultra=d_ultra,
            v_measured=v_measured,
            d_critical=d_critical,
            d_marginal=d_marginal,
            v_high=v_high,
            v_med=v_med,
            sensor_disagree=sensor_disagree,
            telemetry_stale=telemetry_stale,
        )