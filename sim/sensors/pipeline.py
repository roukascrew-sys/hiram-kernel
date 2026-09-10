"""
sim.sensors.pipeline - Synthetic dual-sensor pipeline for the stopping benchmark.

Models two independent range sensors (a Time-of-Flight sensor and an
ultrasonic sensor) plus a wheel encoder, all reading the same underlying
ground-truth cart state, and a discretization engine that turns their
continuous readings into the boolean evidence variables a downstream
hazard-evaluation DAG would consume.

Physical model
--------------
Given ground-truth range d_true = x_obs - x_cart and ground-truth velocity
v_true (the caller's responsibility to compute -- this module has no
dependency on sim.plant, it takes d_true/v_true as plain floats):

    d_ToF   = max(0.0, d_true + b_shared + N(0, sigma=0.02 * noise_scale))
    d_ultra = max(0.0, d_true + b_shared + N(0, sigma=0.04 * noise_scale))
    v_meas  = max(0.0, v_true + N(0, sigma=0.01))

`b_shared` (sensor_bias) and `noise_scale` (sensor_noise_scale) are fixed per
SensorPipeline instance -- they represent one episode's constant sensor
characteristics, per sim.scenarios.strata.ScenarioInstance. d_true, v_true,
and delta_t_sample vary per call, once per simulation timestep.

Determinism: one SensorPipeline owns exactly one random.Random stream,
constructed once from an integer seed; every sample() call draws from it in
a fixed order (ToF noise, then ultrasonic noise, then encoder noise), so
replaying the same seed against the same sequence of (d_true, v_true) calls
reproduces bit-identical readings. __slots__ keeps a pipeline's own
per-instance footprint to its three fields (no per-instance __dict__), and
sample()/discretize() allocate nothing beyond the one SensorReading /
DiscreteEvidence they return -- no lists, buffers, or intermediate
collections -- so a batch run's steady-state memory use does not grow with
the number of samples taken.
"""

import hashlib
import random
from dataclasses import dataclass
from typing import Tuple

TOF_NOISE_SIGMA_BASE = 0.02  # m, at sensor_noise_scale = 1.0
ULTRASONIC_NOISE_SIGMA_BASE = 0.04  # m, at sensor_noise_scale = 1.0
ENCODER_NOISE_SIGMA = 0.01  # m/s, fixed (not scaled by sensor_noise_scale)

D_CRITICAL_THRESHOLD_M = 0.30
D_MARGINAL_THRESHOLD_M = 0.65
V_HIGH_THRESHOLD_MPS = 0.35
V_MED_THRESHOLD_MPS = 0.15
SENSOR_DISAGREE_THRESHOLD_M = 0.08
TELEMETRY_STALE_THRESHOLD_S = 0.05


@dataclass(frozen=True)
class SensorReading:
    """One timestep's continuous sensor outputs."""

    d_tof: float  # m, ToF range reading (clamped >= 0.0)
    d_ultra: float  # m, ultrasonic range reading (clamped >= 0.0)
    v_meas: float  # m/s, wheel-encoder velocity reading (clamped >= 0.0)
    delta_t_sample: float  # s, time elapsed since the previous sample


@dataclass(frozen=True)
class DiscreteEvidence:
    """Boolean DAG evidence derived from one SensorReading."""

    d_critical: bool  # min(d_tof, d_ultra) < 0.30 m
    d_marginal: bool  # 0.30 m <= min(d_tof, d_ultra) < 0.65 m
    v_high: bool  # v_meas >= 0.35 m/s
    v_med: bool  # 0.15 m/s <= v_meas < 0.35 m/s
    sensor_disagree: bool  # |d_tof - d_ultra| > 0.08 m
    telemetry_stale: bool  # delta_t_sample > 0.05 s (50 ms)


def derive_sensor_seed(episode_seed: int) -> int:
    """
    Derive a sensor-stream seed from an episode's own seed, independent of
    the RNG stream sim.scenarios.generator used to draw that episode's
    physical parameters -- same SHA-256 salting convention as
    sim.scenarios.generator.derive_seed, with a ':sensors' suffix so the two
    streams never coincide even when starting from the same episode_seed.
    """
    key = f"{episode_seed}:sensors"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


class SensorPipeline:
    """
    One episode's dual-sensor + wheel-encoder pipeline: fixed sensor_bias and
    sensor_noise_scale (constant for the episode), a private deterministic
    RNG stream, and per-timestep sample() / discretize() / read() calls.
    """

    __slots__ = ("_rng", "sensor_bias", "sensor_noise_scale")

    def __init__(self, seed: int, sensor_bias: float, sensor_noise_scale: float) -> None:
        self._rng = random.Random(seed)
        self.sensor_bias = sensor_bias
        self.sensor_noise_scale = sensor_noise_scale

    @classmethod
    def from_scenario(cls, instance, seed: int = None) -> "SensorPipeline":
        """
        Convenience constructor: sensor_bias/sensor_noise_scale are read off
        a sim.scenarios.strata.ScenarioInstance (duck-typed -- any object
        with those two attributes works, so this module still does not need
        to import sim.scenarios). The RNG seed defaults to
        derive_sensor_seed(instance.seed) unless explicitly overridden.
        """
        if seed is None:
            seed = derive_sensor_seed(instance.seed)
        return cls(seed=seed, sensor_bias=instance.sensor_bias, sensor_noise_scale=instance.sensor_noise_scale)

    def sample(self, d_true: float, v_true: float, delta_t_sample: float = 0.0) -> SensorReading:
        """Draw one timestep's noisy sensor readings from ground truth."""
        d_tof = d_true + self.sensor_bias + self._rng.gauss(0.0, TOF_NOISE_SIGMA_BASE * self.sensor_noise_scale)
        d_ultra = d_true + self.sensor_bias + self._rng.gauss(
            0.0, ULTRASONIC_NOISE_SIGMA_BASE * self.sensor_noise_scale
        )
        v_meas = v_true + self._rng.gauss(0.0, ENCODER_NOISE_SIGMA)

        return SensorReading(
            d_tof=max(0.0, d_tof),
            d_ultra=max(0.0, d_ultra),
            v_meas=max(0.0, v_meas),
            delta_t_sample=delta_t_sample,
        )

    @staticmethod
    def discretize(reading: SensorReading) -> DiscreteEvidence:
        """Map one SensorReading's continuous values to boolean DAG evidence."""
        d_min = min(reading.d_tof, reading.d_ultra)
        return DiscreteEvidence(
            d_critical=d_min < D_CRITICAL_THRESHOLD_M,
            d_marginal=D_CRITICAL_THRESHOLD_M <= d_min < D_MARGINAL_THRESHOLD_M,
            v_high=reading.v_meas >= V_HIGH_THRESHOLD_MPS,
            v_med=V_MED_THRESHOLD_MPS <= reading.v_meas < V_HIGH_THRESHOLD_MPS,
            sensor_disagree=abs(reading.d_tof - reading.d_ultra) > SENSOR_DISAGREE_THRESHOLD_M,
            telemetry_stale=reading.delta_t_sample > TELEMETRY_STALE_THRESHOLD_S,
        )

    def read(self, d_true: float, v_true: float, delta_t_sample: float = 0.0) -> Tuple[SensorReading, DiscreteEvidence]:
        """Sample and discretize in one call."""
        reading = self.sample(d_true, v_true, delta_t_sample)
        return reading, self.discretize(reading)
