"""
sim.sensors - Synthetic dual-sensor pipeline for the stopping benchmark.

sim.sensors.pipeline   ToF + ultrasonic range sensors and a wheel-encoder
                       velocity sensor, reading a shared ground-truth cart
                       state, plus the discretization engine that maps their
                       continuous output to boolean DAG evidence.
"""

from .pipeline import (
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

__all__ = [
    "SensorPipeline",
    "SensorReading",
    "DiscreteEvidence",
    "derive_sensor_seed",
    "TOF_NOISE_SIGMA_BASE",
    "ULTRASONIC_NOISE_SIGMA_BASE",
    "ENCODER_NOISE_SIGMA",
    "D_CRITICAL_THRESHOLD_M",
    "D_MARGINAL_THRESHOLD_M",
    "V_HIGH_THRESHOLD_MPS",
    "V_MED_THRESHOLD_MPS",
    "SENSOR_DISAGREE_THRESHOLD_M",
    "TELEMETRY_STALE_THRESHOLD_S",
]
