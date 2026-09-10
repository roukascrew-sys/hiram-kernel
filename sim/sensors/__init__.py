"""
Synthetic dual-sensor pipeline and boolean DAG evidence discretization.
"""

from .pipeline import (
    TICK_PITCH,
    TOF_NOISE_SIGMA,
    ULTRA_NOISE_SIGMA,
    ENCODER_NOISE_SIGMA,
    D_CRITICAL_THRESHOLD_M,
    D_MARGINAL_THRESHOLD_M,
    V_HIGH_THRESHOLD_MPS,
    V_MED_THRESHOLD_MPS,
    SENSOR_DISAGREE_THRESHOLD_M,
    TELEMETRY_STALE_THRESHOLD_S,
    SensorObservation,
    SensorPipeline,
)

# Backwards compatibility aliases
TOF_NOISE_SIGMA_BASE = TOF_NOISE_SIGMA
ULTRA_NOISE_SIGMA_BASE = ULTRA_NOISE_SIGMA

__all__ = [
    "TICK_PITCH",
    "TOF_NOISE_SIGMA",
    "TOF_NOISE_SIGMA_BASE",
    "ULTRA_NOISE_SIGMA",
    "ULTRA_NOISE_SIGMA_BASE",
    "ENCODER_NOISE_SIGMA",
    "D_CRITICAL_THRESHOLD_M",
    "D_MARGINAL_THRESHOLD_M",
    "V_HIGH_THRESHOLD_MPS",
    "V_MED_THRESHOLD_MPS",
    "SENSOR_DISAGREE_THRESHOLD_M",
    "TELEMETRY_STALE_THRESHOLD_S",
    "SensorObservation",
    "SensorPipeline",
]