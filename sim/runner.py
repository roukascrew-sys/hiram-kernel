"""
sim/runner.py
Closed-loop episode runner and tri-state Bayesian evidence consumer for Milestone M0.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

from sim.plant import PlantParameters, PlantState
from sim.scenarios.strata import ScenarioInstance
from sim.sensors.pipeline import SensorObservation, SensorPipeline


@dataclass(frozen=True)
class EvidenceVector:
    timestamp: float
    # Range literals are Optional[bool]: True, False, or None (Masked / Unobserved)
    d_critical: Optional[bool]
    d_marginal: Optional[bool]
    v_high: bool
    v_med: bool
    sensor_disagree: Optional[bool]
    telemetry_stale: bool


class BayesianEvidenceConsumer:
    """
    Evidence consumer simulating the front-end of the HIRAM Bayesian safety kernel.
    Preserves None (unobserved) rather than treating it as negative evidence.
    """

    def __init__(self):
        self.received_frames: List[EvidenceVector] = []
        self.masked_frames_count: int = 0

    def ingest(self, obs: SensorObservation) -> EvidenceVector:
        ev = EvidenceVector(
            timestamp=obs.timestamp,
            d_critical=obs.d_critical,
            d_marginal=obs.d_marginal,
            v_high=obs.v_high,
            v_med=obs.v_med,
            sensor_disagree=obs.sensor_disagree,
            telemetry_stale=obs.telemetry_stale,
        )
        if obs.d_critical is None:
            self.masked_frames_count += 1
        self.received_frames.append(ev)
        return ev


@dataclass
class EpisodeRunResult:
    scenario: ScenarioInstance
    total_duration_s: float
    total_steps: int
    terminated_early: bool
    termination_reason: str
    dropout_steps_executed: int
    evidence_frames: List[EvidenceVector]


def compute_kinematic_state(
    x0: float, v0: float, a_cmd: float, tau: float, b: float, t: float
) -> PlantState:
    """Exact closed-form 1D cart kinematics under actuator delay and emergency braking."""
    if t <= tau:
        x = x0 + v0 * t + 0.5 * a_cmd * (t ** 2)
        v = max(0.0, v0 + a_cmd * t)
        a = a_cmd
    else:
        # State at end of delay interval
        x_tau = x0 + v0 * tau + 0.5 * a_cmd * (tau ** 2)
        v_tau = max(0.0, v0 + a_cmd * tau)

        t_brake = t - tau
        if v_tau <= 0.0:
            x = x_tau
            v = 0.0
            a = 0.0
        else:
            time_to_stop = v_tau / b
            if t_brake >= time_to_stop:
                x = x_tau + (v_tau ** 2) / (2.0 * b)
                v = 0.0
                a = 0.0
            else:
                x = x_tau + v_tau * t_brake - 0.5 * b * (t_brake ** 2)
                v = max(0.0, v_tau - b * t_brake)
                a = -b

    return PlantState(position=x, velocity=v, acceleration=a)


def run_episode(
    scenario: ScenarioInstance,
    master_seed: str,
    dt: float = 0.01,
    max_duration_s: float = 2.0,
) -> EpisodeRunResult:
    """
    Executes a scenario episode at 100 Hz (dt=0.01s).
    Guarantees that dropout episodes (SHIFT_4 and SHIFT_6) continue sampling
    through the complete 500 ms dropout window, even if the cart comes to a stop.
    """
    pipeline = SensorPipeline(master_seed, scenario)
    consumer = BayesianEvidenceConsumer()

    t = 0.0
    dropout_steps = 0
    terminated = False
    term_reason = "max_duration_reached"

    # Ensure evaluation captures full 500 ms dropout even if cart stops early
    min_horizon = (
        (scenario.dropout_start_s + scenario.dropout_duration_s + 0.05)
        if scenario.dropout_duration_s > 0.0
        else 0.50
    )

    while t <= max_duration_s:
        current_state = compute_kinematic_state(
            x0=scenario.x0,
            v0=scenario.v0,
            a_cmd=scenario.commanded_acceleration,
            tau=scenario.actuator_delay,
            b=scenario.braking_deceleration,
            t=t,
        )

        obs = pipeline.sample(
            t=t,
            current_x=current_state.position,
            current_v=current_state.velocity,
            dt_sample=dt,
        )
        consumer.ingest(obs)

        if obs.d_critical is None:
            dropout_steps += 1

        is_stopped = (current_state.velocity <= 1e-4) and (t > scenario.actuator_delay)
        is_collision = current_state.position >= scenario.obstacle_position

        if is_collision:
            terminated = True
            term_reason = "collision"
            break

        if is_stopped and (t >= min_horizon):
            terminated = True
            term_reason = "cart_stopped_and_dropout_complete"
            break

        t = round(t + dt, 6)

    return EpisodeRunResult(
        scenario=scenario,
        total_duration_s=round(t, 4),
        total_steps=len(consumer.received_frames),
        terminated_early=terminated and term_reason != "max_duration_reached",
        termination_reason=term_reason,
        dropout_steps_executed=dropout_steps,
        evidence_frames=consumer.received_frames,
    )