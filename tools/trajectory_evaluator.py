"""
tools.trajectory_evaluator - Closed-loop integration & trajectory
verification for the HIRAM safety kernel (Task 2.3).

Wires tools.inference_engine.BayesianInferenceEngine into a discrete-time
control loop over sim.plant's 1D cart kinematics: at each timestep, sample
sensor evidence, call engine.step(evidence) to get a decision, apply that
decision's control law to advance the cart's physics, and check for
collision. This is the first place in the repo where the causal-DAG
inference engine actually drives a trajectory, rather than being exercised
in isolation against hand-picked evidence.

Design note on the sensor model (read this before assuming a Lidar/ToF
mismatch is a bug): sim.sensors.pipeline.SensorPipeline (Task 1.3) models a
physical ToF + Ultrasonic dual-range sensor and outputs boolean threshold
evidence (d_critical, d_marginal, v_high, v_med, sensor_disagree,
telemetry_stale) -- a *different* evidence vocabulary from the causal DAG's
tri-state LidarObs/TofObs/WheelSlipObs literals (Task 2.1), and it has no
"Lidar" channel at all. sim/runner.py (M0) consumes that boolean vocabulary
in an *open-loop* replay (the control law is baked into the scenario's
commanded_acceleration/braking_deceleration; nothing feeds back from
inference). Forcing SensorPipeline's two range channels onto LidarObs/TofObs
would be a channel-name relabeling with no principled correspondence, and it
has no time-varying "obstacle becomes real" hook, which the pop-up-obstacle
scenario needs.

Instead, this module generates synthetic evidence by sampling directly from
the *same* sealed CPTs the loaded BayesianInferenceEngine instance already
verified (via engine._nodes), conditioned on ground-truth TrueObstacle/
DecelCapability state that this module tracks per timestep (including a
configurable time at which a previously-undetectable obstacle becomes real).
This is standard practice for testing a Bayesian decision system: generate
observations from the assumed generative model, then verify the resulting
closed-loop behavior is sound. It still exercises real, non-trivial logic
independent of the CPT values themselves -- the discrete-time control loop,
the tri-state/dropout evidence contract, the decision -> control -> physics
-> collision pipeline, and integration with sim.plant and sim.scenarios --
none of which is simply "did the code sample its own distribution correctly".

sim.plant.PlantState is reused for initial conditions (integrating with
tests/test_plant_oracle.py's data contract); tests/test_trajectory_
verification.py separately cross-checks an emergency-braking episode's
achieved stopping position against sim.plant.analytic_plant's independent
closed-form solution. sim.scenarios.generator/StratumType supply realistic
physical parameters (obstacle_position, braking_deceleration, dropout
window) for NOMINAL/SHIFT test scenarios, integrating with Task 1.2's
scenario models.

Known simplification: ScenarioInstance.sensor_bias and .sensor_noise_scale
(used by sim.sensors.pipeline's continuous ToF/Ultrasonic model) are not
consumed here -- this module's evidence always samples from the causal
DAG's own fixed CPTs regardless of a scenario's declared noise/bias level,
since those CPTs have no bias/noise-scale parameters to perturb. SHIFT_2/
SHIFT_3/SHIFT_6's distinguishing sensor-noise characteristics are therefore
not actually reflected in the evidence this module generates for those
strata; only their physical parameters (braking_deceleration, dropout
window) take effect. Flagged here rather than silently assumed away.

Zero external dependencies: standard library only (math, random, dataclasses,
typing). No numpy/scipy/pgmpy.
"""

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sim.plant import PlantState
from sim.scenarios.strata import ScenarioInstance
from tools.inference_engine import BayesianInferenceEngine, EvidenceValue

# Braking-capability bands used to derive a "true DecelCapability" ground
# truth from a scenario's continuous braking_deceleration, matching the
# banding convention established by sim.scenarios.generator's nominal/shift
# strata (NOMINAL bands >= 0.4 m/s^2, the LOW_BRAKING/DEGRADED band
# [0.2, 0.4), and the degraded-brake shift strata below 0.2).
_DECEL_NOMINAL_FLOOR = 0.4
_DECEL_DEGRADED_FLOOR = 0.2


@dataclass(frozen=True)
class TrajectoryProfile:
    """
    Per-episode control-loop parameters (as opposed to the physical
    scenario parameters, which live on ScenarioInstance).

    evidence_period_s decouples how often perception/decision runs
    (engine.step()) from how often physics is integrated (dt): between
    evidence updates, the last decision is held (zero-order hold), matching
    how a real embedded control loop typically actuates faster than its
    perception stack refreshes. This also has a practical statistical
    effect worth being explicit about: engine.step() is a memoryless,
    single-shot decision with no temporal filtering (per the Task 2.2 audit
    freeze, nothing upstream of it may add hysteresis), and this model's own
    sensor CPTs have a small (~1-2%) false-detection rate per independent
    draw -- entirely rational for the model's loss matrix to react to (the
    cost of a missed real obstacle vastly exceeds the cost of one
    unnecessary brake), but it means resampling evidence independently
    every physics tick over a long episode makes "zero false alarms"
    statistically unlikely by construction, for any control law. Sampling
    perception at a slower, still-physically-reasonable rate keeps the
    number of independent trials -- and therefore this effect -- bounded.
    """

    dt: float = 0.02  # s, physics integration period
    evidence_period_s: Optional[float] = None  # s, perception/decision period; None = same as dt
    max_duration_s: float = 20.0  # s, safety cutoff so a runaway loop can't hang a test
    target_velocity: float = 0.35  # m/s, ACCEL's cruise target
    accel_rate: float = 0.15  # m/s^2, ACCEL's applied acceleration toward target_velocity
    coast_friction_decel: float = 0.05  # m/s^2, COAST's natural rolling-resistance deceleration
    obstacle_appears_at_s: Optional[float] = 0.0  # s; None = obstacle never becomes real/detectable this episode
    settle_steps: int = 15  # consecutive at-rest, non-ACCEL steps before ending the episode early
    clearance_margin_m: float = 0.10  # matches the benchmark's standard clearance convention


@dataclass(frozen=True)
class TimestepRecord:
    t: float
    x: float
    v: float
    a: float
    action: str
    evidence: Dict[str, EvidenceValue]
    fallback_active: bool
    true_obstacle_blocked: bool


@dataclass
class EpisodeResult:
    scenario: ScenarioInstance
    profile: TrajectoryProfile
    final_x: float
    final_v: float
    duration_s: float
    steps: int
    collided: bool
    settled: bool  # True if the cart came to rest and stayed there (vs. hit max_duration_s / collided)
    clearance_m: Optional[float]  # obstacle_position - final_x, or None if there's no obstacle
    action_history: List[str] = field(default_factory=list)
    action_change_count: int = 0
    emergency_brake_engaged: bool = False
    first_emergency_brake_t: Optional[float] = None
    false_emergency_stop_count: int = 0
    fallback_event_count: int = 0
    records: List[TimestepRecord] = field(default_factory=list)


def _true_decel_index(braking_deceleration: float) -> int:
    """Ground-truth DecelCapability state index (0=NOMINAL, 1=DEGRADED,
    2=CRITICAL) derived from a scenario's actual braking capability."""
    if braking_deceleration >= _DECEL_NOMINAL_FLOOR:
        return 0
    if braking_deceleration >= _DECEL_DEGRADED_FLOOR:
        return 1
    return 2


def _sample_categorical(cpt_row: List[float], rng: random.Random) -> int:
    """Sample a state index from a discrete distribution given as a CPT row
    (a list of probabilities summing to ~1.0)."""
    r = rng.random()
    cumulative = 0.0
    for i, p in enumerate(cpt_row):
        cumulative += p
        if r < cumulative:
            return i
    return len(cpt_row) - 1  # floating-point safety net for r landing exactly at/past 1.0


class TrajectoryEvaluator:
    """
    Drives a closed-loop rollout: sample evidence -> engine.step(evidence)
    -> apply the selected action's control law -> advance physics -> check
    collision -> repeat. One evaluator is built around one fixed
    (engine, scenario) pair; run_episode() may be called multiple times
    (e.g. with different initial states/profiles) against that same pair.
    """

    def __init__(
        self,
        engine: BayesianInferenceEngine,
        scenario: ScenarioInstance,
        seed: Optional[int] = None,
    ) -> None:
        self.engine = engine
        self.scenario = scenario
        self._rng = random.Random(seed if seed is not None else scenario.seed)

        # Read CPTs/strides directly off the engine's own (digest-verified)
        # loaded model, so the synthetic evidence this module generates is
        # never out of sync with what the engine assumes.
        nodes = engine._nodes  # noqa: SLF001 -- intentional, see module docstring
        self._lidar_cpt = nodes["LidarObs"]["cpt"]
        self._lidar_stride = nodes["LidarObs"]["parent_strides"]["TrueObstacle"]
        self._lidar_dropout_idx = nodes["LidarObs"].get("dropout_state_index")
        self._tof_cpt = nodes["TofObs"]["cpt"]
        self._tof_stride = nodes["TofObs"]["parent_strides"]["TrueObstacle"]
        self._tof_dropout_idx = nodes["TofObs"].get("dropout_state_index")
        self._wheel_cpt = nodes["WheelSlipObs"]["cpt"]
        self._wheel_stride = nodes["WheelSlipObs"]["parent_strides"]["DecelCapability"]
        self._wheel_dropout_idx = nodes["WheelSlipObs"].get("dropout_state_index")

        self._n_lidar_states = len(nodes["LidarObs"]["states"])
        self._n_tof_states = len(nodes["TofObs"]["states"])
        self._n_wheel_states = len(nodes["WheelSlipObs"]["states"])

    # ------------------------------------------------------------------
    # Evidence generation
    # ------------------------------------------------------------------
    def _obstacle_detectable(self, t: float, profile: TrajectoryProfile) -> bool:
        if self.scenario.obstacle_position is None:
            return False
        if profile.obstacle_appears_at_s is None:
            return False
        return t >= profile.obstacle_appears_at_s

    def _in_dropout_window(self, t: float) -> bool:
        return self.scenario.dropout_duration_s > 0.0 and (
            self.scenario.dropout_start_s <= t < self.scenario.dropout_start_s + self.scenario.dropout_duration_s
        )

    def _canonicalize(self, sensor_name: str, value: int, dropout_idx: Optional[int]) -> EvidenceValue:
        """Map a sampled DROPOUT-state index to None, so evidence dicts are
        uniform regardless of the engine's marginalize_dropout_state
        setting (which the audit freeze requires stay at its default True)."""
        if dropout_idx is not None and value == dropout_idx:
            return None
        return value

    def sample_evidence(self, t: float, profile: TrajectoryProfile) -> Dict[str, EvidenceValue]:
        """
        Draw one timestep's tri-state evidence dict. During a dropout
        window, every sensor reads as unobserved (None) -- a full
        telemetry-link outage, not just the range sensors. Otherwise each
        sensor is sampled from its own CPT row, conditioned on this
        timestep's ground-truth TrueObstacle/DecelCapability state.
        """
        if self._in_dropout_window(t):
            return {"LidarObs": None, "TofObs": None, "WheelSlipObs": None}

        true_obstacle_idx = 1 if self._obstacle_detectable(t, profile) else 0
        lidar_row = self._lidar_cpt[
            true_obstacle_idx * self._lidar_stride : true_obstacle_idx * self._lidar_stride + self._n_lidar_states
        ]
        tof_row = self._tof_cpt[
            true_obstacle_idx * self._tof_stride : true_obstacle_idx * self._tof_stride + self._n_tof_states
        ]

        true_decel_idx = _true_decel_index(self.scenario.braking_deceleration)
        wheel_row = self._wheel_cpt[
            true_decel_idx * self._wheel_stride : true_decel_idx * self._wheel_stride + self._n_wheel_states
        ]

        lidar_val = _sample_categorical(lidar_row, self._rng)
        tof_val = _sample_categorical(tof_row, self._rng)
        wheel_val = _sample_categorical(wheel_row, self._rng)

        return {
            "LidarObs": self._canonicalize("LidarObs", lidar_val, self._lidar_dropout_idx),
            "TofObs": self._canonicalize("TofObs", tof_val, self._tof_dropout_idx),
            "WheelSlipObs": self._canonicalize("WheelSlipObs", wheel_val, self._wheel_dropout_idx),
        }

    # ------------------------------------------------------------------
    # Control law
    # ------------------------------------------------------------------
    def _commanded_acceleration(self, action: str, v: float, profile: TrajectoryProfile) -> float:
        if action == "ACCEL":
            return profile.accel_rate if v < profile.target_velocity else 0.0
        if action == "COAST":
            return -profile.coast_friction_decel if v > 0.0 else 0.0
        if action == "EMERGENCY_BRAKE":
            return -self.scenario.braking_deceleration if v > 0.0 else 0.0
        raise ValueError(f"unknown action {action!r}")

    @staticmethod
    def _advance_physics(x: float, v: float, a: float, dt: float) -> Tuple[float, float]:
        v_next = max(0.0, v + a * dt)
        # Trapezoidal integration: exact for constant acceleration over the step.
        x_next = x + 0.5 * (v + v_next) * dt
        return x_next, v_next

    # ------------------------------------------------------------------
    # Episode rollout
    # ------------------------------------------------------------------
    def run_episode(
        self,
        initial_state: PlantState,
        trajectory_profile: TrajectoryProfile,
        evidence_override: Optional[Callable[[float], Dict[str, EvidenceValue]]] = None,
    ) -> EpisodeResult:
        """
        Roll out one closed-loop episode from `initial_state` under
        `trajectory_profile`.

        `evidence_override`, if given, replaces the generative CPT sampler
        with a caller-supplied function of the current time -- used by
        tests to inject specific/corrupted evidence sequences (e.g. to
        force the zero-likelihood fallback) without needing a coincidentally
        self-inconsistent CPT.
        """
        profile = trajectory_profile
        evidence_period = profile.evidence_period_s if profile.evidence_period_s is not None else profile.dt
        x, v = initial_state.position, initial_state.velocity
        t = 0.0

        records: List[TimestepRecord] = []
        action_history: List[str] = []
        false_emergency_stop_count = 0
        fallback_event_count = 0
        emergency_brake_engaged = False
        first_emergency_brake_t: Optional[float] = None
        collided = False
        settled = False
        consecutive_settled_steps = 0

        # Zero-order hold: the decision (and the evidence that produced it)
        # only updates at evidence_period boundaries; physics integrates
        # every dt in between using whatever decision is currently held.
        # A decision's bookkeeping (false-alarm/fallback counts, history) is
        # attributed once, at the moment it is *made*, not once per physics
        # tick it happens to be held over.
        current_evidence: Dict[str, EvidenceValue] = {}
        current_action = "COAST"  # safe default before the first perception update
        current_fallback = False
        next_evidence_t = 0.0
        has_moved = False  # guards "settled" against firing before the cart ever got going

        n_steps = max(1, math.ceil(profile.max_duration_s / profile.dt))
        for _ in range(n_steps):
            if t >= next_evidence_t - 1e-9:
                current_evidence = (
                    evidence_override(t) if evidence_override is not None else self.sample_evidence(t, profile)
                )
                result = self.engine.step(current_evidence)
                current_action = result["selected_action"]
                current_fallback = result["fallback_active"]
                next_evidence_t += evidence_period

                true_blocked = self._obstacle_detectable(t, profile)
                if current_action == "EMERGENCY_BRAKE":
                    if not emergency_brake_engaged:
                        emergency_brake_engaged = True
                        first_emergency_brake_t = t
                    if not true_blocked:
                        false_emergency_stop_count += 1
                if current_fallback:
                    fallback_event_count += 1
                action_history.append(current_action)

            a_cmd = self._commanded_acceleration(current_action, v, profile)
            t_next = t + profile.dt
            x, v = self._advance_physics(x, v, a_cmd, profile.dt)

            records.append(
                TimestepRecord(
                    t=t_next,
                    x=x,
                    v=v,
                    a=a_cmd,
                    action=current_action,
                    evidence=current_evidence,
                    fallback_active=current_fallback,
                    true_obstacle_blocked=self._obstacle_detectable(t_next, profile),
                )
            )
            t = t_next

            if self.scenario.obstacle_position is not None and x >= self.scenario.obstacle_position:
                collided = True
                break

            if v > 0.0:
                has_moved = True

            if has_moved and v <= 0.0 and current_action != "ACCEL":
                consecutive_settled_steps += 1
                if consecutive_settled_steps >= profile.settle_steps:
                    settled = True
                    break
            else:
                consecutive_settled_steps = 0

        action_change_count = sum(
            1 for prev, cur in zip(action_history, action_history[1:]) if prev != cur
        )

        clearance_m = None
        if self.scenario.obstacle_position is not None:
            clearance_m = self.scenario.obstacle_position - x

        return EpisodeResult(
            scenario=self.scenario,
            profile=profile,
            final_x=x,
            final_v=v,
            duration_s=t,
            steps=len(records),
            collided=collided,
            settled=settled,
            clearance_m=clearance_m,
            action_history=action_history,
            action_change_count=action_change_count,
            emergency_brake_engaged=emergency_brake_engaged,
            first_emergency_brake_t=first_emergency_brake_t,
            false_emergency_stop_count=false_emergency_stop_count,
            fallback_event_count=fallback_event_count,
            records=records,
        )
