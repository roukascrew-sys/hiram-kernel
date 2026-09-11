"""
tools.trajectory_evaluator - Closed-loop integration & trajectory
verification for the HIRAM safety kernel (Task 2.3, hardened per the Astra
audit's HOLD findings on commit 140b8a03).

Wires tools.inference_engine.BayesianInferenceEngine into a discrete-time
control loop over sim.plant-compatible 1D cart kinematics: at each timestep,
extract sensor evidence from the cart's exact physical state, call
engine.step(evidence) to get a decision, apply that decision's control law
to advance the cart's physics with exact analytical partial-step
integration, and check for collision.

Physics (exact analytical partial-step integration)
----------------------------------------------------
_advance_physics(x, v, a, dt) never lets a single fixed-dt step silently
overshoot a physical boundary the way naive Euler integration with a
post-hoc clamp would:

  * Decelerating (a < 0) and v would cross zero within this step: the cart
    comes to rest partway through the step, not at its end. The exact
    time-to-stop and distance covered up to that instant are computed in
    closed form (dt_stop = -v/a, dx = -v^2/(2a)), and velocity is reported
    as exactly 0.0, not a small negative residual clamped away.
  * Accelerating (a > 0) and v would exceed v_cruise within this step: the
    cart accelerates only until it reaches v_cruise, then coasts at exactly
    v_cruise for the remainder of the step (dt_acc = (v_cruise - v)/a, dx =
    the accelerating segment's exact displacement plus v_cruise times the
    remaining time).
  * a == 0: trivial constant-velocity displacement.

This is the same principle sim.plant.numerical_plant applies via bisection
root-finding (Task 1.1) -- a fixed-step integrator must not be allowed to
silently integrate across a kink in the dynamics -- specialized here to a
closed form, since both kinks (v=0, v=v_cruise) have exact algebraic
solutions for constant acceleration.

Evidence generation (deterministic continuous discretization)
----------------------------------------------------------------
extract_evidence() no longer samples noisily from the causal DAG's CPTs (the
previous version of this module did, and that is intentionally discarded
here): it deterministically discretizes the cart's actual physical state --
distance-to-obstacle against a fixed detection range, and a physically-
motivated wheel-slip-ratio proxy against a fixed slip threshold -- with no
randomness at all. A sensor dropout window still forces the two range
channels (LidarObs, TofObs) to report unobserved (None); by design (per
this specification) WheelSlipObs is not affected by a range-sensor dropout,
modeling it as a separate, independent telemetry channel (e.g. a wheel
encoder / IMU) that a range-sensor link outage does not take down.

sim.plant.PlantState is reused for initial conditions (integrating with
tests/test_plant_oracle.py's data contract); tests/test_trajectory_
verification.py separately cross-checks an emergency-braking episode's
achieved stopping position against sim.plant.analytic_plant's independent
closed-form solution. sim.scenarios.generator/StratumType supply realistic
physical parameters (obstacle_position, braking_deceleration, dropout
window) for NOMINAL/SHIFT test scenarios, integrating with Task 1.2's
scenario models.

Known simplification: ScenarioInstance has no dedicated "TrackCondition" or
"DecelCapability" ground-truth field (Task 1.2's schema only carries the
continuous braking_deceleration), so both labels used to drive
compute_wheel_slip are derived from braking_deceleration via the same
banding convention sim.scenarios.generator uses for its strata (NOMINAL
>= 0.4 m/s^2, DEGRADED [0.2, 0.4), CRITICAL < 0.2), with track condition
correlated 1:1 to that band. A real deployment would have independent
sensors/estimators for these; this evaluator does not attempt to invent an
independent track-condition signal where the scenario schema has none.

Zero external dependencies: standard library only (math, dataclasses,
typing). No numpy/scipy/pgmpy, and (per this hardening pass) no `random`:
evidence generation is now fully deterministic.
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sim.plant import PlantState
from sim.scenarios.strata import ScenarioInstance
from tools.inference_engine import BayesianInferenceEngine, EvidenceValue

# Braking-capability bands used to derive ground-truth DecelCapability (and,
# by 1:1 correlation, TrackCondition) labels from a scenario's continuous
# braking_deceleration, matching sim.scenarios.generator's strata banding
# (NOMINAL bands >= 0.4 m/s^2, the LOW_BRAKING/DEGRADED band [0.2, 0.4), and
# the degraded-brake shift strata below 0.2).
_DECEL_NOMINAL_FLOOR = 0.4
_DECEL_DEGRADED_FLOOR = 0.2
_DECEL_LABELS = ("NOMINAL", "DEGRADED", "CRITICAL")
_TRACK_LABELS = ("DRY", "WET", "ICY")

# compute_wheel_slip's heuristic parameters (see its docstring): this
# specification names the function and its 0.15 decision threshold, but not
# its internal formula, so these constants are this module's own documented,
# physically-motivated choice, not a value taken from elsewhere in the repo.
_BASE_SLIP_BY_DECEL_CAPABILITY = {"NOMINAL": 0.05, "DEGRADED": 0.20, "CRITICAL": 0.45}
_TRACK_SLIP_MULTIPLIER = {"DRY": 1.0, "WET": 1.3, "ICY": 1.8}
_SLIP_REFERENCE_DECEL_MPS2 = 1.0  # normalizes "how hard are we braking" for the heuristic


def _true_decel_index(braking_deceleration: float) -> int:
    """Ground-truth DecelCapability band index (0=NOMINAL, 1=DEGRADED,
    2=CRITICAL) derived from a scenario's actual braking capability."""
    if braking_deceleration >= _DECEL_NOMINAL_FLOOR:
        return 0
    if braking_deceleration >= _DECEL_DEGRADED_FLOOR:
        return 1
    return 2


@dataclass(frozen=True)
class TrajectoryProfile:
    """
    Per-episode control-loop parameters (as opposed to the physical
    scenario parameters, which live on ScenarioInstance). Field names and
    default values match this hardening pass's exact submitted
    specification.

    evidence_period_s decouples how often perception/decision runs
    (engine.step()) from how often physics is integrated (dt): between
    evidence updates, the last decision is held (zero-order hold), matching
    how a real embedded control loop typically actuates faster than its
    perception stack refreshes. None means "same as dt" (perception every
    physics tick).
    """

    dt: float = 0.010  # s, physics integration period (10 ms)
    evidence_period_s: Optional[float] = None  # s, perception/decision period; None = same as dt
    max_duration_s: float = 20.0  # s, safety cutoff so a runaway loop can't hang a test
    v_cruise: float = 0.50  # m/s, ACCEL's speed ceiling
    a_drive: float = 0.25  # m/s^2, ACCEL's commanded (signed) acceleration
    a_drag: float = -0.05  # m/s^2, COAST's commanded (signed) deceleration from rolling resistance
    obstacle_appears_at_s: Optional[float] = 0.0  # s; None = obstacle never becomes real/detectable this episode
    settle_steps: int = 15  # consecutive at-rest, non-ACCEL steps before ending the episode early
    clearance_margin_m: float = 0.10  # matches the benchmark's standard clearance convention
    detection_range_m: float = 0.300  # m, Lidar/ToF detection threshold
    slip_threshold: float = 0.15  # wheel-slip-ratio threshold for HIGH(1)


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


class TrajectoryEvaluator:
    """
    Drives a closed-loop rollout: extract evidence from exact physical
    state -> engine.step(evidence) -> apply the selected action's control
    law -> advance physics exactly -> check collision -> repeat. One
    evaluator is built around one fixed (engine, scenario) pair;
    run_episode() may be called multiple times (e.g. with different initial
    states/profiles) against that same pair.
    """

    def __init__(self, engine: BayesianInferenceEngine, scenario: ScenarioInstance) -> None:
        self.engine = engine
        self.scenario = scenario
        self.v_cruise = 0.50  # overwritten by run_episode() from the active profile; usable standalone too
        self._current_a = 0.0  # acceleration currently being applied, as read by extract_evidence()

    # ------------------------------------------------------------------
    # Ground truth
    # ------------------------------------------------------------------
    def _ground_truth_labels(self) -> Tuple[str, str]:
        """(track_condition, decel_capability) labels derived from the
        scenario's actual braking capability -- see module docstring."""
        idx = _true_decel_index(self.scenario.braking_deceleration)
        return _TRACK_LABELS[idx], _DECEL_LABELS[idx]

    def _obstacle_detectable(self, t: float, profile: TrajectoryProfile) -> bool:
        if self.scenario.obstacle_position is None:
            return False
        if profile.obstacle_appears_at_s is None:
            return False
        return t >= profile.obstacle_appears_at_s

    def _dropout_window(self) -> Optional[Tuple[float, float]]:
        if self.scenario.dropout_duration_s <= 0.0:
            return None
        return (self.scenario.dropout_start_s, self.scenario.dropout_start_s + self.scenario.dropout_duration_s)

    # ------------------------------------------------------------------
    # Evidence generation (deterministic continuous discretization)
    # ------------------------------------------------------------------
    def compute_wheel_slip(self, v: float, a: float, track_condition: str, decel_capability: str) -> float:
        """
        A physically-motivated proxy for a wheel-slip-ratio sensor: slip
        increases with how hard the cart is decelerating and with how poor
        its actual traction is (worse decel_capability, worse
        track_condition). Not braking (a >= 0) or already stationary
        (v <= 0) implies no slip. This heuristic's exact shape is this
        module's own documented choice (see module docstring); only its
        0.15 decision threshold and its (v, a, track_condition,
        decel_capability) inputs are part of the specification.
        """
        if a >= 0.0 or v <= 0.0:
            return 0.0
        base = _BASE_SLIP_BY_DECEL_CAPABILITY.get(decel_capability, _BASE_SLIP_BY_DECEL_CAPABILITY["NOMINAL"])
        track_mult = _TRACK_SLIP_MULTIPLIER.get(track_condition, 1.0)
        intensity = min(1.0, abs(a) / _SLIP_REFERENCE_DECEL_MPS2)
        return base * track_mult * (0.5 + 0.5 * intensity)

    def extract_evidence(
        self,
        x: float,
        v: float,
        obstacle_x: Optional[float],
        track_condition: str,
        decel_capability: str,
        t: float,
        dropout_window: Optional[Tuple[float, float]],
        detection_range_m: float = 0.300,
        slip_threshold: float = 0.15,
    ) -> Dict[str, EvidenceValue]:
        """
        Deterministically discretize the cart's exact physical state into
        tri-state evidence -- no random sampling. `obstacle_x` is None when
        there is no obstacle, or it exists but has not yet become
        detectable this episode (see TrajectoryProfile.obstacle_appears_at_s
        for the pop-up-obstacle mechanism).

        The acceleration used for the wheel-slip computation is read from
        self._current_a (the acceleration the cart is *currently* being
        subjected to, from the previous control decision) rather than a
        function parameter: perception measures the cart's present
        dynamics before a new decision changes them.
        """
        d_clear = obstacle_x - x if obstacle_x is not None else float("inf")

        if dropout_window is not None and dropout_window[0] <= t <= dropout_window[1]:
            lidar_obs: EvidenceValue = None
            tof_obs: EvidenceValue = None
        else:
            lidar_obs = 1 if d_clear <= detection_range_m else 0
            tof_obs = 1 if d_clear <= detection_range_m else 0

        slip_ratio = self.compute_wheel_slip(v, self._current_a, track_condition, decel_capability)
        wheel_slip_obs: EvidenceValue = 1 if slip_ratio >= slip_threshold else 0

        return {"LidarObs": lidar_obs, "TofObs": tof_obs, "WheelSlipObs": wheel_slip_obs}

    # ------------------------------------------------------------------
    # Control law
    # ------------------------------------------------------------------
    def _commanded_acceleration(self, action: str, profile: TrajectoryProfile) -> float:
        """
        The constant (signed) acceleration commanded for `action`. No
        v-dependent branching is needed here: _advance_physics's exact
        partial-step integration is what enforces v_cruise / v=0 boundaries,
        so the same constant command is correct whether or not this step
        actually reaches one.
        """
        if action == "ACCEL":
            return profile.a_drive
        if action == "COAST":
            return profile.a_drag
        if action == "EMERGENCY_BRAKE":
            return -self.scenario.braking_deceleration
        raise ValueError(f"unknown action {action!r}")

    def _advance_physics(self, x: float, v: float, a: float, dt: float) -> Tuple[float, float]:
        """
        Exact analytical partial-step integration: neither a deceleration
        past v=0 nor an acceleration past v_cruise is allowed to silently
        overshoot within a single dt -- see module docstring.
        """
        if a < 0.0:
            if v + a * dt <= 0.0:
                # Cart comes to rest partway through this step.
                dt_stop = -v / a  # unused beyond documenting the exact crossing time
                dx = -(v * v) / (2.0 * a)
                return x + dx, 0.0
            dx = v * dt + 0.5 * a * dt * dt
            return x + dx, v + a * dt

        if a > 0.0:
            if v + a * dt > self.v_cruise:
                # Cart reaches v_cruise partway through this step, then
                # coasts at exactly v_cruise for the remainder.
                dt_acc = (self.v_cruise - v) / a
                dx = ((self.v_cruise * self.v_cruise) - (v * v)) / (2.0 * a) + self.v_cruise * (dt - dt_acc)
                return x + dx, self.v_cruise
            dx = v * dt + 0.5 * a * dt * dt
            return x + dx, v + a * dt

        return x + v * dt, v

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

        `evidence_override`, if given, replaces extract_evidence() with a
        caller-supplied function of the current time -- used by tests to
        inject specific/corrupted evidence sequences (e.g. to force the
        zero-likelihood fallback) without needing a coincidentally
        self-inconsistent CPT.
        """
        profile = trajectory_profile
        self.v_cruise = profile.v_cruise
        evidence_period = profile.evidence_period_s if profile.evidence_period_s is not None else profile.dt

        x, v = initial_state.position, initial_state.velocity
        t = 0.0

        track_condition, decel_capability = self._ground_truth_labels()
        dropout_window = self._dropout_window()

        records: List[TimestepRecord] = []
        action_history: List[str] = []
        false_emergency_stop_count = 0
        fallback_event_count = 0
        emergency_brake_engaged = False
        first_emergency_brake_t: Optional[float] = None
        collided = False
        settled = False
        consecutive_settled_steps = 0
        has_moved = False

        # Zero-order hold: the decision (and the evidence that produced it)
        # only updates at evidence_period boundaries; physics integrates
        # every dt in between using whatever decision is currently held.
        # A decision's bookkeeping (false-alarm/fallback counts, history) is
        # attributed once, at the moment it is *made*, not once per physics
        # tick it happens to be held over.
        current_evidence: Dict[str, EvidenceValue] = {}
        current_action = "COAST"  # safe default before the first perception update
        current_fallback = False
        self._current_a = 0.0
        next_evidence_t = 0.0

        n_steps = max(1, math.ceil(profile.max_duration_s / profile.dt))
        for _ in range(n_steps):
            if t >= next_evidence_t - 1e-9:
                obstacle_x = self.scenario.obstacle_position if self._obstacle_detectable(t, profile) else None
                if evidence_override is not None:
                    current_evidence = evidence_override(t)
                else:
                    current_evidence = self.extract_evidence(
                        x,
                        v,
                        obstacle_x,
                        track_condition,
                        decel_capability,
                        t,
                        dropout_window,
                        detection_range_m=profile.detection_range_m,
                        slip_threshold=profile.slip_threshold,
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

                self._current_a = self._commanded_acceleration(current_action, profile)

            a_cmd = self._current_a
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
