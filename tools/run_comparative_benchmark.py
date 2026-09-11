"""
tools/run_comparative_benchmark.py

Executes a 5-way comparative efficacy benchmark:
  1. HIRAM (Bayesian C) -- the compiled C99 kernel via the ctypes bridge
  2. HIRAM (Ablated: No Wheel-Slip) -- same kernel, WheelSlipObs forced
     HIRAM_UNOBSERVED on every step; isolates whether wheel-slip inference is
     causally load-bearing for collision avoidance (see "Wheel-Slip Ablation"
     below), run across all 5 scenarios like every other controller
  3. Naive Threshold Controller (brakes on any range-sensor detection)
  4. Conservative Rule-Based Controller (a fixed discrete-evidence rule
     table, not a continuous kinematic stopping-distance calculation --
     see ConservativeRuleBasedEngine)
  5. Always-Brake Reference (trivial safety baseline)

Evaluates performance across 5 critical operational regimes:
  - Nominal Clear Track (traversal speed and false-positive evaluation)
  - Static Popup Obstacle (stopping clearance and collision avoidance)
  - Sensor Dropout Blackout (500 ms blindness window)
  - Degraded Braking on Slippery Track (reduced friction dynamics)
  - Simultaneous Faults (moving start + degraded braking + 500 ms sensor blackout)

Wheel-Slip Ablation: rather than a separate, independently re-simulated
study, HIRAM (Ablated: No Wheel-Slip)'s Scenario 5 run *is* the ablation
study's "ablated" arm, and HIRAM (Bayesian C)'s Scenario 5 run is its
"baseline" arm -- see run_wheel_slip_ablation(), which builds the causal
chronology and finding from those two already-computed traces (bit-exact
with the 5-way table above; nothing is re-run under different conditions).

Latency methodology: every controller's timed span covers exactly the same
unit of work -- evidence-to-decision-dict, i.e. everything from the start of
`.step(evidence)` through the fully constructed return dict, not just the
"decision logic" portion. This matters because a controller that returns a
dict built *before* its timer starts (or stops before building it) would be
timed inconsistently with one where the return dict itself is the memoized
result of a heavier call (e.g. HIRAM's `step_c_kernel()`).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.hiram_c_binding import step_c_kernel
from tools.trajectory_evaluator import TrajectoryEvaluator


def _first_present(evidence: Dict[str, Optional[int]], *keys: str) -> Optional[int]:
    """Returns the value of the first key present in `evidence` (PascalCase
    keys, e.g. "LidarObs", checked before their lowercase aliases), or None
    if none of `keys` is present at all -- distinct from a key being present
    with an explicit value of None (unobserved), which is returned as-is."""
    for key in keys:
        if key in evidence:
            return evidence[key]
    return None


def _extract_obs(evidence: Dict[str, Optional[int]]) -> tuple[int, int, int]:
    """Robustly extracts lidar, tof, and slip values, accepting either the
    kernel's own PascalCase evidence keys (LidarObs/TofObs/WheelSlipObs) or
    common lowercase/snake_case aliases (lidar/lidar_obs, tof/tof_obs,
    wheel_slip/wheel_slip_obs/slip) -- so this benchmark can drive both
    tools.trajectory_evaluator's evidence dicts (PascalCase) and ad hoc
    lowercase evidence a caller might hand-write."""
    lidar = _first_present(evidence, "LidarObs", "lidar_obs", "lidar")
    tof = _first_present(evidence, "TofObs", "tof_obs", "tof")
    slip = _first_present(evidence, "WheelSlipObs", "wheel_slip_obs", "wheel_slip", "slip")

    c_lidar = -1 if lidar is None else int(lidar)
    c_tof = -1 if tof is None else int(tof)
    c_slip = -1 if slip is None else int(slip)
    return c_lidar, c_tof, c_slip


# ==============================================================================
# Controller Adapter Implementations
# ==============================================================================

class HiramCKernelEngine:
    """Invokes the compiled C99 kernel via the ctypes bridge."""
    def __init__(self):
        self.total_eval_time_us = 0.0
        self.eval_count = 0

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        # Timed span starts at entry (matches every other engine below):
        # evidence extraction through the fully-constructed return dict --
        # step_c_kernel() itself builds and returns that dict, so no
        # separate dict-construction step is needed here.
        t0 = time.perf_counter()

        c_lidar, c_tof, c_slip = _extract_obs(evidence)
        out = step_c_kernel(lidar=c_lidar, tof=c_tof, slip=c_slip)

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return out


class WheelSlipAblatedHiramEngine(HiramCKernelEngine):
    """Wheel-Slip Ablation Study (Task 3.2 Astra follow-up): identical to
    HiramCKernelEngine, except WheelSlipObs is forced to HIRAM_UNOBSERVED
    (-1) on every step before evaluation, regardless of what the plant
    simulator actually measured -- isolating whether wheel-slip evidence is
    causally load-bearing for collision avoidance, as opposed to merely
    correlated with it (see run_wheel_slip_ablation). Run directly in
    controller_factories like every other engine below, so it is evaluated
    across all 5 scenarios, not just Scenario 5.

    Overrides step() outright (rather than doing the ablation and then
    calling super().step()) so its own timed span still covers its entire
    entry-to-return-dict work, including the ablation itself -- consistent
    with every other engine's latency measurement boundary."""

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        ablated = dict(evidence)
        ablated["WheelSlipObs"] = None
        ablated.pop("wheel_slip_obs", None)
        ablated.pop("wheel_slip", None)
        ablated.pop("slip", None)
        c_lidar, c_tof, c_slip = _extract_obs(ablated)
        out = step_c_kernel(lidar=c_lidar, tof=c_tof, slip=c_slip)

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return out


class NaiveThresholdEngine:
    """
    Standard simple logic:
    Brakes if either range sensor indicates an obstacle (state 1).
    Accelerates on clear (0) or dropout (-1 / 2).
    """
    def __init__(self):
        self.total_eval_time_us = 0.0
        self.eval_count = 0

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        c_lidar, c_tof, _ = _extract_obs(evidence)
        if c_lidar == 1 or c_tof == 1:
            action = "EMERGENCY_BRAKE"
        else:
            action = "ACCEL"
        result = {"selected_action": action, "best_action_name": action}

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return result


class ConservativeRuleBasedEngine:
    """
    A fixed discrete-evidence rule table (NOT a continuous kinematic
    worst-case-stopping-distance calculation, despite this class's earlier
    name -- it never computes a stopping distance, brake deceleration, or
    time-to-contact; it only pattern-matches this step's tri-state evidence
    values against a small if/elif chain): brakes conservatively whenever
    sensing is compromised/unobserved or a range sensor detects an obstacle,
    coasts on elevated wheel slip, otherwise accelerates.
    """
    def __init__(self):
        self.total_eval_time_us = 0.0
        self.eval_count = 0

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        c_lidar, c_tof, c_slip = _extract_obs(evidence)
        # In unobserved state or dropout, play it safe and emergency brake
        if c_lidar in (-1, 2) or c_tof in (-1, 2):
            action = "EMERGENCY_BRAKE"
        elif c_lidar == 1 or c_tof == 1:
            action = "EMERGENCY_BRAKE"
        elif c_slip == 1:
            action = "COAST"
        else:
            action = "ACCEL"
        result = {"selected_action": action, "best_action_name": action}

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return result


class AlwaysBrakeEngine:
    """Trivial reference baseline."""
    def __init__(self):
        self.total_eval_time_us = 0.0
        self.eval_count = 0

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        t0 = time.perf_counter()

        result = {"selected_action": "EMERGENCY_BRAKE", "best_action_name": "EMERGENCY_BRAKE"}

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return result


# ==============================================================================
# Benchmark Harness
# ==============================================================================

BUILD_DIR = REPO_ROOT / "build"
JSON_RESULTS_PATH = BUILD_DIR / "benchmark_results.json"
TABLE_RESULTS_PATH = BUILD_DIR / "benchmark_results.md"


@dataclass
class ScenarioSpec:
    name: str
    initial_x: float
    initial_v: float
    max_duration: float
    obstacle_x: Optional[float]
    track_condition: str
    decel_capability: str
    dropout_window: Optional[tuple[float, float]]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "initial_x": self.initial_x,
            "initial_v": self.initial_v,
            "max_duration": self.max_duration,
            "obstacle_x": self.obstacle_x,
            "track_condition": self.track_condition,
            "decel_capability": self.decel_capability,
            "dropout_window": list(self.dropout_window) if self.dropout_window else None,
        }


def _trace_as_dict(res) -> Dict[str, Any]:
    """Serializes one TrajectoryResult's full per-step trace (Astra
    requirement: "per-scenario traces" / "per-step action trace
    recording") -- every field needed to reproduce and independently audit
    the episode without re-running the simulator."""
    return {
        "time_history": res.time_history,
        "x_history": res.x_history,
        "v_history": res.v_history,
        "a_history": res.a_history,
        "action_history": res.action_history,
        "collision": res.collision,
        "contact_time": res.contact_time,
        "contact_velocity": res.contact_velocity,
        "impact_energy": res.impact_energy,
        "final_x": res.final_x,
        "final_v": res.final_v,
        "total_time": res.total_time,
    }


@dataclass
class ControllerScorecard:
    """Astra note: `results` (episodes_evaluated, collisions,
    nuisance_stops, total_progress_m, impact_velocities, scenario_traces --
    every field derived purely from the deterministic plant/kernel
    simulation) and `timing` (wall-clock host latency, which varies run to
    run with OS scheduling jitter) are kept as two clearly separate groups
    -- see as_dict() -- specifically so re-running this script reproduces
    `results` bit-exactly while `timing` is free to differ."""

    episodes_evaluated: int = 0
    collisions: int = 0
    nuisance_stops: int = 0
    total_progress_m: float = 0.0
    impact_velocities: List[float] = field(default_factory=list)
    scenario_traces: Dict[str, Any] = field(default_factory=dict)

    total_time_us: float = 0.0
    steps_evaluated: int = 0

    def as_dict(self) -> Dict[str, Any]:
        avg_latency_us = (self.total_time_us / self.steps_evaluated) if self.steps_evaluated > 0 else 0.0
        return {
            "results": {
                "episodes_evaluated": self.episodes_evaluated,
                "collisions": self.collisions,
                "nuisance_stops": self.nuisance_stops,
                "total_progress_m": self.total_progress_m,
                "impact_velocities": self.impact_velocities,
                "scenario_traces": self.scenario_traces,
            },
            "timing": {
                "note": "Observed host wall-clock execution time (time.perf_counter()); "
                "non-deterministic run-to-run, and NOT a WCET bound. Kept separate from "
                "'results' above so that section reproduces bit-exactly across runs.",
                "total_eval_time_us": self.total_time_us,
                "steps_evaluated": self.steps_evaluated,
                "avg_step_latency_us": avg_latency_us,
            },
        }


def _first_brake_or_coast_index(action_history: List[str]) -> Optional[int]:
    for i, action in enumerate(action_history):
        if action in ("EMERGENCY_BRAKE", "COAST"):
            return i
    return None


def _first_action_index(action_history: List[str], target: str) -> Optional[int]:
    for i, action in enumerate(action_history):
        if action == target:
            return i
    return None


_SCENARIO5_EXPECTED_ABLATED_CONTACT_VELOCITY_MPS = 0.275
_SCENARIO5_CONTACT_VELOCITY_TOLERANCE_MPS = 1e-6


def _build_scenario5_chronology(
    scenario5: ScenarioSpec, baseline_trace: Dict[str, Any], ablated_trace: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Builds the causal chronology of Scenario 5 ("Simultaneous Faults")
    entirely from the two already-computed traces -- every t/x/v/action
    value below is read out of `baseline_trace`/`ablated_trace`, never
    hardcoded, so this stays correct if the scenario parameters ever
    change. See run_wheel_slip_ablation for how the two traces are
    obtained (the same 5-way benchmark run that produced the results
    table, not a separately re-simulated pair)."""
    dropout_start, dropout_end = scenario5.dropout_window
    obstacle_x = scenario5.obstacle_x

    baseline_actions = baseline_trace["action_history"]
    ablated_actions = ablated_trace["action_history"]
    # Deliberately the EMERGENCY_BRAKE-specific index here (not
    # _first_brake_or_coast_index, which would also match this scenario's
    # shared initial COAST at step 0 for both controllers and so could
    # never distinguish "braking" from "still coasting").
    baseline_brake_idx = _first_action_index(baseline_actions, "EMERGENCY_BRAKE")
    ablated_brake_idx = _first_action_index(ablated_actions, "EMERGENCY_BRAKE")

    events: List[Dict[str, Any]] = []

    events.append({
        "t": baseline_trace["time_history"][0],
        "event": (
            "Range-sensor blackout begins (LidarObs/TofObs unobserved). No braking-induced "
            "wheel slip has occurred yet (WheelSlipObs=0). Both baseline and ablated HIRAM "
            "command COAST."
        ),
        "baseline_action": baseline_actions[0],
        "ablated_action": ablated_actions[0],
    })

    if baseline_brake_idx is not None:
        t_brake = baseline_trace["time_history"][baseline_brake_idx]
        events.append({
            "t": t_brake,
            "event": (
                "Deceleration from the prior COAST step's drag, combined with the WET/"
                "DEGRADED track in this scenario's sensor model (see "
                "TrajectoryEvaluator.extract_evidence: slip_ratio=0.25 when a_current<0 on "
                "WET/ICY or non-NOMINAL decel capability, vs. 0.02 otherwise -- both above "
                "and below slip_threshold=0.15 respectively), produces WheelSlipObs=1. "
                "Baseline HIRAM observes this and transitions to EMERGENCY_BRAKE. Ablated "
                "HIRAM has WheelSlipObs forced unobserved and cannot see this signal -- it "
                "continues COAST."
            ),
            "baseline_action": baseline_actions[baseline_brake_idx],
            "ablated_action": ablated_actions[baseline_brake_idx] if baseline_brake_idx < len(ablated_actions) else None,
        })

    baseline_stop_x = baseline_trace["final_x"]
    baseline_clearance = (obstacle_x - baseline_stop_x) if obstacle_x is not None else None
    events.append({
        "t": dropout_end,
        "event": (
            f"Range-sensor blackout ends; LidarObs/TofObs become observed again. Baseline "
            f"HIRAM has been braking since t={baseline_trace['time_history'][baseline_brake_idx]:.4f}s "
            f"and is already decelerating toward a full stop (it will halt at "
            f"x={baseline_stop_x:.6f} m, clearance {baseline_clearance:.6f} m -- no "
            f"collision). Ablated HIRAM only now (first) observes the obstacle directly and "
            f"transitions to EMERGENCY_BRAKE -- too late, given its speed and this "
            f"scenario's degraded braking deceleration, to stop short of the obstacle."
            if baseline_brake_idx is not None else
            "Range-sensor blackout ends; LidarObs/TofObs become observed again."
        ),
        "baseline_action": "EMERGENCY_BRAKE",
        "ablated_action": ablated_actions[ablated_brake_idx] if ablated_brake_idx is not None else None,
    })

    if ablated_trace["collision"]:
        events.append({
            "t": ablated_trace["contact_time"],
            "event": (
                f"Ablated HIRAM makes contact with the obstacle at "
                f"v={ablated_trace['contact_velocity']:.6f} m/s. Baseline HIRAM (same "
                f"scenario, WheelSlipObs observed) did not collide."
            ),
            "baseline_action": None,
            "ablated_action": "EMERGENCY_BRAKE (post-contact)",
        })

    return events


def run_wheel_slip_ablation(
    scenario5: ScenarioSpec, baseline_trace: Dict[str, Any], ablated_trace: Dict[str, Any]
) -> Dict[str, Any]:
    """Wheel-Slip Ablation Study: compares HIRAM (Bayesian C)'s and HIRAM
    (Ablated: No Wheel-Slip)'s already-computed Scenario 5 ("Simultaneous
    Faults") traces -- both produced by the same 5-way benchmark loop in
    run_benchmark(), not a separate re-simulation -- to determine whether
    wheel-slip inference is a causal factor in this scenario's collision
    avoidance (as opposed to merely correlated with it), and reconstructs
    the causal chronology of what each controller observed and commanded,
    and when.
    """
    # EMERGENCY_BRAKE-specific (not _first_brake_or_coast_index): Scenario 5
    # has both controllers start with COAST, so a "brake-or-coast" index
    # would resolve to step 0 for both and could never distinguish them.
    baseline_brake_idx = _first_action_index(baseline_trace["action_history"], "EMERGENCY_BRAKE")
    ablated_brake_idx = _first_action_index(ablated_trace["action_history"], "EMERGENCY_BRAKE")

    baseline_collision = baseline_trace["collision"]
    ablated_collision = ablated_trace["collision"]

    if ablated_collision and not baseline_collision:
        finding = (
            "Ablating WheelSlipObs (forcing it unobserved) introduced a collision that did "
            "not occur in the baseline HIRAM run -- wheel-slip inference is causally "
            "load-bearing for collision avoidance in this scenario, not merely correlated "
            "with it."
        )
        collision_delta = "ablation-caused collision"
    elif ablated_collision and baseline_collision:
        bv = baseline_trace["contact_velocity"] or 0.0
        av = ablated_trace["contact_velocity"] or 0.0
        if av > bv + 1e-9:
            finding = (
                f"Both runs collided, but the ablated run made contact at a higher velocity "
                f"({av:.4f} m/s vs {bv:.4f} m/s) -- wheel-slip inference reduced impact "
                f"severity even though it did not prevent contact in this scenario."
            )
        else:
            finding = (
                "Both the baseline and ablated runs collided at essentially the same "
                "velocity -- wheel-slip inference alone did not change the collision "
                "outcome for this scenario instance."
            )
        collision_delta = "both collided"
    elif not ablated_collision and not baseline_collision:
        if ablated_brake_idx is not None and baseline_brake_idx is not None and ablated_brake_idx > baseline_brake_idx:
            delay_steps = ablated_brake_idx - baseline_brake_idx
            finding = (
                f"Neither run collided, but the ablated run braked {delay_steps} step(s) "
                f"later than the baseline (first EMERGENCY_BRAKE at step {ablated_brake_idx} vs "
                f"{baseline_brake_idx}) -- wheel-slip inference brought braking forward, "
                f"though it was not strictly necessary to avoid contact in this scenario "
                f"instance."
            )
        else:
            finding = (
                "Neither run collided and braking timing was not meaningfully different -- "
                "wheel-slip inference was not the deciding factor in this scenario instance."
            )
        collision_delta = "neither collided"
    else:
        finding = (
            "Unexpected: the ablated run avoided a collision the baseline run had -- "
            "this does not fit the expected causal direction and warrants scenario review."
        )
        collision_delta = "unexpected: ablation avoided baseline collision"

    if ablated_collision:
        observed_v = ablated_trace["contact_velocity"]
        if observed_v is None or abs(observed_v - _SCENARIO5_EXPECTED_ABLATED_CONTACT_VELOCITY_MPS) > _SCENARIO5_CONTACT_VELOCITY_TOLERANCE_MPS:
            raise AssertionError(
                f"Scenario 5 ablated-run contact velocity drifted from the audited value: "
                f"expected {_SCENARIO5_EXPECTED_ABLATED_CONTACT_VELOCITY_MPS} m/s "
                f"(+/- {_SCENARIO5_CONTACT_VELOCITY_TOLERANCE_MPS}), got {observed_v!r} m/s. "
                f"This must be investigated before the dossier is regenerated -- it means "
                f"either the scenario/model changed or a prior report figure was wrong."
            )

    return {
        "scenario": scenario5.name,
        "collision_delta": collision_delta,
        "baseline": {
            "wheel_slip_ablated": False,
            "collision": baseline_collision,
            "contact_time": baseline_trace["contact_time"],
            "contact_velocity": baseline_trace["contact_velocity"],
            "final_x": baseline_trace["final_x"],
            "clearance_m": (scenario5.obstacle_x - baseline_trace["final_x"]) if scenario5.obstacle_x is not None else None,
            "first_emergency_brake_step": baseline_brake_idx,
        },
        "ablated": {
            "wheel_slip_ablated": True,
            "collision": ablated_collision,
            "contact_time": ablated_trace["contact_time"],
            "contact_velocity": ablated_trace["contact_velocity"],
            "first_emergency_brake_step": ablated_brake_idx,
        },
        "chronology": _build_scenario5_chronology(scenario5, baseline_trace, ablated_trace),
        "finding": finding,
    }


def run_benchmark():
    scenarios = [
        ScenarioSpec(
            name="1. Nominal Clear Track",
            initial_x=0.0,
            initial_v=0.0,
            max_duration=2.0,
            obstacle_x=None,
            track_condition="DRY",
            decel_capability="NOMINAL",
            dropout_window=None,
        ),
        ScenarioSpec(
            name="2. Static Popup Obstacle (0.35m)",
            initial_x=0.0,
            initial_v=0.0,
            max_duration=3.0,
            obstacle_x=0.35,
            track_condition="DRY",
            decel_capability="NOMINAL",
            dropout_window=None,
        ),
        ScenarioSpec(
            name="3. Moving Sensor Dropout (500ms)",
            initial_x=0.0,
            initial_v=0.35,
            max_duration=3.0,
            obstacle_x=0.45,
            track_condition="DRY",
            decel_capability="NOMINAL",
            dropout_window=(0.20, 0.70),
        ),
        ScenarioSpec(
            name="4. Degraded Brakes on Wet Track",
            initial_x=0.0,
            initial_v=0.20,
            max_duration=3.0,
            obstacle_x=0.30,
            track_condition="WET",
            decel_capability="DEGRADED",
            dropout_window=None,
        ),
        ScenarioSpec(
            name="5. Simultaneous Faults (Dropout+Degraded)",
            initial_x=0.0,
            initial_v=0.35,
            max_duration=3.0,
            obstacle_x=0.20,
            track_condition="WET",
            decel_capability="DEGRADED",
            dropout_window=(0.0, 0.50),
        ),
    ]

    # HIRAM (Ablated: No Wheel-Slip) runs here in the main loop, across all
    # 5 scenarios, exactly like every other controller -- not just Scenario
    # 5 -- per the Astra follow-up (its earlier restriction to Scenario 5
    # only understated its total progress figure). Its Scenario 5 row is
    # simultaneously the ablation study's "ablated" arm; see
    # run_wheel_slip_ablation below, which reuses these same traces rather
    # than re-simulating.
    controller_factories = {
        "HIRAM (Bayesian C)": HiramCKernelEngine,
        "HIRAM (Ablated: No Wheel-Slip)": WheelSlipAblatedHiramEngine,
        "Naive Threshold": NaiveThresholdEngine,
        "Conservative Rule-Based": ConservativeRuleBasedEngine,
        "Always-Brake": AlwaysBrakeEngine,
    }

    results: Dict[str, ControllerScorecard] = {}

    print("\n" + "=" * 80)
    print("      HIRAM 5-WAY COMPARATIVE EFFICACY BENCHMARK (TASK 3.2)")
    print("=" * 80)

    for ctrl_name, factory in controller_factories.items():
        scorecard = ControllerScorecard()
        engine = factory()
        evaluator = TrajectoryEvaluator(engine=engine)

        for sc in scenarios:
            res = evaluator.run_episode(
                initial_x=sc.initial_x,
                initial_v=sc.initial_v,
                max_duration=sc.max_duration,
                obstacle_x=sc.obstacle_x,
                track_condition=sc.track_condition,
                decel_capability=sc.decel_capability,
                dropout_window=sc.dropout_window,
            )

            scorecard.episodes_evaluated += 1

            # Metric: Collision
            if res.collision:
                scorecard.collisions += 1
                if res.contact_velocity is not None:
                    scorecard.impact_velocities.append(res.contact_velocity)

            # Metric: Nuisance / False-Positive Stop
            if sc.obstacle_x is None and ("EMERGENCY_BRAKE" in res.action_history):
                scorecard.nuisance_stops += 1

            # Metric: Progress
            scorecard.total_progress_m += (res.final_x - sc.initial_x)

            # Per-scenario step trace, for reproducibility/audit.
            scorecard.scenario_traces[sc.name] = _trace_as_dict(res)

        scorecard.total_time_us = engine.total_eval_time_us
        scorecard.steps_evaluated = engine.eval_count
        results[ctrl_name] = scorecard

    # Print Report
    print(f"\n{'Controller Name':<24} | {'Collisions':<11} | {'False Halts':<12} | {'Total Progress':<15} | {'Avg Step Latency'}")
    print("-" * 80)

    for name, sc in results.items():
        avg_latency = (sc.total_time_us / sc.steps_evaluated) if sc.steps_evaluated > 0 else 0.0
        print(
            f"{name:<24} | "
            f"{sc.collisions:<11} | "
            f"{sc.nuisance_stops:<12} | "
            f"{sc.total_progress_m:<13.3f} m | "
            f"{avg_latency:.2f} us"
        )

    print("-" * 80)
    print("Detailed Impact & Severity:")
    for name, sc in results.items():
        if sc.impact_velocities:
            v_str = ", ".join(f"{v:.3f} m/s" for v in sc.impact_velocities)
            print(f"  * {name}: {len(sc.impact_velocities)} collisions (Velocities: {v_str})")
        else:
            print(f"  * {name}: 0 collisions recorded.")

    print("\n" + "=" * 80)

    print("\n" + "=" * 80)
    print("      WHEEL-SLIP ABLATION STUDY (Scenario 5) -- from the 5-way rows above")
    print("=" * 80)
    scenario5 = next(sc for sc in scenarios if sc.name.startswith("5."))
    baseline_trace5 = results["HIRAM (Bayesian C)"].scenario_traces[scenario5.name]
    ablated_trace5 = results["HIRAM (Ablated: No Wheel-Slip)"].scenario_traces[scenario5.name]
    ablation = run_wheel_slip_ablation(scenario5, baseline_trace5, ablated_trace5)
    print(f"Scenario: {ablation['scenario']}")
    print(
        f"  Baseline : collision={ablation['baseline']['collision']} "
        f"contact_time={ablation['baseline']['contact_time']} "
        f"contact_velocity={ablation['baseline']['contact_velocity']} "
        f"final_x={ablation['baseline']['final_x']} clearance_m={ablation['baseline']['clearance_m']}"
    )
    print(
        f"  Ablated  : collision={ablation['ablated']['collision']} "
        f"contact_time={ablation['ablated']['contact_time']} "
        f"contact_velocity={ablation['ablated']['contact_velocity']}"
    )
    print("  Chronology:")
    for evt in ablation["chronology"]:
        print(f"    t={evt['t']:.4f}s  baseline={evt['baseline_action']}  ablated={evt['ablated_action']}")
        print(f"             {evt['event']}")
    print(f"  Finding  : {ablation['finding']}")
    print("=" * 80)

    return scenarios, results, ablation


def _write_json_results(scenarios: List[ScenarioSpec], results: Dict[str, ControllerScorecard], ablation: Dict[str, Any]) -> None:
    payload = {
        "scenarios": [sc.as_dict() for sc in scenarios],
        "controllers": {name: sc.as_dict() for name, sc in results.items()},
        "wheel_slip_ablation": ablation,
    }
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    JSON_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_markdown_table(scenarios: List[ScenarioSpec], results: Dict[str, ControllerScorecard], ablation: Dict[str, Any]) -> None:
    lines = [
        "# HIRAM 5-Way Comparative Efficacy Benchmark (Task 3.2)",
        "",
        "| Controller | Collisions | False Halts | Total Progress (m) | Avg Step Latency (us) |",
        "|---|---|---|---|---|",
    ]
    for name, sc in results.items():
        avg_latency = (sc.total_time_us / sc.steps_evaluated) if sc.steps_evaluated > 0 else 0.0
        lines.append(f"| {name} | {sc.collisions} | {sc.nuisance_stops} | {sc.total_progress_m:.3f} | {avg_latency:.2f} |")

    lines += ["", "## Detailed Impact & Severity", ""]
    for name, sc in results.items():
        if sc.impact_velocities:
            v_str = ", ".join(f"{v:.6f} m/s" for v in sc.impact_velocities)
            lines.append(f"- **{name}**: {len(sc.impact_velocities)} collision(s) (velocities: {v_str})")
        else:
            lines.append(f"- **{name}**: 0 collisions recorded.")

    lines += [
        "",
        "## Wheel-Slip Ablation Study (Scenario 5)",
        "",
        f"Scenario: {ablation['scenario']}",
        "",
        f"- Baseline: collision={ablation['baseline']['collision']}, "
        f"contact_time={ablation['baseline']['contact_time']}, "
        f"contact_velocity={ablation['baseline']['contact_velocity']}, "
        f"final_x={ablation['baseline']['final_x']}, clearance_m={ablation['baseline']['clearance_m']}",
        f"- Ablated (WheelSlipObs forced unobserved): collision={ablation['ablated']['collision']}, "
        f"contact_time={ablation['ablated']['contact_time']}, "
        f"contact_velocity={ablation['ablated']['contact_velocity']}",
        "",
        "### Causal chronology",
        "",
        "| t (s) | Baseline action | Ablated action | Event |",
        "|---|---|---|---|",
    ]
    for evt in ablation["chronology"]:
        lines.append(f"| {evt['t']:.4f} | {evt['baseline_action']} | {evt['ablated_action']} | {evt['event']} |")

    lines += [
        "",
        f"**Finding:** {ablation['finding']}",
        "",
        "Sensor model scope: in the current simulation model "
        "(tools/trajectory_evaluator.py:TrajectoryEvaluator.extract_evidence), wheel slip "
        "is triggered purely as a function of commanded deceleration on a WET/ICY track or "
        "under non-NOMINAL decel capability (slip_ratio=0.25, above slip_threshold=0.15) "
        "vs. 0.02 otherwise -- it is not an independent random fault injection. WheelSlipObs "
        "therefore only carries information once the plant is decelerating; it is silent "
        "during the initial COAST phase in Scenario 5.",
        "",
        "Note: timing figures throughout this report are observed host wall-clock "
        "execution times of the Python/ctypes call path (`time.perf_counter()`, spanning "
        "each controller's full evidence-to-decision-dict `step()` call -- the same unit "
        "of work for every controller), not a hard worst-case-execution-time (WCET) bound "
        "for the on-target Cortex-M7 kernel.",
    ]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> Dict[str, Any]:
    scenarios, results, ablation = run_benchmark()
    _write_json_results(scenarios, results, ablation)
    _write_markdown_table(scenarios, results, ablation)
    print(f"\nWrote {JSON_RESULTS_PATH}")
    print(f"Wrote {TABLE_RESULTS_PATH}")
    return {
        "scenarios": [sc.as_dict() for sc in scenarios],
        "controllers": {name: sc.as_dict() for name, sc in results.items()},
        "wheel_slip_ablation": ablation,
    }


if __name__ == "__main__":
    main()