"""
tools/run_comparative_benchmark.py

Executes a 4-way comparative efficacy benchmark:
  1. HIRAM C-Kernel (Bayesian Inference + Loss Matrix)
  2. Naive Threshold Controller (Brakes on any detection)
  3. Conservative Kinematic Controller (Worst-case stopping distance + safety margin)
  4. Always-Brake Reference (Trivial safety baseline)

Evaluates performance across 5 critical operational regimes:
  - Nominal Clear Track (traversal speed and false-positive evaluation)
  - Static Popup Obstacle (stopping clearance and collision avoidance)
  - Sensor Dropout Blackout (500 ms blindness window)
  - Degraded Braking on Slippery Track (reduced friction dynamics)
  - Simultaneous Faults (moving start + degraded braking + 500 ms sensor blackout)
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
        c_lidar, c_tof, c_slip = _extract_obs(evidence)

        t0 = time.perf_counter()
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
    correlated with it (see run_wheel_slip_ablation)."""

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        ablated = dict(evidence)
        ablated["WheelSlipObs"] = None
        ablated.pop("wheel_slip_obs", None)
        ablated.pop("wheel_slip", None)
        ablated.pop("slip", None)
        return super().step(ablated)


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
        c_lidar, c_tof, _ = _extract_obs(evidence)

        t0 = time.perf_counter()
        if c_lidar == 1 or c_tof == 1:
            action = "EMERGENCY_BRAKE"
        else:
            action = "ACCEL"

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return {"selected_action": action, "best_action_name": action}


class ConservativeKinematicEngine:
    """
    Industry safety standard:
    Brakes conservatively whenever sensing is compromised or unobserved.
    """
    def __init__(self):
        self.total_eval_time_us = 0.0
        self.eval_count = 0

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        c_lidar, c_tof, c_slip = _extract_obs(evidence)

        t0 = time.perf_counter()
        # In unobserved state or dropout, play it safe and emergency brake
        if c_lidar in (-1, 2) or c_tof in (-1, 2):
            action = "EMERGENCY_BRAKE"
        elif c_lidar == 1 or c_tof == 1:
            action = "EMERGENCY_BRAKE"
        elif c_slip == 1:
            action = "COAST"
        else:
            action = "ACCEL"

        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return {"selected_action": action, "best_action_name": action}


class AlwaysBrakeEngine:
    """Trivial reference baseline."""
    def __init__(self):
        self.total_eval_time_us = 0.0
        self.eval_count = 0

    def step(self, evidence: Dict[str, Optional[int]]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        dt_us = (time.perf_counter() - t0) * 1e6
        self.total_eval_time_us += dt_us
        self.eval_count += 1
        return {"selected_action": "EMERGENCY_BRAKE", "best_action_name": "EMERGENCY_BRAKE"}


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
    collisions: int = 0
    nuisance_stops: int = 0
    total_progress_m: float = 0.0
    impact_velocities: List[float] = field(default_factory=list)
    total_time_us: float = 0.0
    steps_evaluated: int = 0
    scenario_traces: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        avg_latency_us = (self.total_time_us / self.steps_evaluated) if self.steps_evaluated > 0 else 0.0
        return {
            "collisions": self.collisions,
            "nuisance_stops": self.nuisance_stops,
            "total_progress_m": self.total_progress_m,
            "impact_velocities": self.impact_velocities,
            "total_eval_time_us": self.total_time_us,
            "steps_evaluated": self.steps_evaluated,
            "avg_step_latency_us": avg_latency_us,
            "scenario_traces": self.scenario_traces,
        }


def _first_brake_or_coast_index(action_history: List[str]) -> Optional[int]:
    for i, action in enumerate(action_history):
        if action in ("EMERGENCY_BRAKE", "COAST"):
            return i
    return None


def run_wheel_slip_ablation(scenarios: List[ScenarioSpec]) -> Dict[str, Any]:
    """Wheel-Slip Ablation Study: runs Scenario 5 ("Simultaneous Faults")
    twice under HIRAM -- once normally, once with WheelSlipObs forced to
    HIRAM_UNOBSERVED on every step (WheelSlipAblatedHiramEngine) -- and
    empirically compares the two outcomes to determine whether wheel-slip
    inference is a causal factor in this scenario's collision avoidance
    (as opposed to merely correlated with it), rather than asserting that
    from first principles.
    """
    scenario5 = next(sc for sc in scenarios if sc.name.startswith("5."))

    baseline_engine = HiramCKernelEngine()
    baseline_res = TrajectoryEvaluator(engine=baseline_engine).run_episode(
        initial_x=scenario5.initial_x,
        initial_v=scenario5.initial_v,
        max_duration=scenario5.max_duration,
        obstacle_x=scenario5.obstacle_x,
        track_condition=scenario5.track_condition,
        decel_capability=scenario5.decel_capability,
        dropout_window=scenario5.dropout_window,
    )

    ablated_engine = WheelSlipAblatedHiramEngine()
    ablated_res = TrajectoryEvaluator(engine=ablated_engine).run_episode(
        initial_x=scenario5.initial_x,
        initial_v=scenario5.initial_v,
        max_duration=scenario5.max_duration,
        obstacle_x=scenario5.obstacle_x,
        track_condition=scenario5.track_condition,
        decel_capability=scenario5.decel_capability,
        dropout_window=scenario5.dropout_window,
    )

    baseline_brake_idx = _first_brake_or_coast_index(baseline_res.action_history)
    ablated_brake_idx = _first_brake_or_coast_index(ablated_res.action_history)

    if ablated_res.collision and not baseline_res.collision:
        finding = (
            "Ablating WheelSlipObs (forcing it unobserved) introduced a collision that did "
            "not occur in the baseline HIRAM run -- wheel-slip inference is causally "
            "load-bearing for collision avoidance in this scenario, not merely correlated "
            "with it."
        )
        collision_delta = "ablation-caused collision"
    elif ablated_res.collision and baseline_res.collision:
        bv = baseline_res.contact_velocity or 0.0
        av = ablated_res.contact_velocity or 0.0
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
    elif not ablated_res.collision and not baseline_res.collision:
        if ablated_brake_idx is not None and baseline_brake_idx is not None and ablated_brake_idx > baseline_brake_idx:
            delay_steps = ablated_brake_idx - baseline_brake_idx
            finding = (
                f"Neither run collided, but the ablated run braked {delay_steps} step(s) "
                f"later than the baseline (first brake/coast at step {ablated_brake_idx} vs "
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

    return {
        "scenario": scenario5.name,
        "collision_delta": collision_delta,
        "baseline": {
            "wheel_slip_ablated": False,
            "collision": baseline_res.collision,
            "contact_time": baseline_res.contact_time,
            "contact_velocity": baseline_res.contact_velocity,
            "first_brake_or_coast_step": baseline_brake_idx,
            "trace": _trace_as_dict(baseline_res),
        },
        "ablated": {
            "wheel_slip_ablated": True,
            "collision": ablated_res.collision,
            "contact_time": ablated_res.contact_time,
            "contact_velocity": ablated_res.contact_velocity,
            "first_brake_or_coast_step": ablated_brake_idx,
            "trace": _trace_as_dict(ablated_res),
        },
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

    controller_factories = {
        "HIRAM (Bayesian C)": HiramCKernelEngine,
        "Naive Threshold": NaiveThresholdEngine,
        "Conservative Kinematic": ConservativeKinematicEngine,
        "Always-Brake": AlwaysBrakeEngine,
    }

    results: Dict[str, ControllerScorecard] = {}

    print("\n" + "=" * 80)
    print("      HIRAM 4-WAY COMPARATIVE EFFICACY BENCHMARK (TASK 3.2)")
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
    print("      WHEEL-SLIP ABLATION STUDY (Scenario 5)")
    print("=" * 80)
    ablation = run_wheel_slip_ablation(scenarios)
    print(f"Scenario: {ablation['scenario']}")
    print(
        f"  Baseline : collision={ablation['baseline']['collision']} "
        f"contact_time={ablation['baseline']['contact_time']} "
        f"contact_velocity={ablation['baseline']['contact_velocity']}"
    )
    print(
        f"  Ablated  : collision={ablation['ablated']['collision']} "
        f"contact_time={ablation['ablated']['contact_time']} "
        f"contact_velocity={ablation['ablated']['contact_velocity']}"
    )
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
        "# HIRAM 4-Way Comparative Efficacy Benchmark (Task 3.2)",
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
            v_str = ", ".join(f"{v:.3f} m/s" for v in sc.impact_velocities)
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
        f"contact_velocity={ablation['baseline']['contact_velocity']}",
        f"- Ablated (WheelSlipObs forced unobserved): collision={ablation['ablated']['collision']}, "
        f"contact_time={ablation['ablated']['contact_time']}, "
        f"contact_velocity={ablation['ablated']['contact_velocity']}",
        "",
        f"**Finding:** {ablation['finding']}",
        "",
        "Note: timing figures throughout this report are observed host wall-clock "
        "execution times of the Python/ctypes call path (`time.perf_counter()` around "
        "`step_c_kernel()`), not a hard worst-case-execution-time (WCET) bound for the "
        "on-target Cortex-M7 kernel.",
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