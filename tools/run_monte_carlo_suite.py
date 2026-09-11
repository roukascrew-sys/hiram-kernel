"""
tools/run_monte_carlo_suite.py

Statistically rigorous 1,200-episode comparative Monte Carlo suite: evaluates
all 5 controllers from tools/run_comparative_benchmark.py (HIRAM Bayesian C,
HIRAM Ablated: No Wheel-Slip, Naive Threshold, Conservative Rule-Based,
Always-Brake) across 12 operational strata x 100 randomized-but-reproducible
episodes each = 1,200 episodes per controller (6,000 episode-controller runs
total).

Relationship to sim/scenarios/: sim/scenarios/strata.py + generator.py define
a 12-stratum canonical dataset (NOMINAL_1..6 / SHIFT_1..6) for a *different*
physical model (sim.plant.analytic_plant's continuous-time actuator-delay /
sensor-bias / sensor-noise plant, addressed by ScenarioInstance's
braking_deceleration/actuator_delay/sensor_bias/sensor_noise_scale fields).
That model has no notion of the discrete track_condition/decel_capability
evidence surface tools.trajectory_evaluator.TrajectoryEvaluator and this
repo's 5 comparative controllers actually consume, so its ScenarioInstance
records cannot be fed to them directly. This suite instead defines its own
12-stratum design (STRATA below) directly over TrajectoryEvaluator's
parameter surface (v0, obstacle_x, track_condition, decel_capability,
dropout_window), covering every sweep range this task specifies, and reuses
sim.scenarios.strata.derive_domain_seed -- the same SHA-256-based, per-
(stratum, episode, domain) seed derivation sim/scenarios/ already uses --
for cross-run reproducibility and cross-stratum/cross-episode independence,
rather than reinventing seed derivation.

Performance: each (stratum, controller) pair's 100 episodes run as one
multiprocessing.Pool task (12 strata x 5 controllers = 60 tasks), so a
controller's model/library is loaded once per task rather than once per
episode. No wall-clock timing is recorded anywhere in this suite (Astra
requirement: "measure pure trajectory outcomes independently of OS timing
jitter") -- see tools/run_comparative_benchmark.py for latency measurement,
which is a separate, smaller-scale benchmark.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.scenarios.strata import derive_domain_seed
from tools.run_comparative_benchmark import (
    AlwaysBrakeEngine,
    ConservativeRuleBasedEngine,
    HiramCKernelEngine,
    NaiveThresholdEngine,
    WheelSlipAblatedHiramEngine,
)
from tools.trajectory_evaluator import TrajectoryEvaluator

SEED = 42
EPISODES_PER_STRATUM = 100
MAX_EPISODE_DURATION_S = 3.0

BUILD_DIR = REPO_ROOT / "build"
JSON_RESULTS_PATH = BUILD_DIR / "monte_carlo_results.json"

# Registered by name (not by class object) so multiprocessing worker
# processes on Windows (spawn start method: each worker re-imports this
# module fresh) look the factory up locally rather than needing the class
# itself pickled across the process boundary.
CONTROLLER_FACTORIES = {
    "HIRAM (Bayesian C)": HiramCKernelEngine,
    "HIRAM (Ablated: No Wheel-Slip)": WheelSlipAblatedHiramEngine,
    "Naive Threshold": NaiveThresholdEngine,
    "Conservative Rule-Based": ConservativeRuleBasedEngine,
    "Always-Brake": AlwaysBrakeEngine,
}


@dataclass(frozen=True)
class StratumSpec:
    name: str
    v0_range: Tuple[float, float]
    clear_track: bool  # True => obstacle_x is always None for this stratum
    obstacle_x_range: Tuple[float, float]
    track_options: Tuple[str, ...]
    decel_options: Tuple[str, ...]
    has_dropout: bool
    dropout_start_range: Tuple[float, float]
    dropout_duration_range: Tuple[float, float]
    degraded_friction: bool  # tag: WET track and/or DEGRADED decel present in this stratum's sampling


# 12 strata spanning every sweep range this task specifies (v0 0.0-0.50,
# obstacle_x 0.15-0.65 with None for clear-track strata, dropout_start
# 0.0-1.5 / duration 0.1-0.8, track {DRY, WET}, decel {NOMINAL, DEGRADED}),
# arranged in the same nominal-then-shift severity progression
# sim/scenarios/strata.py's StratumType uses, tagged for the "High-Dropout"
# / "Degraded-Friction" report groupings in section 5 of this suite's output.
STRATA: List[StratumSpec] = [
    StratumSpec("1. Nominal Clear Dry", (0.0, 0.50), True, (0.15, 0.65), ("DRY",), ("NOMINAL",), False, (0.0, 0.0), (0.0, 0.0), False),
    StratumSpec("2. Nominal Clear Wet", (0.0, 0.50), True, (0.15, 0.65), ("WET",), ("NOMINAL",), False, (0.0, 0.0), (0.0, 0.0), True),
    StratumSpec("3. Nominal Obstacle Dry", (0.0, 0.50), False, (0.15, 0.65), ("DRY",), ("NOMINAL",), False, (0.0, 0.0), (0.0, 0.0), False),
    StratumSpec("4. Nominal Obstacle Wet", (0.0, 0.50), False, (0.15, 0.65), ("WET",), ("NOMINAL",), False, (0.0, 0.0), (0.0, 0.0), True),
    StratumSpec("5. Nominal Degraded Braking", (0.0, 0.50), False, (0.15, 0.65), ("DRY",), ("DEGRADED",), False, (0.0, 0.0), (0.0, 0.0), True),
    StratumSpec("6. Nominal Tight Clearance", (0.0, 0.50), False, (0.15, 0.30), ("DRY", "WET"), ("NOMINAL", "DEGRADED"), False, (0.0, 0.0), (0.0, 0.0), True),
    StratumSpec("7. Shift Wet+Degraded Braking", (0.0, 0.50), False, (0.15, 0.65), ("WET",), ("DEGRADED",), False, (0.0, 0.0), (0.0, 0.0), True),
    StratumSpec("8. Shift Short Dropout", (0.0, 0.50), False, (0.15, 0.65), ("DRY",), ("NOMINAL",), True, (0.0, 0.5), (0.1, 0.3), False),
    StratumSpec("9. Shift Long Dropout", (0.0, 0.50), False, (0.15, 0.65), ("DRY",), ("NOMINAL",), True, (0.0, 0.5), (0.5, 0.8), False),
    StratumSpec("10. Shift Late Dropout", (0.0, 0.50), False, (0.15, 0.65), ("DRY",), ("NOMINAL",), True, (1.0, 1.5), (0.1, 0.4), False),
    StratumSpec("11. Shift Dropout+Wet+Degraded", (0.0, 0.50), False, (0.15, 0.65), ("WET",), ("DEGRADED",), True, (0.0, 0.3), (0.3, 0.8), True),
    StratumSpec("12. Shift Worst-Case Compound", (0.0, 0.50), False, (0.15, 0.30), ("WET",), ("DEGRADED",), True, (0.0, 0.2), (0.4, 0.8), True),
]

assert len(STRATA) == 12, f"expected 12 strata, got {len(STRATA)}"
_STRATA_BY_NAME = {s.name: s for s in STRATA}


def _sample_episode_params(spec: StratumSpec, rng) -> Dict[str, Any]:
    v0 = rng.uniform(*spec.v0_range)
    obstacle_x = None if spec.clear_track else rng.uniform(*spec.obstacle_x_range)
    track = rng.choice(spec.track_options)
    decel = rng.choice(spec.decel_options)
    if spec.has_dropout:
        start = rng.uniform(*spec.dropout_start_range)
        duration = rng.uniform(*spec.dropout_duration_range)
        dropout_window = (start, min(start + duration, MAX_EPISODE_DURATION_S))
    else:
        dropout_window = None
    return {
        "initial_x": 0.0,
        "initial_v": v0,
        "max_duration": MAX_EPISODE_DURATION_S,
        "obstacle_x": obstacle_x,
        "track_condition": track,
        "decel_capability": decel,
        "dropout_window": dropout_window,
    }


def _run_stratum_controller_task(args: Tuple[str, str]) -> Dict[str, Any]:
    """One multiprocessing.Pool task: all EPISODES_PER_STRATUM episodes of
    one (stratum, controller) pair. Constructs the controller's engine once
    (not once per episode) and runs every episode through the same engine
    instance and a fresh TrajectoryEvaluator, mirroring
    tools/run_comparative_benchmark.py's per-scenario evaluation loop but
    with no timing instrumentation -- see module docstring."""
    stratum_name, controller_name = args
    spec = _STRATA_BY_NAME[stratum_name]
    engine = CONTROLLER_FACTORIES[controller_name]()
    evaluator = TrajectoryEvaluator(engine=engine)

    episodes: List[Dict[str, Any]] = []
    for episode_id in range(EPISODES_PER_STRATUM):
        seed = derive_domain_seed(str(SEED), stratum_name, episode_id, "monte_carlo")
        rng = random.Random(seed)
        params = _sample_episode_params(spec, rng)

        res = evaluator.run_episode(
            initial_x=params["initial_x"],
            initial_v=params["initial_v"],
            max_duration=params["max_duration"],
            obstacle_x=params["obstacle_x"],
            track_condition=params["track_condition"],
            decel_capability=params["decel_capability"],
            dropout_window=params["dropout_window"],
        )

        is_clear_track = params["obstacle_x"] is None
        nuisance_stop = is_clear_track and ("EMERGENCY_BRAKE" in res.action_history)

        episodes.append({
            "episode_id": episode_id,
            "seed": seed,
            "v0": params["initial_v"],
            "obstacle_x": params["obstacle_x"],
            "track_condition": params["track_condition"],
            "decel_capability": params["decel_capability"],
            "dropout_window": list(params["dropout_window"]) if params["dropout_window"] else None,
            "collision": res.collision,
            "contact_time": res.contact_time,
            "contact_velocity": res.contact_velocity,
            "final_x": res.final_x,
            "progress_m": res.final_x - params["initial_x"],
            "is_clear_track": is_clear_track,
            "nuisance_stop": nuisance_stop,
        })

    return {"stratum": stratum_name, "controller": controller_name, "episodes": episodes}


def _aggregate(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Statistical Metrics: collision rate, mean contact velocity (over
    colliding episodes only), nuisance/false-positive halt rate (over
    clear-track episodes only), mean progress, and an Efficiency Ratio
    defined as mean_progress_m * (1 - collision_rate_fraction) -- progress
    achieved, discounted by the fraction of episodes that ended in
    collision (0 at a 100% collision rate, equal to mean progress at a 0%
    collision rate)."""
    n = len(episodes)
    if n == 0:
        return {
            "episodes_evaluated": 0, "collision_rate_pct": None, "mean_contact_velocity_mps": None,
            "nuisance_halt_rate_pct": None, "mean_progress_m": None, "efficiency_ratio": None,
        }

    collisions = [e for e in episodes if e["collision"]]
    collision_rate = len(collisions) / n
    contact_velocities = [e["contact_velocity"] for e in collisions if e["contact_velocity"] is not None]
    mean_contact_velocity = (sum(contact_velocities) / len(contact_velocities)) if contact_velocities else None

    clear_track_episodes = [e for e in episodes if e["is_clear_track"]]
    if clear_track_episodes:
        nuisance_rate = sum(1 for e in clear_track_episodes if e["nuisance_stop"]) / len(clear_track_episodes)
    else:
        nuisance_rate = None

    mean_progress = sum(e["progress_m"] for e in episodes) / n
    efficiency_ratio = mean_progress * (1.0 - collision_rate)

    return {
        "episodes_evaluated": n,
        "collision_rate_pct": collision_rate * 100.0,
        "mean_contact_velocity_mps": mean_contact_velocity,
        "nuisance_halt_rate_pct": (nuisance_rate * 100.0) if nuisance_rate is not None else None,
        "mean_progress_m": mean_progress,
        "efficiency_ratio": efficiency_ratio,
    }


def run_suite() -> Dict[str, Any]:
    tasks = [(s.name, c) for s in STRATA for c in CONTROLLER_FACTORIES]
    assert len(tasks) == 12 * len(CONTROLLER_FACTORIES)

    n_workers = min(len(tasks), os.cpu_count() or 4)
    print(f"Running {len(tasks)} (stratum, controller) tasks "
          f"({len(STRATA)} strata x {len(CONTROLLER_FACTORIES)} controllers x "
          f"{EPISODES_PER_STRATUM} episodes = {len(STRATA) * EPISODES_PER_STRATUM * len(CONTROLLER_FACTORIES)} "
          f"episode-controller runs) across {n_workers} worker processes...")

    with multiprocessing.Pool(processes=n_workers) as pool:
        task_results = pool.map(_run_stratum_controller_task, tasks)

    # controller -> stratum -> {episodes: [...], aggregate: {...}}
    by_controller: Dict[str, Dict[str, Any]] = {c: {} for c in CONTROLLER_FACTORIES}
    for tr in task_results:
        by_controller[tr["controller"]][tr["stratum"]] = {
            "episodes": tr["episodes"],
            "aggregate": _aggregate(tr["episodes"]),
        }

    dropout_strata = [s.name for s in STRATA if s.has_dropout]
    degraded_strata = [s.name for s in STRATA if s.degraded_friction]

    report_groups = {
        "All Strata Aggregate": [s.name for s in STRATA],
        "High-Dropout Strata": dropout_strata,
        "Degraded-Friction Strata": degraded_strata,
    }

    group_summaries: Dict[str, Dict[str, Any]] = {}
    for group_name, stratum_names in report_groups.items():
        group_summaries[group_name] = {}
        for controller_name in CONTROLLER_FACTORIES:
            pooled_episodes: List[Dict[str, Any]] = []
            for sn in stratum_names:
                pooled_episodes.extend(by_controller[controller_name][sn]["episodes"])
            group_summaries[group_name][controller_name] = _aggregate(pooled_episodes)

    return {
        "seed": SEED,
        "episodes_per_stratum": EPISODES_PER_STRATUM,
        "total_episodes_per_controller": len(STRATA) * EPISODES_PER_STRATUM,
        "strata": [
            {
                "name": s.name, "has_dropout": s.has_dropout, "degraded_friction": s.degraded_friction,
                "clear_track": s.clear_track,
            }
            for s in STRATA
        ],
        "report_groups": {g: names for g, names in report_groups.items()},
        "efficiency_ratio_formula": "mean_progress_m * (1 - collision_rate_fraction)",
        "by_controller": by_controller,
        "group_summaries": group_summaries,
    }


def _fmt(v: Optional[float], fmt: str) -> str:
    return "N/A" if v is None else format(v, fmt)


def _print_group_table(group_name: str, group_summary: Dict[str, Any]) -> None:
    print(f"\n{group_name}")
    print("-" * 100)
    print(f"{'Controller':<32} | {'Collision %':<12} | {'Mean Contact v (m/s)':<21} | "
          f"{'Nuisance %':<11} | {'Mean Progress (m)':<18} | {'Efficiency Ratio'}")
    print("-" * 100)
    for controller_name, agg in group_summary.items():
        print(
            f"{controller_name:<32} | "
            f"{_fmt(agg['collision_rate_pct'], '.2f'):<12} | "
            f"{_fmt(agg['mean_contact_velocity_mps'], '.4f'):<21} | "
            f"{_fmt(agg['nuisance_halt_rate_pct'], '.2f'):<11} | "
            f"{_fmt(agg['mean_progress_m'], '.4f'):<18} | "
            f"{_fmt(agg['efficiency_ratio'], '.4f')}"
        )


def main() -> Dict[str, Any]:
    payload = run_suite()

    print("\n" + "=" * 100)
    print(f"      HIRAM MONTE CARLO SUITE -- {payload['total_episodes_per_controller']} episodes/controller "
          f"({len(STRATA)} strata x {EPISODES_PER_STRATUM}), seed={SEED}")
    print("=" * 100)
    for group_name, group_summary in payload["group_summaries"].items():
        _print_group_table(group_name, group_summary)
    print("\n" + "=" * 100)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    JSON_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {JSON_RESULTS_PATH}")

    return payload


if __name__ == "__main__":
    main()
