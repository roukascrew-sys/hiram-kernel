"""
HIRAM Safety Kernel - Task 2.3: Closed-Loop Trajectory Evaluator
Simulates discrete-time 1D cart longitudinal kinematics, continuous-to-discrete sensor
evidence extraction, and closed-loop control integration with BayesianInferenceEngine.
Target: Pure Python 3 standard library reference simulation.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple

from tools.inference_engine import BayesianInferenceEngine


def solve_step_kinematics_and_contact(
    x0: float,
    v0: float,
    a_cmd: float,
    dt: float,
    v_cruise: float = 0.50,
    obstacle_x: Optional[float] = None,
) -> Tuple[float, float, bool, Optional[float], Optional[float]]:
    """
    Simulates longitudinal point-mass kinematics over interval dt with cruise speed clamping
    and exact intra-step contact detection against obstacle_x.

    Returns:
        (x_next, v_next, collision, contact_dt, contact_v)
    """
    # 0. Immediate boundary check: initial contact or initial penetration
    if obstacle_x is not None and x0 >= obstacle_x:
        # Starting at or beyond the obstacle boundary is immediate contact at t=0
        return obstacle_x, v0, True, 0.0, v0

    # 1. Kinematic trajectory segmentation over dt
    if a_cmd < 0.0:
        dt_stop = -v0 / a_cmd if a_cmd != 0 else float("inf")
        if dt_stop <= dt:
            dt1 = dt_stop
            dx1 = - (v0 * v0) / (2.0 * a_cmd)
            # Phase 2: cart brought to rest
            x_next = x0 + dx1
            v_next = 0.0
        else:
            dt1 = dt
            dx1 = v0 * dt + 0.5 * a_cmd * dt * dt
            x_next = x0 + dx1
            v_next = v0 + a_cmd * dt
        dx_total = x_next - x0

    elif a_cmd > 0.0:
        if v0 >= v_cruise:
            dt1 = 0.0
            dx1 = 0.0
            x_next = x0 + v_cruise * dt
            v_next = v_cruise
        else:
            dt_cruise = (v_cruise - v0) / a_cmd
            if dt_cruise <= dt:
                dt1 = dt_cruise
                dx1 = (v_cruise * v_cruise - v0 * v0) / (2.0 * a_cmd)
                # Phase 2: cruising at v_cruise
                dt2 = dt - dt1
                dx2 = v_cruise * dt2
                x_next = x0 + dx1 + dx2
                v_next = v_cruise
            else:
                dt1 = dt
                dx1 = v0 * dt + 0.5 * a_cmd * dt * dt
                x_next = x0 + dx1
                v_next = v0 + a_cmd * dt
        dx_total = x_next - x0

    else:
        # a_cmd == 0.0 (pure coasting / constant velocity)
        dt1 = dt
        dx1 = v0 * dt
        x_next = x0 + dx1
        v_next = v0
        dx_total = dx1

    # 2. Contact resolution against obstacle_x
    if obstacle_x is None:
        return x_next, v_next, False, None, None

    d = obstacle_x - x0
    if d > dx_total:
        # Step trajectory did not reach the obstacle boundary
        return x_next, v_next, False, None, None

    # Step breached obstacle: solve piecewise collision kinematics
    if a_cmd < 0.0:
        v_contact = math.sqrt(max(0.0, v0 * v0 + 2.0 * a_cmd * d))
        contact_dt = (v_contact - v0) / a_cmd

    elif a_cmd > 0.0:
        if v0 >= v_cruise:
            v_contact = v_cruise
            contact_dt = d / v_cruise
        elif d <= dx1:
            v_contact = math.sqrt(v0 * v0 + 2.0 * a_cmd * d)
            contact_dt = (v_contact - v0) / a_cmd
        else:
            v_contact = v_cruise
            contact_dt = dt1 + (d - dx1) / v_cruise

    else:
        v_contact = v0
        contact_dt = d / v0 if v0 > 0 else 0.0

    return x_next, v_next, True, contact_dt, v_contact


class TrajectoryResult:
    def __init__(
        self,
        time_history: List[float],
        x_history: List[float],
        v_history: List[float],
        a_history: List[float],
        action_history: List[str],
        collision: bool,
        contact_time: Optional[float],
        contact_velocity: Optional[float],
        impact_energy: Optional[float],
        final_x: float,
        final_v: float,
        total_time: float,
    ) -> None:
        self.time_history = time_history
        self.x_history = x_history
        self.v_history = v_history
        self.a_history = a_history
        self.action_history = action_history
        self.collision = collision
        self.contact_time = contact_time
        self.contact_velocity = contact_velocity
        self.impact_energy = impact_energy
        self.final_x = final_x
        self.final_v = final_v
        self.total_time = total_time


class TrajectoryEvaluator:
    def __init__(
        self,
        engine: BayesianInferenceEngine,
        dt: float = 0.010,
        v_cruise: float = 0.50,
        a_drive: float = +0.25,
        a_drag: float = -0.05,
        detection_threshold: float = 0.300,
        slip_threshold: float = 0.15,
        cart_mass: float = 5.0,
    ) -> None:
        self.engine = engine
        self.dt = dt
        self.v_cruise = v_cruise
        self.a_drive = a_drive
        self.a_drag = a_drag
        self.detection_threshold = detection_threshold
        self.slip_threshold = slip_threshold
        self.cart_mass = cart_mass

        # Base nominal deceleration capability by surface condition
        self.brake_decel_table = {
            "DRY": 1.20,
            "WET": 0.80,
            "ICY": 0.35,
        }

    def _get_braking_deceleration(
        self, track_condition: str, decel_capability: str, override_brake_decel: Optional[float] = None
    ) -> float:
        if override_brake_decel is not None:
            return override_brake_decel

        base = self.brake_decel_table.get(track_condition, 1.20)
        if decel_capability == "NOMINAL":
            return base
        elif decel_capability == "DEGRADED":
            return 0.60 * base
        elif decel_capability == "CRITICAL":
            return 0.20 * base
        return base

    def extract_evidence(
        self,
        x: float,
        v: float,
        a_current: float,
        obstacle_x: Optional[float],
        track_condition: str,
        decel_capability: str,
        t: float,
        dropout_window: Optional[Tuple[float, float]] = None,
        force_slip_high: bool = False,
    ) -> Dict[str, Optional[int]]:
        """
        Transforms continuous plant states into discrete Bayesian evidence literals.
        Range sensor blackout windows yield None for range sensors.
        """
        in_range_dropout = (
            dropout_window is not None
            and (dropout_window[0] <= t <= dropout_window[1])
        )

        if in_range_dropout:
            lidar_obs = None
            tof_obs = None
        else:
            if obstacle_x is not None:
                clearance = obstacle_x - x
                is_detected = 1 if clearance <= self.detection_threshold else 0
            else:
                is_detected = 0
            lidar_obs = is_detected
            tof_obs = is_detected

        if force_slip_high:
            wheel_slip_obs = 1
        else:
            if a_current < 0 and (track_condition in ("WET", "ICY") or decel_capability != "NOMINAL"):
                slip_ratio = 0.25
            else:
                slip_ratio = 0.02
            wheel_slip_obs = 1 if slip_ratio >= self.slip_threshold else 0

        return {
            "LidarObs": lidar_obs,
            "TofObs": tof_obs,
            "WheelSlipObs": wheel_slip_obs,
        }

    def run_episode(
        self,
        initial_x: float = 0.0,
        initial_v: float = 0.0,
        max_duration: float = 5.0,
        obstacle_x: Optional[float] = None,
        track_condition: str = "DRY",
        decel_capability: str = "NOMINAL",
        dropout_window: Optional[Tuple[float, float]] = None,
        override_brake_decel: Optional[float] = None,
        force_slip_high: bool = False,
    ) -> TrajectoryResult:
        """
        Executes a closed-loop discrete-time simulation episode.
        Terminates immediately at first contact or when cart stops.
        """
        t = 0.0
        x = initial_x
        v = initial_v
        a = 0.0

        t_hist: List[float] = [t]
        x_hist: List[float] = [x]
        v_hist: List[float] = [v]
        a_hist: List[float] = [a]
        action_hist: List[str] = []

        # Check for immediate starting contact/penetration
        if obstacle_x is not None and initial_x >= obstacle_x:
            impact_e = 0.5 * self.cart_mass * (initial_v * initial_v)
            return TrajectoryResult(
                time_history=t_hist,
                x_history=x_hist,
                v_history=v_hist,
                a_history=a_hist,
                action_history=["EMERGENCY_BRAKE"],
                collision=True,
                contact_time=0.0,
                contact_velocity=initial_v,
                impact_energy=impact_e,
                final_x=obstacle_x,
                final_v=initial_v,
                total_time=0.0,
            )

        a_brake_mag = self._get_braking_deceleration(
            track_condition, decel_capability, override_brake_decel
        )

        steps = int(math.ceil(max_duration / self.dt))
        for _ in range(steps):
            # 1. Sample continuous evidence
            evidence = self.extract_evidence(
                x, v, a, obstacle_x, track_condition, decel_capability, t, dropout_window, force_slip_high
            )

            # 2. Kernel decision execution
            step_out = self.engine.step(evidence)
            action = (
                step_out.get("selected_action")
                or step_out.get("best_action_name")
                or step_out.get("action")
                or step_out.get("best_action")
            )
            action_hist.append(action)

            # 3. Command acceleration translation
            if action == "ACCEL":
                a_cmd = self.a_drive
            elif action == "COAST":
                a_cmd = self.a_drag
            elif action == "EMERGENCY_BRAKE":
                a_cmd = -a_brake_mag
            else:
                a_cmd = -a_brake_mag

            # 4. Advance kinematics and check for obstacle collision
            x_next, v_next, step_coll, c_dt, c_v = solve_step_kinematics_and_contact(
                x0=x,
                v0=v,
                a_cmd=a_cmd,
                dt=self.dt,
                v_cruise=self.v_cruise,
                obstacle_x=obstacle_x,
            )

            if step_coll:
                # Freeze state and terminate at exact first contact point
                contact_time = t + c_dt
                contact_velocity = c_v
                impact_energy = 0.5 * self.cart_mass * (c_v * c_v)

                t_hist.append(contact_time)
                x_hist.append(obstacle_x)
                v_hist.append(c_v)
                a_hist.append(a_cmd)

                return TrajectoryResult(
                    time_history=t_hist,
                    x_history=x_hist,
                    v_history=v_hist,
                    a_history=a_hist,
                    action_history=action_hist,
                    collision=True,
                    contact_time=contact_time,
                    contact_velocity=contact_velocity,
                    impact_energy=impact_energy,
                    final_x=obstacle_x,
                    final_v=c_v,
                    total_time=contact_time,
                )

            # State update
            t += self.dt
            x = x_next
            v = v_next
            a = a_cmd

            t_hist.append(t)
            x_hist.append(x)
            v_hist.append(v)
            a_hist.append(a)

            # Terminate when brought to rest after motion has occurred
            if v <= 0.0 and a_cmd < 0 and (x > 1e-5 or t > 0.05):
                break

        return TrajectoryResult(
            time_history=t_hist,
            x_history=x_hist,
            v_history=v_hist,
            a_history=a_hist,
            action_history=action_hist,
            collision=False,
            contact_time=None,
            contact_velocity=None,
            impact_energy=None,
            final_x=x,
            final_v=v,
            total_time=t,
        )