"""
Numerical reference plant: fixed-step RK4 with independent bisection
zero-crossing detection.

Integrates the 1D cart longitudinal kinematics independently without
closed-form event timing.
"""

import math
from typing import List, Tuple

from . import PlantParameters, PlantState
from .analytic_plant import PlantOracleContractError, PlantOracleConvergenceError

BISECTION_TOLERANCE_S = 1e-9


def _validate_inputs(
    state: PlantState,
    params: PlantParameters,
    command_time: float,
    dt: float,
    margin: float,
    record_every: int,
) -> None:
    for val, name in [
        (state.position, "position"),
        (state.velocity, "velocity"),
        (state.acceleration, "acceleration"),
        (params.commanded_acceleration, "commanded_acceleration"),
        (params.actuator_delay, "actuator_delay"),
        (params.braking_deceleration, "braking_deceleration"),
        (params.obstacle_position, "obstacle_position"),
        (command_time, "command_time"),
        (dt, "dt"),
        (margin, "margin"),
    ]:
        if not math.isfinite(val):
            raise PlantOracleContractError(f"{name} must be finite, got {val}")

    if state.velocity < 0.0:
        raise PlantOracleContractError(
            f"initial velocity must be >= 0.0 m/s, got {state.velocity!r}"
        )
    if command_time < 0.0:
        raise PlantOracleContractError(
            f"command_time must be >= 0.0 s, got {command_time!r}"
        )
    if params.actuator_delay < 0.0:
        raise PlantOracleContractError(
            f"actuator_delay must be >= 0.0 s, got {params.actuator_delay!r}"
        )
    if params.braking_deceleration <= 0.0:
        raise PlantOracleContractError(
            f"braking_deceleration must be > 0.0 m/s^2, got {params.braking_deceleration!r}"
        )
    if dt <= 0.0:
        raise PlantOracleContractError(f"dt must be > 0.0 s, got {dt!r}")
    if margin < 0.0:
        raise PlantOracleContractError(f"margin must be >= 0.0 s, got {margin!r}")
    if record_every < 1:
        raise PlantOracleContractError(
            f"record_every must be >= 1, got {record_every!r}"
        )


def _rk4_step(x: float, v: float, h: float, a: float) -> Tuple[float, float]:
    k1v = a
    k2v = a
    k3v = a
    k4v = a

    k1x = v
    k2x = v + h / 2.0 * k1v
    k3x = v + h / 2.0 * k2v
    k4x = v + h * k3v

    x_next = x + (h / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    v_next = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return x_next, v_next


def _bisect_zero_crossing(
    x0: float, v0: float, a: float, h: float, tol: float = BISECTION_TOLERANCE_S
) -> Tuple[float, float]:
    lo, hi = 0.0, h
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        _, v_mid = _rk4_step(x0, v0, mid, a)
        if v_mid > 0.0:
            lo = mid
        else:
            hi = mid
    x_star, _ = _rk4_step(x0, v0, hi, a)
    return hi, x_star


def simulate(
    initial_state: PlantState,
    params: PlantParameters,
    dt: float = 0.0001,
    command_time: float = 0.0,
    margin: float = 0.05,
    record_every: int = 20,
) -> List[Tuple[float, PlantState]]:
    _validate_inputs(initial_state, params, command_time, dt, margin, record_every)

    x, v, t = initial_state.position, initial_state.velocity, 0.0
    delay_end = command_time + params.actuator_delay

    # Immediate halt check: at rest and commanded acceleration is non-positive
    if v == 0.0 and params.commanded_acceleration <= 0.0:
        return [
            (0.0, PlantState(position=x, velocity=0.0, acceleration=0.0)),
            (margin, PlantState(position=x, velocity=0.0, acceleration=0.0)),
        ]

    init_a = 0.0 if t < command_time else (
        params.commanded_acceleration if t < delay_end else -params.braking_deceleration
    )
    trace: List[Tuple[float, PlantState]] = [
        (t, PlantState(position=x, velocity=v, acceleration=init_a))
    ]

    max_iterations = 20_000_000
    iterations = 0
    step_count = 0

    while iterations < max_iterations:
        iterations += 1

        if t < command_time:
            a, boundary = 0.0, command_time
        elif t < delay_end:
            a, boundary = params.commanded_acceleration, delay_end
        else:
            a, boundary = -params.braking_deceleration, None

        step = dt if boundary is None else min(dt, boundary - t)
        step = max(step, 1e-12)

        x_next, v_next = _rk4_step(x, v, step, a)

        # Catch zero-crossing during deceleration (braking or negative command while moving)
        if a < 0.0 and v > 0.0 and v_next <= 0.0:
            t_star, x_star = _bisect_zero_crossing(x, v, a, step)
            t += t_star
            x = x_star
            trace.append((t, PlantState(position=x, velocity=0.0, acceleration=0.0)))
            trace.append((t + margin, PlantState(position=x, velocity=0.0, acceleration=0.0)))
            return trace

        # Clamp if starting at zero and negative acceleration attempted
        if v == 0.0 and a <= 0.0:
            trace.append((t, PlantState(position=x, velocity=0.0, acceleration=0.0)))
            trace.append((t + margin, PlantState(position=x, velocity=0.0, acceleration=0.0)))
            return trace

        x, v = x_next, v_next
        if v < 0.0:
            v = 0.0
        t += step
        step_count += 1

        if step_count % record_every == 0:
            forward_a = 0.0 if t < command_time else (
                params.commanded_acceleration if t < delay_end else -params.braking_deceleration
            )
            trace.append((t, PlantState(position=x, velocity=v, acceleration=forward_a)))

    raise PlantOracleConvergenceError(
        f"Numerical plant exceeded maximum step limit ({max_iterations}) without halting."
    )