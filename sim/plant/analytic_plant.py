"""
Closed-form analytic reference plant.

Models a single emergency-braking maneuver in up to four exact kinematic
phases: pre-command coast, actuator-delay phase under commanded acceleration
a_cmd (with early-halt handling if a_cmd < 0), braking phase under
-braking_deceleration, and rest. Evaluated directly from equations of
motion without numerical time-stepping.
"""

import math
from typing import List, NamedTuple, Tuple

from . import PlantParameters, PlantState


class PlantOracleContractError(ValueError):
    """Raised when oracle inputs violate physical bounds or contracts."""
    pass


class PlantOracleConvergenceError(RuntimeError):
    """Raised when numerical integrator fails to converge within safety limits."""
    pass


def _validate_inputs(
    state: PlantState, params: PlantParameters, command_time: float = 0.0
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


class _Plan(NamedTuple):
    x1: float
    halted_in_delay: bool
    t_zero: float
    x_halt: float
    v_d: float
    x_d: float
    braking: bool
    t_brake: float
    x_stop: float
    t_stop: float


def _build_plan(
    initial_state: PlantState, params: PlantParameters, command_time: float
) -> _Plan:
    _validate_inputs(initial_state, params, command_time)
    v0 = initial_state.velocity
    x1 = initial_state.position + v0 * command_time
    a_cmd = params.commanded_acceleration
    tau = params.actuator_delay
    b = params.braking_deceleration

    # If starting at rest and commanded acceleration is non-positive, cart remains at rest
    if v0 == 0.0 and a_cmd <= 0.0:
        return _Plan(
            x1=x1,
            halted_in_delay=True,
            t_zero=0.0,
            x_halt=x1,
            v_d=0.0,
            x_d=x1,
            braking=False,
            t_brake=0.0,
            x_stop=x1,
            t_stop=command_time,
        )

    # If moving forward and braking/decelerating during delay
    if a_cmd < 0.0 and v0 > 0.0:
        t_zero = -v0 / a_cmd
        if t_zero < tau:
            x_halt = x1 - (v0 * v0) / (2.0 * a_cmd)
            return _Plan(
                x1=x1,
                halted_in_delay=True,
                t_zero=t_zero,
                x_halt=x_halt,
                v_d=0.0,
                x_d=x_halt,
                braking=False,
                t_brake=0.0,
                x_stop=x_halt,
                t_stop=command_time + t_zero,
            )

    v_d = v0 + a_cmd * tau
    if v_d < 0.0:
        v_d = 0.0
    x_d = x1 + v0 * tau + 0.5 * a_cmd * tau * tau

    if v_d <= 0.0:
        return _Plan(
            x1=x1,
            halted_in_delay=False,
            t_zero=tau,
            x_halt=x_d,
            v_d=0.0,
            x_d=x_d,
            braking=False,
            t_brake=0.0,
            x_stop=x_d,
            t_stop=command_time + tau,
        )

    t_brake = v_d / b
    x_stop = x_d + (v_d * v_d) / (2.0 * b)
    return _Plan(
        x1=x1,
        halted_in_delay=False,
        t_zero=tau,
        x_halt=x_d,
        v_d=v_d,
        x_d=x_d,
        braking=True,
        t_brake=t_brake,
        x_stop=x_stop,
        t_stop=command_time + tau + t_brake,
    )


def stopping_time(
    initial_state: PlantState,
    params: PlantParameters,
    command_time: float = 0.0,
) -> float:
    return _build_plan(initial_state, params, command_time).t_stop


def stopping_position(
    initial_state: PlantState, params: PlantParameters, command_time: float = 0.0
) -> float:
    return _build_plan(initial_state, params, command_time).x_stop


def state_at(
    t: float,
    initial_state: PlantState,
    params: PlantParameters,
    command_time: float = 0.0,
) -> PlantState:
    """Exact kinematic state at time t using right-continuous phase acceleration."""
    _validate_inputs(initial_state, params, command_time)
    if not math.isfinite(t) or t < 0.0:
        raise PlantOracleContractError(f"evaluation time t must be finite and >= 0, got {t}")

    v0 = initial_state.velocity
    x0 = initial_state.position

    if t < command_time:
        return PlantState(position=x0 + v0 * t, velocity=v0, acceleration=0.0)

    plan = _build_plan(initial_state, params, command_time)
    local = t - command_time
    a_cmd = params.commanded_acceleration
    tau = params.actuator_delay

    if plan.halted_in_delay:
        if local < plan.t_zero:
            x = plan.x1 + v0 * local + 0.5 * a_cmd * local * local
            v = v0 + a_cmd * local
            return PlantState(position=x, velocity=max(0.0, v), acceleration=a_cmd)
        return PlantState(position=plan.x_halt, velocity=0.0, acceleration=0.0)

    if local < tau:
        x = plan.x1 + v0 * local + 0.5 * a_cmd * local * local
        v = v0 + a_cmd * local
        return PlantState(position=x, velocity=max(0.0, v), acceleration=a_cmd)

    if not plan.braking:
        return PlantState(position=plan.x_d, velocity=0.0, acceleration=0.0)

    local2 = local - tau
    if local2 < plan.t_brake:
        b = params.braking_deceleration
        v = plan.v_d - b * local2
        x = plan.x_d + plan.v_d * local2 - 0.5 * b * local2 * local2
        return PlantState(position=x, velocity=max(0.0, v), acceleration=-b)

    return PlantState(position=plan.x_stop, velocity=0.0, acceleration=0.0)


def simulate(
    initial_state: PlantState,
    params: PlantParameters,
    dt: float = 0.001,
    command_time: float = 0.0,
    margin: float = 0.05,
    max_samples_per_phase: int = 200,
) -> List[Tuple[float, PlantState]]:
    if not math.isfinite(dt) or dt <= 0.0:
        raise PlantOracleContractError(f"dt must be finite and > 0, got {dt}")
    if not math.isfinite(margin) or margin < 0.0:
        raise PlantOracleContractError(f"margin must be finite and >= 0, got {margin}")

    plan = _build_plan(initial_state, params, command_time)
    delay_end = command_time + params.actuator_delay
    t_stop = plan.t_stop
    t_end = t_stop + margin

    def sample_span(t_lo: float, t_hi: float) -> List[float]:
        duration = t_hi - t_lo
        if duration <= 0.0:
            return []
        n = min(max(1, math.ceil(duration / dt)), max_samples_per_phase)
        return [t_lo + duration * k / n for k in range(1, n + 1)]

    times = [0.0]
    if command_time > 0.0:
        times += sample_span(0.0, min(command_time, t_stop))
    delay_hi = min(delay_end, t_stop)
    if delay_hi > command_time:
        times += sample_span(command_time, delay_hi)
    times += sample_span(delay_hi, t_stop)
    times += sample_span(t_stop, t_end)

    return [(t, state_at(t, initial_state, params, command_time)) for t in times]