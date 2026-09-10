"""
sim.plant - Dual independent reference plant oracles for the stopping benchmark.

This package supplies the ground-truth physics used to grade the safety
benchmark's stopping behaviour. It provides two mathematically independent
simulators of 1D longitudinal cart dynamics under a single emergency-braking
maneuver, commanded at `command_time` against a cart already coasting at
`initial_state.velocity`:

    sim.plant.analytic_plant   closed-form kinematic solution (no integration)
    sim.plant.numerical_plant  fourth-order Runge-Kutta (RK4) numerical integration,
                                with independent (bisection-based) event detection

Both engines model the same physical system:

    Phase 0 (pre-command)  t in [0, command_time):
                            acceleration = 0, cart coasts at initial_state.velocity.
    Phase 1 (actuator delay) t in [command_time, command_time + actuator_delay):
                            acceleration = commanded_acceleration (a_cmd) -- the
                            control interface's last command, still being
                            executed while the emergency-brake actuator lags.
                            If a_cmd < 0, the cart may reach v=0 before the
                            delay window elapses; it then holds at rest for
                            the remainder of the delay (never reverses).
    Phase 2 (braking)      from the end of the delay (if velocity remains)
                            until velocity reaches 0: acceleration =
                            -braking_deceleration.
    Phase 3 (rest)          once velocity has reached 0: held at rest,
                            acceleration = 0.

They deliberately share no solving code -- one is pure algebra, the other is
numerical quadrature with its own independent root-finding for the moment
velocity reaches zero -- so that agreement between them is evidence the plant
model itself is implemented correctly, rather than evidence of a shared bug.
Only this module (data contracts) and the physical constants passed in by the
caller are shared.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlantState:
    """Instantaneous kinematic state of the cart."""

    position: float  # meters (distance to origin)
    velocity: float  # m/s (clamped to >= 0.0 always; v0 < 0.0 is rejected)
    acceleration: float  # m/s^2


@dataclass(frozen=True)
class PlantParameters:
    """Physical/control parameters governing one braking maneuver."""

    actuator_delay: float  # seconds (0.0 to 0.10 s): time from brake command to onset of braking deceleration
    braking_deceleration: float  # m/s^2 (positive scalar, 0.2 to 1.0 m/s^2)
    commanded_acceleration: float = 0.0  # m/s^2 (a_cmd, -0.25 to 0.25): control command still active during actuator_delay
    obstacle_position: float = 3.0  # meters (default: 3.0 m)


__all__ = ["PlantState", "PlantParameters"]
