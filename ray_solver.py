"""Ray tracing through the vertically stratified atmosphere n(y).

Ray equation for n = n(y), with p = dy/dx:

    dy/dx = p
    dp/dx = (1 / n(y)) * (dn/dy) * (1 + p**2)

The atmospheric model lives in atmosphere.py and is imported, not rewritten.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from atmosphere import AtmosphereParams, dn_dy, refractive_index

__all__ = ["RayPath", "trace_ray"]

_RTOL = 1e-10
_ATOL = 1e-12


@dataclass(frozen=True)
class RayPath:
    """Result of a ray trace.

    Attributes:
        x_m: Horizontal positions [m].
        y_m: Ray heights [m].
        slope: dy/dx at each point.
        hit_ground: True if the ray reached y = 0 (integration stopped there).
    """

    x_m: np.ndarray
    y_m: np.ndarray
    slope: np.ndarray
    hit_ground: bool


def _validate_inputs(atmosphere_params, y0_m, theta0_deg, x_max_m, max_step_m, rtol, atol):
    if not isinstance(atmosphere_params, AtmosphereParams):
        raise TypeError("atmosphere_params must be an AtmosphereParams instance")
    for name, value in (
        ("y0_m", y0_m),
        ("theta0_deg", theta0_deg),
        ("x_max_m", x_max_m),
        ("max_step_m", max_step_m),
        ("rtol", rtol),
        ("atol", atol),
    ):
        if not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar, got {value!r}")
    if y0_m <= 0:
        raise ValueError(f"y0_m must be > 0 (start above the ground), got {y0_m!r}")
    if not -90.0 < theta0_deg < 90.0:
        raise ValueError(
            f"theta0_deg must be in (-90, 90) degrees, got {theta0_deg!r}"
        )
    if x_max_m <= 0:
        raise ValueError(f"x_max_m must be > 0, got {x_max_m!r}")
    if max_step_m <= 0:
        raise ValueError(f"max_step_m must be > 0, got {max_step_m!r}")
    if rtol <= 0:
        raise ValueError(f"rtol must be > 0, got {rtol!r}")
    if atol <= 0:
        raise ValueError(f"atol must be > 0, got {atol!r}")


def trace_ray(
    atmosphere_params: AtmosphereParams,
    y0_m: float = 0.30,
    theta0_deg: float = -0.30,
    x_max_m: float = 120.0,
    max_step_m: float = 0.25,
    rtol: float = _RTOL,
    atol: float = _ATOL,
) -> RayPath:
    """Trace a single ray from x = 0 until x_max_m or until it reaches y = 0.

    Args:
        atmosphere_params: Atmosphere model parameters.
        y0_m: Initial height [m], must be > 0.
        theta0_deg: Initial launch angle [deg]; negative points downward.
        x_max_m: Maximum horizontal distance [m].
        max_step_m: Maximum integrator step in x [m].
        rtol: Relative error tolerance passed to solve_ivp.
        atol: Absolute error tolerance passed to solve_ivp.

    Returns:
        RayPath. If the ray reaches the ground, the final point is exactly
        y = 0 and hit_ground is True; otherwise the path ends at x_max_m.

    Raises:
        ValueError / TypeError: on invalid inputs.
        RuntimeError: if the integrator fails or yields non-finite values.
    """
    _validate_inputs(
        atmosphere_params, y0_m, theta0_deg, x_max_m, max_step_m, rtol, atol
    )

    def rhs(_x, state):
        # Clip so RK stage evaluations just below the ground stay valid
        # for the atmosphere model (which rejects y < 0).
        y = max(state[0], 0.0)
        p = state[1]
        n = refractive_index(y, atmosphere_params)
        grad = dn_dy(y, atmosphere_params)
        return [p, (grad / n) * (1.0 + p * p)]

    def ground_event(_x, state):
        return state[0]

    ground_event.terminal = True
    ground_event.direction = -1.0  # only trigger when descending through y = 0

    p0 = float(np.tan(np.deg2rad(theta0_deg)))
    sol = solve_ivp(
        rhs,
        (0.0, float(x_max_m)),
        [float(y0_m), p0],
        method="DOP853",
        events=ground_event,
        max_step=float(max_step_m),
        rtol=float(rtol),
        atol=float(atol),
    )

    if sol.status == -1:
        raise RuntimeError(f"Ray integration failed: {sol.message}")

    x = np.asarray(sol.t, dtype=float)
    y = np.asarray(sol.y[0], dtype=float).copy()
    slope = np.asarray(sol.y[1], dtype=float).copy()

    hit_ground = bool(sol.status == 1 and len(sol.t_events[0]) > 0)
    if hit_ground:
        y[-1] = 0.0  # snap the terminal point onto the ground

    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y)) and np.all(np.isfinite(slope))):
        raise RuntimeError("Ray integration produced non-finite values")

    return RayPath(x_m=x, y_m=y, slope=slope, hit_ground=hit_ground)