"""Atmospheric refractive-index model for a heated-surface (mirage) setup.

Temperature profile:
    T(y) = T_a + dT * exp(-y / H)

Refractive index (ideal-gas scaling, n - 1 proportional to 1/T):
    n(y) = 1 + (n_ref - 1) * T_ref / T(y)

Vertical gradient:
    dn/dy = (n_ref - 1) * T_ref * dT * exp(-y / H) / (H * T(y)**2)

All functions accept scalars or array-likes. Scalar input returns a Python
float; array input returns a float64 ndarray of the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["AtmosphereParams", "temperature", "refractive_index", "dn_dy"]


@dataclass(frozen=True)
class AtmosphereParams:
    """Parameters of the exponential temperature-excess atmosphere.

    Attributes:
        ambient_temp_K: Far-field (ambient) air temperature T_a [K].
        surface_temp_excess_K: Temperature excess dT at y = 0 [K]. May be zero
            (uniform medium). Negative values (cold surface) are allowed as
            long as the temperature stays positive everywhere.
        scale_height_m: Exponential decay height H [m]. Must be > 0.
        n_ref: Refractive index at the reference temperature.
        reference_temp_K: Temperature T_ref at which n_ref applies [K].
    """

    ambient_temp_K: float = 293.15
    surface_temp_excess_K: float = 40.0
    scale_height_m: float = 0.20
    n_ref: float = 1.000277
    reference_temp_K: float = 293.15

    def __post_init__(self) -> None:
        validate_params(self)


def validate_params(params: AtmosphereParams) -> None:
    """Raise ValueError if the parameters are not physically valid."""
    values = {
        "ambient_temp_K": params.ambient_temp_K,
        "surface_temp_excess_K": params.surface_temp_excess_K,
        "scale_height_m": params.scale_height_m,
        "n_ref": params.n_ref,
        "reference_temp_K": params.reference_temp_K,
    }
    for name, value in values.items():
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")

    if params.ambient_temp_K <= 0:
        raise ValueError(
            f"ambient_temp_K must be > 0 K, got {params.ambient_temp_K!r}"
        )
    if params.reference_temp_K <= 0:
        raise ValueError(
            f"reference_temp_K must be > 0 K, got {params.reference_temp_K!r}"
        )
    if params.scale_height_m <= 0:
        raise ValueError(
            f"scale_height_m must be > 0 m, got {params.scale_height_m!r}"
        )
    # T(y) lies between T_a and T_a + dT, so the surface value must be > 0 too.
    if params.ambient_temp_K + params.surface_temp_excess_K <= 0:
        raise ValueError(
            "ambient_temp_K + surface_temp_excess_K must be > 0 K "
            f"(got {params.ambient_temp_K + params.surface_temp_excess_K!r})"
        )
    if params.n_ref < 1.0:
        raise ValueError(f"n_ref must be >= 1, got {params.n_ref!r}")


def _prepare_height(y_m):
    """Convert height input to a float64 array and validate it."""
    y = np.asarray(y_m, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("y_m must contain only finite values")
    if np.any(y < 0):
        raise ValueError("y_m must be >= 0 (height above the surface)")
    return y


def _finalize(result: np.ndarray):
    """Return a Python float for 0-d results, otherwise the ndarray."""
    return float(result) if result.ndim == 0 else result


def temperature(y_m, params: AtmosphereParams | None = None):
    """Air temperature T(y) = T_a + dT * exp(-y / H) in kelvin."""
    params = AtmosphereParams() if params is None else params
    validate_params(params)
    y = _prepare_height(y_m)
    T = params.ambient_temp_K + params.surface_temp_excess_K * np.exp(
        -y / params.scale_height_m
    )
    return _finalize(T)


def refractive_index(y_m, params: AtmosphereParams | None = None):
    """Refractive index n(y) = 1 + (n_ref - 1) * T_ref / T(y)."""
    params = AtmosphereParams() if params is None else params
    validate_params(params)
    y = _prepare_height(y_m)
    T = np.asarray(temperature(y, params), dtype=float)
    n = 1.0 + (params.n_ref - 1.0) * params.reference_temp_K / T
    return _finalize(n)


def dn_dy(y_m, params: AtmosphereParams | None = None):
    """Analytic gradient dn/dy in 1/m.

    dn/dy = (n_ref - 1) * T_ref * dT * exp(-y/H) / (H * T(y)^2)
    Exactly zero everywhere when surface_temp_excess_K == 0.
    """
    params = AtmosphereParams() if params is None else params
    validate_params(params)
    y = _prepare_height(y_m)
    T = np.asarray(temperature(y, params), dtype=float)
    decay = np.exp(-y / params.scale_height_m)
    grad = (
        (params.n_ref - 1.0)
        * params.reference_temp_K
        * params.surface_temp_excess_K
        * decay
        / (params.scale_height_m * T**2)
    )
    return _finalize(grad)