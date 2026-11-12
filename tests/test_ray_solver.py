"""Unit tests for ray_solver.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmosphere import AtmosphereParams  # noqa: E402
from ray_solver import RayPath, trace_ray  # noqa: E402


@pytest.fixture
def hot():
    return AtmosphereParams()  # 40 K excess


@pytest.fixture
def uniform():
    return AtmosphereParams(surface_temp_excess_K=0.0)


# 1. Uniform medium -> straight line ---------------------------------------
def test_uniform_medium_is_straight_line(uniform):
    y0, theta = 0.30, -0.10  # does not reach the ground within 120 m
    path = trace_ray(uniform, y0_m=y0, theta0_deg=theta, x_max_m=120.0)
    expected = y0 + path.x_m * math.tan(math.radians(theta))
    assert not path.hit_ground
    assert path.x_m[-1] == pytest.approx(120.0)
    np.testing.assert_allclose(path.y_m, expected, rtol=0, atol=1e-9)
    np.testing.assert_allclose(path.slope, math.tan(math.radians(theta)), atol=1e-12)


def test_uniform_medium_ground_hit_location(uniform):
    y0, theta = 0.30, -0.30
    path = trace_ray(uniform, y0_m=y0, theta0_deg=theta, x_max_m=120.0)
    x_expected = y0 / -math.tan(math.radians(theta))
    assert path.hit_ground
    assert path.x_m[-1] == pytest.approx(x_expected, rel=1e-6)


# 2. Positive excess bends the ray upward ----------------------------------
def test_gradient_bends_ray_upward(hot):
    y0, theta = 0.30, -0.10
    path = trace_ray(hot, y0_m=y0, theta0_deg=theta, x_max_m=60.0)
    t0 = math.tan(math.radians(theta))
    straight = y0 + path.x_m * t0
    assert np.all(path.y_m[1:] > straight[1:])
    assert path.slope[-1] > t0
    assert np.all(np.diff(path.slope) >= -1e-15)  # slope never decreases (n' > 0)


def test_default_ray_is_refracted_away_from_ground(hot):
    # Straight line would hit the ground at ~57 m; the mirage layer turns it.
    path = trace_ray(hot)
    assert not path.hit_ground
    assert np.min(path.y_m) > 0.0
    assert path.slope[-1] > math.tan(math.radians(-0.30))


# 3. Ground hit stops the integration --------------------------------------
def test_ray_hits_ground_and_stops(hot):
    path = trace_ray(hot, y0_m=0.30, theta0_deg=-1.0, x_max_m=120.0)
    assert path.hit_ground
    assert path.y_m[-1] == 0.0
    assert path.x_m[-1] < 120.0
    assert np.all(path.y_m[:-1] > 0.0)
    assert path.x_m[-1] == pytest.approx(0.30 / math.tan(math.radians(1.0)), rel=0.05)


# 4. No NaN / inf -----------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"theta0_deg": -1.0},
        {"theta0_deg": -30.0},
        {"theta0_deg": 5.0},
        {"y0_m": 1e-3, "theta0_deg": -0.01},
        {"x_max_m": 1000.0, "max_step_m": 1.0},
    ],
)
def test_outputs_are_finite(hot, kwargs):
    path = trace_ray(hot, **kwargs)
    assert isinstance(path, RayPath)
    for arr in (path.x_m, path.y_m, path.slope):
        assert np.all(np.isfinite(arr))
    assert path.x_m.shape == path.y_m.shape == path.slope.shape
    assert np.all(np.diff(path.x_m) > 0)
    assert np.all(path.y_m >= 0.0)


def test_max_step_respected(hot):
    path = trace_ray(hot, theta0_deg=0.5, x_max_m=10.0, max_step_m=0.25)
    assert np.max(np.diff(path.x_m)) <= 0.25 + 1e-12


# Validation ----------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"y0_m": 0.0},
        {"y0_m": -1.0},
        {"y0_m": float("nan")},
        {"theta0_deg": 90.0},
        {"theta0_deg": float("inf")},
        {"x_max_m": 0.0},
        {"x_max_m": -5.0},
        {"max_step_m": 0.0},
        {"max_step_m": float("nan")},
    ],
)
def test_invalid_inputs_raise(hot, kwargs):
    with pytest.raises(ValueError):
        trace_ray(hot, **kwargs)


def test_invalid_params_type_raises():
    with pytest.raises(TypeError):
        trace_ray({"ambient_temp_K": 293.15})
        

def test_custom_solver_tolerances_are_supported(hot):
    path = trace_ray(
        hot, y0_m=0.30, theta0_deg=-0.30, x_max_m=20.0,
        max_step_m=2.0, rtol=1e-5, atol=1e-8,
    )
    assert path.x_m[-1] == pytest.approx(20.0)
    assert np.all(np.isfinite(path.y_m))


@pytest.mark.parametrize("kwargs", [{"rtol": 0.0}, {"atol": -1.0}, {"rtol": float("nan")}])
def test_invalid_solver_tolerances_raise(hot, kwargs):
    with pytest.raises(ValueError):
        trace_ray(hot, x_max_m=10.0, **kwargs)
