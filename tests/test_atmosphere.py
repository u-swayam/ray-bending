"""Unit tests for atmosphere.py."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the project root importable without touching any other project file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmosphere import AtmosphereParams, dn_dy, refractive_index, temperature  # noqa: E402


@pytest.fixture
def p():
    return AtmosphereParams()


# ---------------------------------------------------------------- defaults
def test_default_parameters(p):
    assert p.ambient_temp_K == 293.15
    assert p.surface_temp_excess_K == 40.0
    assert p.scale_height_m == 0.20
    assert p.n_ref == 1.000277
    assert p.reference_temp_K == 293.15


# ------------------------------------------------------------- temperature
def test_temperature_at_surface(p):
    assert temperature(0.0, p) == pytest.approx(333.15)


def test_temperature_at_one_scale_height(p):
    expected = 293.15 + 40.0 / math.e
    assert temperature(p.scale_height_m, p) == pytest.approx(expected)


def test_temperature_decays_to_ambient(p):
    assert temperature(10.0, p) == pytest.approx(293.15, abs=1e-12)


def test_temperature_monotonic_decreasing(p):
    T = temperature(np.linspace(0, 2, 200), p)
    assert np.all(np.diff(T) < 0)


def test_temperature_array_and_scalar_types(p):
    assert isinstance(temperature(0.1, p), float)
    out = temperature(np.array([0.0, 0.1, 0.2]), p)
    assert isinstance(out, np.ndarray) and out.shape == (3,)
    assert out[1] == pytest.approx(temperature(0.1, p))
    out2d = temperature(np.zeros((2, 3)), p)
    assert out2d.shape == (2, 3)


# -------------------------------------------------------- refractive index
def test_refractive_index_at_surface(p):
    expected = 1.0 + 0.000277 * 293.15 / 333.15
    assert refractive_index(0.0, p) == pytest.approx(expected, rel=1e-12)


def test_refractive_index_far_field_equals_n_ref(p):
    # T -> T_a == T_ref far away, so n -> n_ref.
    assert refractive_index(10.0, p) == pytest.approx(p.n_ref, abs=1e-12)


def test_refractive_index_increases_with_height(p):
    n = refractive_index(np.linspace(0, 2, 200), p)
    assert np.all(np.diff(n) > 0)
    assert np.all(n >= 1.0)


def test_refractive_index_array_matches_scalar(p):
    ys = np.array([0.0, 0.05, 0.3])
    arr = refractive_index(ys, p)
    for y, val in zip(ys, arr):
        assert val == pytest.approx(refractive_index(float(y), p))
    assert isinstance(refractive_index(0.1, p), float)


# -------------------------------------------------------------- derivative
def test_dn_dy_at_surface(p):
    k = (p.n_ref - 1.0) * p.reference_temp_K
    expected = k * 40.0 / (0.20 * 333.15**2)
    assert dn_dy(0.0, p) == pytest.approx(expected, rel=1e-12)


def test_dn_dy_matches_finite_difference(p):
    ys = np.linspace(0.01, 1.0, 25)
    # h is a compromise: large enough that roundoff in n ~ 1.0003 is negligible
    # (~1e-16/h), small enough that truncation error (~h^2 * n'''/6) is tiny.
    h = 1e-4
    numeric = (refractive_index(ys + h, p) - refractive_index(ys - h, p)) / (2 * h)
    np.testing.assert_allclose(dn_dy(ys, p), numeric, rtol=1e-5)


def test_dn_dy_positive_and_decaying(p):
    g = dn_dy(np.linspace(0, 2, 200), p)
    assert np.all(g > 0)
    assert np.all(np.diff(g) < 0)


def test_dn_dy_scalar_and_array_types(p):
    assert isinstance(dn_dy(0.1, p), float)
    assert dn_dy(np.array([0.0, 0.1]), p).shape == (2,)


# ---------------------------------------------------------- uniform medium
def test_uniform_medium_temperature_constant():
    p = AtmosphereParams(surface_temp_excess_K=0.0)
    np.testing.assert_allclose(temperature(np.linspace(0, 5, 20), p), 293.15)


def test_uniform_medium_index_constant():
    p = AtmosphereParams(surface_temp_excess_K=0.0)
    n = refractive_index(np.linspace(0, 5, 20), p)
    np.testing.assert_allclose(n, p.n_ref, rtol=0, atol=1e-15)


def test_uniform_medium_gradient_exactly_zero():
    p = AtmosphereParams(surface_temp_excess_K=0.0)
    assert dn_dy(0.0, p) == 0.0
    assert np.all(dn_dy(np.linspace(0, 5, 20), p) == 0.0)


def test_uniform_medium_with_different_ambient_temperature():
    p = AtmosphereParams(ambient_temp_K=320.0, surface_temp_excess_K=0.0)
    expected = 1.0 + 0.000277 * 293.15 / 320.0
    assert refractive_index(0.3, p) == pytest.approx(expected, rel=1e-12)


# -------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale_height_m": 0.0},
        {"scale_height_m": -0.1},
        {"scale_height_m": float("nan")},
        {"scale_height_m": float("inf")},
        {"ambient_temp_K": 0.0},
        {"ambient_temp_K": -10.0},
        {"reference_temp_K": 0.0},
        {"reference_temp_K": -1.0},
        {"surface_temp_excess_K": -300.0},  # surface T = -6.85 K
        {"surface_temp_excess_K": float("nan")},
        {"n_ref": 0.5},
    ],
)
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        AtmosphereParams(**kwargs)


@pytest.mark.parametrize("bad_y", [-0.1, float("nan"), float("inf")])
def test_invalid_height_raises(p, bad_y):
    for fn in (temperature, refractive_index, dn_dy):
        with pytest.raises(ValueError):
            fn(bad_y, p)


def test_invalid_height_in_array_raises(p):
    with pytest.raises(ValueError):
        temperature(np.array([0.0, -0.5, 1.0]), p)


def test_negative_excess_allowed_when_temperature_stays_positive():
    p = AtmosphereParams(surface_temp_excess_K=-20.0)
    assert temperature(0.0, p) == pytest.approx(273.15)
    assert dn_dy(0.1, p) < 0  # cold surface: n decreases with height