from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from additional_experiments import (  # noqa: E402
    critical_downward_angle_deg,
    run_critical_angle_experiment,
    run_thermal_gradient_comparison,
)
from atmosphere import AtmosphereParams, dn_dy, refractive_index  # noqa: E402
from ray_solver import trace_ray  # noqa: E402


def test_critical_angle_matches_invariant_formula():
    params = AtmosphereParams()
    y0 = 0.30
    angle = critical_downward_angle_deg(params, y0)
    expected = math.degrees(
        math.acos(refractive_index(0.0, params) / refractive_index(y0, params))
    )
    assert angle == pytest.approx(expected, rel=0, abs=1e-13)
    assert angle == pytest.approx(0.4056875533, abs=1e-8)


def test_critical_angle_brackets_numerical_transition():
    params = AtmosphereParams()
    y0 = 0.30
    critical = critical_downward_angle_deg(params, y0)

    below = trace_ray(params, y0_m=y0, theta0_deg=-0.999 * critical, x_max_m=120.0)
    above = trace_ray(params, y0_m=y0, theta0_deg=-1.001 * critical, x_max_m=120.0)

    assert not below.hit_ground
    assert np.min(below.y_m) > 0.0
    assert np.any((below.slope[:-1] < 0.0) & (below.slope[1:] >= 0.0))
    assert above.hit_ground


def test_no_positive_critical_angle_for_uniform_or_cooled_medium():
    for excess in (0.0, -20.0):
        params = AtmosphereParams(surface_temp_excess_K=excess)
        with pytest.raises(ValueError, match="positive critical downward angle"):
            critical_downward_angle_deg(params, 0.30)


def test_invalid_launch_height_rejected():
    with pytest.raises(ValueError):
        critical_downward_angle_deg(AtmosphereParams(), 0.0)


def test_cold_medium_has_negative_index_gradient_and_more_downward_bending():
    heated = AtmosphereParams(surface_temp_excess_K=40.0)
    uniform = AtmosphereParams(surface_temp_excess_K=0.0)
    cooled = AtmosphereParams(surface_temp_excess_K=-20.0)

    assert dn_dy(0.1, heated) > 0.0
    assert dn_dy(0.1, uniform) == 0.0
    assert dn_dy(0.1, cooled) < 0.0

    hot_path = trace_ray(heated, 0.30, -0.30, 40.0)
    uniform_path = trace_ray(uniform, 0.30, -0.30, 40.0)
    cold_path = trace_ray(cooled, 0.30, -0.30, 40.0)

    assert hot_path.slope[-1] > uniform_path.slope[-1]
    assert cold_path.slope[-1] < uniform_path.slope[-1]


def test_additional_figures_and_results_are_written(tmp_path):
    params = AtmosphereParams()
    fig1, result = run_critical_angle_experiment(
        params, 0.30, 120.0, 0.25, tmp_path
    )
    fig2, paths = run_thermal_gradient_comparison(
        params, 0.30, -0.30, 120.0, 0.25, tmp_path
    )

    assert fig1.is_file() and fig1.stat().st_size > 0
    assert fig2.is_file() and fig2.stat().st_size > 0
    assert result["critical_angle_deg"] == pytest.approx(0.4056875533, abs=1e-8)
    assert result["lower_turning_sample_deg"] < result["critical_angle_deg"]
    assert result["upper_ground_hit_sample_deg"] > result["critical_angle_deg"]
    assert result["critical_sample_min_y_m"] < 2e-6
    assert set(paths) == {40.0, 0.0, -20.0}
    assert not paths[40.0].hit_ground
    assert paths[0.0].hit_ground
    assert paths[-20.0].hit_ground
