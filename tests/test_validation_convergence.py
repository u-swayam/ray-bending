from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atmosphere import AtmosphereParams  # noqa: E402
from validation import convergence_study  # noqa: E402


def test_convergence_error_reduces_with_tighter_solver_settings():
    result = convergence_study(
        AtmosphereParams(),
        y0_m=0.30,
        theta0_deg=-0.30,
        x_eval_m=50.0,
        steps_m=(20.0, 10.0, 5.0),
        solver_tolerances=((1e-2, 1e-5), (1e-4, 1e-7), (1e-8, 1e-11)),
    )
    errors = [row.abs_error_m for row in result.rows]
    assert all(error is not None for error in errors)
    assert errors[0] > errors[1] > errors[2]
    assert result.reference_height_m is not None
    assert result.reference_rtol < result.rows[-1].rtol
    assert result.reference_atol < result.rows[-1].atol
