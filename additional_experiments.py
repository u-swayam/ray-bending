"""Additional physical experiments for the graded-index ray-bending project.

This module adds:
1. An analytical critical launch angle, checked against numerical rays.
2. A comparison of heated, uniform, and cooled near-surface media.

The underlying atmospheric model and ray equation remain in atmosphere.py and
ray_solver.py respectively.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from atmosphere import AtmosphereParams, refractive_index
from config_io import Config
from ray_solver import RayPath, trace_ray

__all__ = [
    "critical_downward_angle_deg",
    "run_critical_angle_experiment",
    "run_thermal_gradient_comparison",
    "run_additional_experiments",
]

DPI = 160
FILE_CRITICAL = "07_critical_launch_angle.png"
FILE_THERMAL = "08_heated_uniform_cooled.png"
FILE_REPORT = "additional_results.txt"


def critical_downward_angle_deg(
    params: AtmosphereParams, y0_m: float
) -> float:
    """Return the critical downward launch-angle magnitude in degrees.

    At the limiting ray, the ray becomes horizontal as it reaches y=0.
    Conservation of n*cos(theta) therefore gives

        n(y0) cos(theta_c) = n(0).

    A positive critical angle exists for the heated-surface case, where
    n(y0) > n(0). Uniform and cooled profiles do not have this threshold.
    """
    if not isinstance(params, AtmosphereParams):
        raise TypeError("params must be an AtmosphereParams instance")
    if not np.isscalar(y0_m) or not np.isfinite(y0_m) or y0_m <= 0:
        raise ValueError("y0_m must be a finite positive scalar")

    n_ground = float(refractive_index(0.0, params))
    n_launch = float(refractive_index(float(y0_m), params))
    ratio = n_ground / n_launch

    if not ratio < 1.0:
        raise ValueError(
            "A positive critical downward angle exists only when the "
            "refractive index increases with height (heated-surface profile)."
        )
    if ratio <= 0.0 or not np.isfinite(ratio):
        raise ValueError("The refractive-index ratio is outside its physical range")

    return float(np.degrees(np.arccos(np.clip(ratio, -1.0, 1.0))))


def _has_turning_point(path: RayPath) -> bool:
    """Detect a downward-to-upward reversal in the sampled ray slope."""
    slope = np.asarray(path.slope, dtype=float)
    if slope.size < 2:
        return False
    tol = 1e-12
    if abs(float(slope[0])) <= tol and slope[1] > tol:
        return True
    return bool(np.any((slope[:-1] < 0.0) & (slope[1:] >= 0.0)))


def _interpolated_turning_point(path: RayPath) -> tuple[float, float] | None:
    """Interpolate the first p=dy/dx crossing to estimate a turning point."""
    x = np.asarray(path.x_m, dtype=float)
    y = np.asarray(path.y_m, dtype=float)
    p = np.asarray(path.slope, dtype=float)
    if x.size < 2:
        return None
    tol = 1e-12
    if abs(float(p[0])) <= tol and p[1] > tol:
        return float(x[0]), float(y[0])
    crossings = np.flatnonzero((p[:-1] < 0.0) & (p[1:] >= 0.0))
    if crossings.size == 0:
        return None
    i = int(crossings[0])
    dp = float(p[i + 1] - p[i])
    f = 0.0 if dp == 0.0 else float(np.clip(-p[i] / dp, 0.0, 1.0))
    return (
        float(x[i] + f * (x[i + 1] - x[i])),
        float(y[i] + f * (y[i + 1] - y[i])),
    )


def run_critical_angle_experiment(
    params: AtmosphereParams,
    y0_m: float,
    x_max_m: float,
    max_step_m: float,
    output_dir: str | Path,
) -> tuple[Path, dict]:
    """Plot numerical minimum ray height versus launch-angle magnitude."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    theta_c = critical_downward_angle_deg(params, y0_m)

    # Include closely spaced samples on both sides of theta_c so the numerical
    # ground-hit/turning transition can be compared directly with theory.
    multipliers = np.concatenate(
        [
            np.linspace(0.0, 1.50, 9),
            np.array([0.98, 0.999, 1.0, 1.001, 1.02]),
        ]
    )
    multipliers = np.unique(np.round(multipliers, 9))
    angles = multipliers * theta_c

    rows: list[dict] = []
    for angle in angles:
        path = trace_ray(
            params,
            y0_m=y0_m,
            theta0_deg=-float(angle),
            x_max_m=x_max_m,
            max_step_m=max_step_m,
        )
        turning = _has_turning_point(path)
        is_critical_sample = abs(float(angle) - theta_c) <= max(1e-10, theta_c * 1e-8)
        if path.hit_ground:
            status = "ground_hit"
        elif turning:
            status = "critical" if is_critical_sample else "turned"
        else:
            status = "not_resolved"

        min_index = int(np.argmin(path.y_m))
        turn_point = _interpolated_turning_point(path)
        rows.append(
            {
                "angle_deg": float(angle),
                "min_height_m": float(np.min(path.y_m)),
                "min_height_x_m": float(path.x_m[min_index]),
                "x_end_m": float(path.x_m[-1]),
                "hit_ground": bool(path.hit_ground),
                "turning": turning,
                "turning_x_m": turn_point[0] if turn_point else None,
                "turning_y_m": turn_point[1] if turn_point else None,
                "status": status,
            }
        )

    # Nearest sampled numerical bracket on either side of the analytic value.
    below_turn = [r for r in rows if r["angle_deg"] < theta_c and r["status"] == "turned"]
    above_hit = [r for r in rows if r["angle_deg"] > theta_c and r["status"] == "ground_hit"]
    lower = max(below_turn, key=lambda r: r["angle_deg"]) if below_turn else None
    upper = min(above_hit, key=lambda r: r["angle_deg"]) if above_hit else None

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    groups = [
        ("turned above ground", "turned", "tab:blue", "o"),
        ("critical (theory sample)", "critical", "gold", "*"),
        ("ground impact", "ground_hit", "tab:red", "x"),
        ("not resolved within x range", "not_resolved", "tab:orange", "^"),
    ]
    for label, status, color, marker in groups:
        subset = [r for r in rows if r["status"] == status]
        if not subset:
            continue
        ax.scatter(
            [r["angle_deg"] for r in subset],
            [r["min_height_m"] for r in subset],
            color=color,
            marker=marker,
            s=150 if status == "critical" else 42,
            edgecolors="black" if status == "critical" else None,
            linewidths=0.8 if status == "critical" else None,
            label=label,
            zorder=6 if status == "critical" else 3,
        )

    ax.axvline(
        theta_c,
        color="black",
        ls="--",
        lw=1.5,
        label=rf"theoretical $|\theta_c|={theta_c:.5f}^\circ$",
    )
    ax.axhline(0.0, color="saddlebrown", lw=1.8, label="ground")

    # A small zoom makes the critical sample distinguishable from the first
    # finite-distance ground-hit sample, whose angles are intentionally close.
    near = [r for r in rows if abs(r["angle_deg"] - theta_c) <= 0.012]
    ax_zoom = ax.inset_axes([0.54, 0.14, 0.38, 0.42])
    for label, status, color, marker in groups:
        subset = [r for r in near if r["status"] == status]
        if subset:
            ax_zoom.scatter(
                [r["angle_deg"] for r in subset],
                [r["min_height_m"] for r in subset],
                color=color,
                marker=marker,
                s=65 if status == "critical" else 24,
                edgecolors="black" if status == "critical" else None,
                linewidths=0.7 if status == "critical" else None,
                zorder=5,
            )
    ax_zoom.axvline(theta_c, color="black", ls="--", lw=1.0)
    ax_zoom.axhline(0.0, color="saddlebrown", lw=1.0)
    ax_zoom.set_xlim(theta_c - 0.012, theta_c + 0.012)
    ax_zoom.set_ylim(-0.00015, max(0.0045, max(r["min_height_m"] for r in near) * 1.1))
    ax_zoom.set_title("Threshold detail", fontsize=8)
    ax_zoom.tick_params(labelsize=7)
    ax_zoom.grid(alpha=0.25)

    ax.set_xlabel("Magnitude of initial downward angle |θ₀| (degrees)")
    ax.set_ylabel("Minimum ray height during simulation (m)")
    ax.set_title("Critical launch angle for a heated graded-index atmosphere")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig_path = out / FILE_CRITICAL
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)

    critical_sample = next((r for r in rows if r["status"] == "critical"), None)
    result = {
        "critical_angle_deg": theta_c,
        "n_ground": float(refractive_index(0.0, params)),
        "n_launch": float(refractive_index(y0_m, params)),
        "critical_sample_min_y_m": critical_sample["min_height_m"] if critical_sample else None,
        "critical_sample_min_x_m": critical_sample["min_height_x_m"] if critical_sample else None,
        "lower_turning_sample_deg": lower["angle_deg"] if lower else None,
        "upper_ground_hit_sample_deg": upper["angle_deg"] if upper else None,
        "rows": rows,
    }
    return fig_path, result


def run_thermal_gradient_comparison(
    params: AtmosphereParams,
    y0_m: float,
    theta0_deg: float,
    x_max_m: float,
    max_step_m: float,
    output_dir: str | Path,
) -> tuple[Path, dict]:
    """Compare ray propagation in heated, uniform, and cooled media."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    heated_excess = float(params.surface_temp_excess_K)
    if heated_excess <= 0.0:
        raise ValueError(
            "The heated/uniform/cooled comparison requires the configured "
            "reference atmosphere to have a positive surface temperature excess."
        )
    cooled_excess = -20.0
    cases = (
        (f"Heated (+{heated_excess:g} K excess)", heated_excess, "tab:red"),
        ("Uniform (0 K excess)", 0.0, "black"),
        (f"Cooled ({cooled_excess:g} K excess)", cooled_excess, "tab:blue"),
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    results = {}
    for label, excess, color in cases:
        case_params = replace(params, surface_temp_excess_K=excess)
        path = trace_ray(
            case_params,
            y0_m=y0_m,
            theta0_deg=theta0_deg,
            x_max_m=x_max_m,
            max_step_m=max_step_m,
        )
        results[excess] = path
        impact_text = (
            f", ground hit at {path.x_m[-1]:.1f} m" if path.hit_ground else ""
        )
        ax.plot(
            path.x_m,
            path.y_m,
            color=color,
            lw=2,
            label=f"{label}{impact_text}",
        )
        if path.hit_ground:
            ax.plot(path.x_m[-1], 0.0, marker="x", color=color, ms=8, mew=2)

    ax.axhline(0.0, color="saddlebrown", lw=2, label="Ground (y = 0)")
    ax.set_xlim(0.0, x_max_m)
    top = max(float(np.max(path.y_m)) for path in results.values())
    ax.set_ylim(-0.025 * top, 1.10 * top)
    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel("Height above ground y (m)")
    ax.set_title(
        f"Effect of the sign of the refractive-index gradient\n"
        f"(y₀ = {y0_m:g} m, θ₀ = {theta0_deg:g}°)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig_path = out / FILE_THERMAL
    fig.savefig(fig_path, dpi=DPI)
    plt.close(fig)
    return fig_path, results


def run_additional_experiments(config: Config, output_dir: str | Path) -> tuple[Path, ...]:
    """Run the critical-angle and heated/uniform/cooled experiments."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    first = config.simulation.rays[0]

    critical_path, critical = run_critical_angle_experiment(
        config.atmosphere,
        y0_m=first.y0_m,
        x_max_m=config.simulation.x_max_m,
        max_step_m=config.simulation.max_step_m,
        output_dir=out,
    )
    thermal_path, thermal = run_thermal_gradient_comparison(
        config.atmosphere,
        y0_m=first.y0_m,
        theta0_deg=first.theta0_deg,
        x_max_m=config.simulation.x_max_m,
        max_step_m=config.simulation.max_step_m,
        output_dir=out,
    )

    lines = [
        "ADDITIONAL RAY-BENDING EXPERIMENTS",
        "=" * 64,
        "",
        "1. CRITICAL DOWNWARD LAUNCH ANGLE",
        "At the limiting ray, theta=0 at y=0, so n(y0) cos(theta_c) = n(0).",
        "theta_c = arccos(n(0)/n(y0)).",
        f"Launch height y0 = {first.y0_m:.6f} m",
        f"n(0) = {critical['n_ground']:.12f}",
        f"n(y0) = {critical['n_launch']:.12f}",
        f"Theoretical critical angle magnitude = {critical['critical_angle_deg']:.9f} deg",
    ]
    if critical["critical_sample_min_y_m"] is not None:
        lines.append(
            "Numerical critical-angle sample minimum height = "
            f"{critical['critical_sample_min_y_m']:.9e} m at "
            f"x = {critical['critical_sample_min_x_m']:.6f} m"
        )
    if critical["lower_turning_sample_deg"] is not None:
        lines.append(
            "Nearest sampled turning ray below threshold = "
            f"{critical['lower_turning_sample_deg']:.9f} deg"
        )
    else:
        lines.append("No sampled turning ray below threshold was classified.")
    if critical["upper_ground_hit_sample_deg"] is not None:
        lines.append(
            "Nearest sampled ground-hit ray above threshold = "
            f"{critical['upper_ground_hit_sample_deg']:.9f} deg"
        )
    else:
        lines.append("No sampled ground-hit ray above threshold was classified.")

    lines += [
        "",
        "2. SIGN OF THE REFRACTIVE-INDEX GRADIENT",
        f"Initial height = {first.y0_m:.6f} m; initial angle = {first.theta0_deg:.6f} deg",
        "Temperature excess (K) | outcome | final/minimum height (m)",
    ]
    for excess in (float(config.atmosphere.surface_temp_excess_K), 0.0, -20.0):
        path = thermal[excess]
        outcome = f"ground impact at x={path.x_m[-1]:.6f} m" if path.hit_ground else "no ground impact"
        lines.append(
            f"{excess:>21.1f} | {outcome} | min y={float(np.min(path.y_m)):.9f}"
        )
    lines += [
        "",
        "Interpretation: positive dn/dy bends a downward ray upward; zero dn/dy",
        "produces a straight ray; negative dn/dy bends the ray downward.",
        "",
    ]
    report_path = out / FILE_REPORT
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return critical_path, thermal_path, report_path
