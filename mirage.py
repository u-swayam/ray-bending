"""Geometrical-optics illustration of apparent line-of-sight displacement.

A source sits at (x = 0, y = source_y_m) near a flat heated ground and an
observer at (x = observer_distance_m, y = observer_y_m). The launch angle that
makes the real, curved ray reach the observer is found by shooting with the
existing trace_ray() and scipy.optimize.brentq. The tangent to the ray at the
observer is then extended backwards to the ground.

SCOPE / LIMITATIONS
This is a ray-diagram demonstration only. It is NOT a simulation of a
photographic inferior mirage: scattering, surface reflection, turbulence and
extended-source imaging are all excluded. Only the single ray that connects
source and observer through the n(y) model is drawn; nothing is invented.

Public interface:
    compute_mirage(params, ...)   -> MirageResult   (raises MirageError)
    plot_mirage(result, params, path) -> Path
    run_mirage(params, ..., output_dir="outputs") -> MirageResult
    run_mirage_from_config(cfg, output_dir="outputs") -> MirageResult

Run standalone:  python mirage.py [input.json]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # file output only, no GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from atmosphere import AtmosphereParams  # noqa: E402
from ray_solver import RayPath, trace_ray  # noqa: E402

__all__ = [
    "MirageError",
    "MirageResult",
    "solve_observer_rays",
    "backward_tangent_ground_x",
    "compute_mirage",
    "plot_mirage",
    "run_mirage",
    "run_mirage_from_config",
]

FILE_MIRAGE = "04_mirage_visualisation.png"
DPI = 150


class MirageError(RuntimeError):
    """No valid source-to-observer ray (message explains what to change)."""


class _GroundHit(Exception):
    """Internal: a trial ray hit the ground before the observer."""


@dataclass(frozen=True)
class MirageResult:
    source_y_m: float
    observer_y_m: float
    observer_distance_m: float
    launch_angle_deg: float
    n_solutions: int  # number of valid launch angles found
    path: RayPath  # actual curved ray, source -> observer
    reference: RayPath  # straight ray in a uniform medium (dT = 0)
    residual_m: float  # |y(observer_distance) - observer_y|
    final_slope: float  # dy/dx at the observer
    final_angle_deg: float  # arrival elevation angle
    geometric_angle_deg: float  # straight source->observer elevation angle
    tangent_ground_x_m: float | None  # backward tangent meets y = 0 here
    apparent_displacement_m: float | None  # tangent_ground_x - source x (= 0)
    apparent_source_y_m: float  # backward tangent height at x = 0


# --------------------------------------------------------- shooting method
def _height_at_observer(params, source_y_m, theta_deg, distance_m, max_step_m):
    """Ray height at x = distance_m, or None if it grounded first."""
    path = trace_ray(params, source_y_m, theta_deg, distance_m, max_step_m)
    if path.hit_ground or path.x_m[-1] < distance_m * (1.0 - 1e-12):
        return None
    return float(np.interp(distance_m, path.x_m, path.y_m))


def solve_observer_rays(
    params: AtmosphereParams,
    source_y_m: float,
    observer_y_m: float,
    observer_distance_m: float,
    max_step_m: float = 0.25,
    angle_range_deg: tuple[float, float] = (-2.0, 3.0),
    n_scan: int = 26,
):
    """Find launch angles whose ray passes through the observer.

    Scans angle_range_deg on n_scan points, records the (valid) ray height at
    the observer, and runs brentq only on brackets where BOTH ends are valid
    (did not hit the ground) and the height error changes sign. A bracket whose
    interior hits the ground is discarded.

    Returns (roots_deg sorted ascending, scan_angles_deg, scan_heights) where
    scan_heights[i] is None for rays that hit the ground.
    """
    lo, hi = angle_range_deg
    if not (-90.0 < lo < hi < 90.0):
        raise ValueError(f"angle_range_deg must satisfy -90 < lo < hi < 90, got {angle_range_deg!r}")
    if n_scan < 3:
        raise ValueError("n_scan must be >= 3")

    angles = np.linspace(lo, hi, int(n_scan))
    heights = [
        _height_at_observer(params, source_y_m, a, observer_distance_m, max_step_m)
        for a in angles
    ]

    def err(theta):
        h = _height_at_observer(params, source_y_m, theta, observer_distance_m, max_step_m)
        if h is None:
            raise _GroundHit
        return h - observer_y_m

    roots: list[float] = []
    for i in range(len(angles) - 1):
        ha, hb = heights[i], heights[i + 1]
        if ha is None or hb is None:
            continue  # no valid bracket across a grounded ray
        fa, fb = ha - observer_y_m, hb - observer_y_m
        if fa * fb > 0:
            continue
        try:
            root = brentq(err, angles[i], angles[i + 1], xtol=1e-10, rtol=1e-12)
        except _GroundHit:
            continue
        if not any(abs(root - r) < 1e-8 for r in roots):
            roots.append(float(root))
    return sorted(roots), angles, heights


def backward_tangent_ground_x(x_obs: float, y_obs: float, slope: float) -> float | None:
    """x where the tangent line through (x_obs, y_obs) meets y = 0 (None if never).

    The line is y = y_obs + slope * (x - x_obs); for slope <= 0 extending it
    backwards never descends to the ground.
    """
    if slope <= 0.0:
        return None
    return x_obs - y_obs / slope


def _failure_message(scan_angles, scan_heights, params, ys, yo, D, roots_found):
    lo, hi = float(scan_angles[0]), float(scan_angles[-1])
    valid = [h for h in scan_heights if h is not None]
    head = (f"No valid ray from the source (y = {ys:g} m) reaches the observer "
            f"(y = {yo:g} m at x = {D:g} m) for launch angles in [{lo:g}, {hi:g}] deg.")
    if not valid:
        return (head + " Every scanned ray hit the ground first. Raise source_y_m, "
                "shorten observer_distance_m, increase surface_temp_excess_K "
                "(stronger upward bending), or scan more upward angles.")
    hmin, hmax = min(valid), max(valid)
    if yo > hmax:
        return (head + f" Rays that avoid the ground arrive between {hmin:.4f} and "
                f"{hmax:.4f} m, below the observer. Lower observer_y_m, or widen "
                "angle_range_deg towards larger upward angles.")
    if yo < hmin:
        return (head + f" The lowest ray that avoids the ground arrives at {hmin:.4f} m, "
                "above the observer. Raise observer_y_m, shorten observer_distance_m, "
                "or increase surface_temp_excess_K so rays bend upward sooner.")
    return (head + f" Valid rays span {hmin:.4f}-{hmax:.4f} m at the observer, but no "
            "sign change was bracketed between neighbouring valid scan angles. "
            "Increase n_scan or widen angle_range_deg.")


def compute_mirage(
    params: AtmosphereParams,
    source_y_m: float = 0.05,
    observer_y_m: float = 1.5,
    observer_distance_m: float = 120.0,
    max_step_m: float = 0.25,
    angle_range_deg: tuple[float, float] = (-2.0, 3.0),
    n_scan: int = 26,
    branch: str = "lowest",
) -> MirageResult:
    """Find the source->observer ray and its backward tangent.

    branch selects among several valid launch angles: "lowest" (default, the
    ray that stays closest to the ground) or "highest".
    Raises MirageError when no valid ray exists.
    """
    for name, v in (("source_y_m", source_y_m), ("observer_y_m", observer_y_m),
                    ("observer_distance_m", observer_distance_m)):
        if not np.isscalar(v) or not np.isfinite(v) or v <= 0:
            raise ValueError(f"{name} must be a finite number > 0, got {v!r}")
    if branch not in ("lowest", "highest"):
        raise ValueError(f"branch must be 'lowest' or 'highest', got {branch!r}")

    roots, angles, heights = solve_observer_rays(
        params, source_y_m, observer_y_m, observer_distance_m,
        max_step_m, angle_range_deg, n_scan)
    if not roots:
        raise MirageError(_failure_message(
            angles, heights, params, source_y_m, observer_y_m, observer_distance_m, roots))

    theta = roots[0] if branch == "lowest" else roots[-1]
    path = trace_ray(params, source_y_m, theta, observer_distance_m, max_step_m)
    y_end = float(path.y_m[-1])
    slope = float(path.slope[-1])

    geo_deg = math.degrees(math.atan2(observer_y_m - source_y_m, observer_distance_m))
    uniform = replace(params, surface_temp_excess_K=0.0)
    reference = trace_ray(uniform, source_y_m, geo_deg, observer_distance_m, max_step_m)

    x_g = backward_tangent_ground_x(observer_distance_m, y_end, slope)
    return MirageResult(
        source_y_m=float(source_y_m),
        observer_y_m=float(observer_y_m),
        observer_distance_m=float(observer_distance_m),
        launch_angle_deg=float(theta),
        n_solutions=len(roots),
        path=path,
        reference=reference,
        residual_m=abs(y_end - observer_y_m),
        final_slope=slope,
        final_angle_deg=math.degrees(math.atan(slope)),
        geometric_angle_deg=geo_deg,
        tangent_ground_x_m=x_g,
        apparent_displacement_m=x_g,  # source is at x = 0
        apparent_source_y_m=y_end - slope * observer_distance_m,
    )


# -------------------------------------------------------------------- plot
def plot_mirage(result: MirageResult, params: AtmosphereParams, path: str | Path) -> Path:
    r = result
    D, ys, yo = r.observer_distance_m, r.source_y_m, r.observer_y_m
    x_g = r.tangent_ground_x_m

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Actual ray (solid) and straight uniform-medium reference (dash-dot).
    ax.plot(r.path.x_m, r.path.y_m, color="tab:blue", lw=2.2,
            label=f"Actual ray (dT = {params.surface_temp_excess_K:g} K), "
                  f"launch {r.launch_angle_deg:.4f} deg")
    ax.plot(r.reference.x_m, r.reference.y_m, color="0.35", lw=1.6, ls="-.",
            label="Straight reference (uniform medium)")

    # Backward tangent from the observer to the ground (extension, dashed red).
    if x_g is not None:
        ax.plot([D, x_g], [yo, 0.0], color="tab:red", lw=1.8, ls="--",
                label="Backward tangent at observer (extension)")
        ax.plot([x_g], [0.0], marker="X", color="tab:red", ms=11, ls="none",
                mec="k", label=f"Apparent ground intercept (x = {x_g:.2f} m)", zorder=5)
    else:
        ax.plot([0.0, D], [r.apparent_source_y_m, yo], color="tab:red", lw=1.8, ls="--",
                label="Backward tangent at observer (extension; slope <= 0, no ground intercept)")

    ax.plot([0.0], [ys], "o", color="gold", mec="k", ms=10, label="Source", zorder=6)
    ax.plot([D], [yo], "s", color="limegreen", mec="k", ms=10, label="Observer", zorder=6)

    top = 1.2 * max(yo, ys, float(np.max(r.path.y_m)))
    bottom = -0.12 * top
    left = min(0.0, x_g if x_g is not None else 0.0) - 0.05 * D
    right = 1.04 * D
    ax.axhline(0.0, color="saddlebrown", lw=2.2)
    ax.fill_between([left, right], bottom, 0.0, color="saddlebrown", alpha=0.25, label="Ground (y = 0)")
    ax.set_xlim(left, right)
    ax.set_ylim(bottom, top)

    ax.annotate(f"Source\n({0:g}, {ys:g}) m", (0.0, ys), xytext=(14, 22),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="0.4"))
    ax.annotate(f"Observer\n({D:g}, {yo:g}) m", (D, yo), xytext=(-85, 14),
                textcoords="offset points", fontsize=9,
                arrowprops=dict(arrowstyle="-", color="0.4"))

    lines = [
        f"arrival angle:        {r.final_angle_deg:.4f} deg (slope {r.final_slope:.3e})",
        f"straight-line angle:  {r.geometric_angle_deg:.4f} deg",
    ]
    if x_g is not None:
        lines.append(f"apparent displacement: {r.apparent_displacement_m:+.3f} m "
                     "(intercept - source x)")
    lines.append(f"tangent height at x=0: {r.apparent_source_y_m:+.4f} m (source: {ys:g} m)")
    ax.text(0.015, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
            family="monospace", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel("Height above ground y (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    bbox = ax.get_window_extent()
    exaggeration = ((right - left) / bbox.width) / ((top - bottom) / bbox.height)
    ax.set_title(f"Ray diagram: apparent line-of-sight displacement near a heated surface "
                 f"(vertical scale exaggerated ~{exaggeration:.0f}x)")
    fig.text(0.5, 0.012,
             "Geometrical-optics illustration only: not a simulation of a complete inferior "
             "mirage (no scattering, surface reflection, turbulence or extended-source imaging).",
             ha="center", fontsize=8, style="italic")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ driver
def run_mirage(
    params: AtmosphereParams | None = None,
    source_y_m: float = 0.05,
    observer_y_m: float = 1.5,
    observer_distance_m: float = 120.0,
    max_step_m: float = 0.25,
    output_dir: str | Path = "outputs",
    angle_range_deg: tuple[float, float] = (-2.0, 3.0),
    n_scan: int = 26,
    branch: str = "lowest",
) -> MirageResult:
    """Compute the mirage ray, save outputs/04_mirage_visualisation.png, print a summary.

    Raises MirageError (nothing is plotted) if no valid ray exists.
    """
    params = AtmosphereParams() if params is None else params
    res = compute_mirage(params, source_y_m, observer_y_m, observer_distance_m,
                         max_step_m, angle_range_deg, n_scan, branch)
    out = plot_mirage(res, params, Path(output_dir) / FILE_MIRAGE)

    print("Mirage ray-diagram summary (geometrical optics only)")
    print(f"  valid launch angles found: {res.n_solutions}; using {branch}: "
          f"{res.launch_angle_deg:.6f} deg (miss distance {res.residual_m:.2e} m)")
    print(f"  arrival angle {res.final_angle_deg:.4f} deg vs straight line "
          f"{res.geometric_angle_deg:.4f} deg")
    if res.tangent_ground_x_m is None:
        print("  backward tangent never meets the ground (arrival slope <= 0)")
    else:
        print(f"  backward tangent meets ground at x = {res.tangent_ground_x_m:.3f} m "
              f"(displacement from source: {res.apparent_displacement_m:+.3f} m)")
    print(f"Saved: {out}")
    return res


def run_mirage_from_config(cfg, output_dir: str | Path = "outputs", **kwargs) -> MirageResult:
    """Convenience wrapper for a config_io.Config (uses cfg.atmosphere, cfg.mirage,
    cfg.simulation.max_step_m). Extra kwargs go to run_mirage."""
    m = cfg.mirage
    return run_mirage(cfg.atmosphere, m.source_y_m, m.observer_y_m, m.observer_distance_m,
                      cfg.simulation.max_step_m, output_dir, **kwargs)


if __name__ == "__main__":
    import sys

    try:
        if len(sys.argv) > 1:
            from config_io import load_config
            run_mirage_from_config(load_config(sys.argv[1]))
        else:
            run_mirage()
    except MirageError as exc:
        print(f"Mirage failed: {exc}", file=sys.stderr)
        sys.exit(1)