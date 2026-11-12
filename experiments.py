"""Figures and parameter studies for the ray-bending project.

The module can be used standalone with defaults or driven by a validated
``config_io.Config`` object:

    run_all(output_dir="outputs")
    run_all(output_dir="outputs", config=load_config("input.json"))

All ray trajectories are computed by ``ray_solver.trace_ray``.  This module
contains plotting and experiment orchestration only.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # file output only, no GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from atmosphere import AtmosphereParams, dn_dy, refractive_index, temperature  # noqa: E402
from ray_solver import trace_ray  # noqa: E402

__all__ = ["run_all"]

FILE_1 = "01_temperature_and_index.png"
FILE_2 = "02_ray_trajectories.png"
FILE_3 = "03_surface_temperature_sensitivity.png"
DPI = 150


# --------------------------------------------------------------------------
# Figure 1: temperature and refractive-index profiles
# --------------------------------------------------------------------------
def _figure_profiles(params: AtmosphereParams, out: Path) -> dict:
    y = np.linspace(0.0, 2.0, 500)
    T = temperature(y, params)
    n_excess = (refractive_index(y, params) - 1.0) * 1e6
    index_gradient = dn_dy(y, params) * 1e6

    fig, (ax_t, ax_n, ax_g) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    ax_t.plot(y, T, color="tab:red", lw=2)
    ax_t.axhline(
        params.ambient_temp_K,
        color="gray",
        ls=":",
        lw=1,
        label=f"ambient = {params.ambient_temp_K:.2f} K",
    )
    ax_t.set_ylabel("Temperature T (K)")
    ax_t.set_title("Air temperature profile above a heated surface")
    ax_t.grid(alpha=0.3)
    ax_t.legend()

    ax_n.plot(y, n_excess, color="tab:blue", lw=2)
    ax_n.set_ylabel(r"Refractive index variation $(n-1)\times10^{6}$")
    ax_n.set_title("Refractive index profile")
    ax_n.grid(alpha=0.3)
    ax_n.set_xlim(0.0, 2.0)

    ax_g.plot(y, index_gradient, color="tab:green", lw=2)
    ax_g.axhline(0.0, color="gray", ls=":", lw=1)
    ax_g.set_xlabel("Height above ground y (m)")
    ax_g.set_ylabel(r"$(dn/dy)\times10^{6}$ (m$^{-1}$)")
    ax_g.set_title("Vertical refractive-index gradient")
    ax_g.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out / FILE_1, dpi=DPI)
    plt.close(fig)

    return {
        "T_surface": float(T[0]),
        "T_top": float(T[-1]),
        "n_excess_surface": float(n_excess[0]),
        "n_excess_top": float(n_excess[-1]),
    }


# --------------------------------------------------------------------------
# Figure 2: ray trajectories
# --------------------------------------------------------------------------
def _plot_path(ax, path, **kwargs):
    line, = ax.plot(path.x_m, path.y_m, **kwargs)
    if path.hit_ground:
        ax.plot(
            path.x_m[-1], 0.0, marker="x", color=line.get_color(),
            ms=8, mew=2, ls="none",
        )
    return line


def _figure_trajectories(
    params: AtmosphereParams,
    out: Path,
    ray_specs: Sequence[tuple[str, float, float]],
    x_max_m: float,
    max_step_m: float,
) -> dict:
    """Plot the configured rays and a uniform-medium control for ray 1."""
    if not ray_specs:
        raise ValueError("At least one ray specification is required")

    first_id, first_y0, first_angle = ray_specs[0]
    uniform = replace(params, surface_temp_excess_K=0.0)
    ray_uniform = trace_ray(
        uniform, first_y0, first_angle, x_max_m, max_step_m
    )

    ray_paths = {}
    for ray_id, y0, angle in ray_specs:
        ray_paths[ray_id] = trace_ray(
            params, y0, angle, x_max_m, max_step_m
        )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    _plot_path(
        ax,
        ray_uniform,
        color="black",
        ls="--",
        lw=1.8,
        label=f"Uniform medium (ΔT = 0 K), {first_id}",
    )

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["tab:blue"])
    for i, (ray_id, _y0, angle) in enumerate(ray_specs):
        _plot_path(
            ax,
            ray_paths[ray_id],
            color=colors[i % len(colors)],
            lw=1.8,
            label=(
                f"Heated medium, {ray_id} "
                f"(θ₀ = {angle:g}°, y₀ = {_y0:g} m)"
            ),
        )

    ax.axhline(0.0, color="saddlebrown", lw=2)
    ax.fill_between(
        [0.0, x_max_m], -1.0, 0.0, color="saddlebrown", alpha=0.25,
        label="Ground (y = 0)",
    )

    all_paths = [ray_uniform, *ray_paths.values()]
    top = max(float(np.max(p.y_m)) for p in all_paths)
    ax.set_xlim(0.0, x_max_m)
    ax.set_ylim(-0.03 * top, 1.1 * top)
    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel("Height above ground y (m)")
    ax.set_title("Ray trajectories in uniform and vertically graded media")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out / FILE_2, dpi=DPI)
    plt.close(fig)

    return {"uniform": ray_uniform, "rays": ray_paths}


# --------------------------------------------------------------------------
# Figure 3: sensitivity to surface heating
# --------------------------------------------------------------------------
def _figure_sensitivity(
    params: AtmosphereParams,
    out: Path,
    x_eval_m: float,
    excesses: Sequence[float],
    y0_m: float,
    theta0_deg: float,
    max_step_m: float,
) -> list[tuple[float, float | None, float]]:
    rows: list[tuple[float, float | None, float]] = []
    for dT in excesses:
        p = replace(params, surface_temp_excess_K=float(dT))
        path = trace_ray(p, y0_m, theta0_deg, x_eval_m, max_step_m)
        if path.hit_ground:
            rows.append((float(dT), None, float(path.x_m[-1])))
        else:
            height = float(np.interp(x_eval_m, path.x_m, path.y_m))
            rows.append((float(dT), height, float(path.x_m[-1])))

    fig, ax = plt.subplots(figsize=(7.5, 5))
    reached = [(dT, h) for dT, h, _ in rows if h is not None]
    grounded = [(dT, x_end) for dT, h, x_end in rows if h is None]

    if reached:
        ax.plot(
            *zip(*reached),
            "o-",
            color="tab:blue",
            lw=2,
            label=f"Ray height at x = {x_eval_m:g} m",
        )
    if grounded:
        ax.plot(
            [dT for dT, _ in grounded],
            np.zeros(len(grounded)),
            "s",
            mfc="none",
            mec="tab:red",
            ms=9,
            ls="none",
            label=f"Ray hit ground before x = {x_eval_m:g} m",
        )
    ax.axhline(0.0, color="saddlebrown", lw=2, label="Ground (y = 0)")

    ax.set_xlabel("Surface temperature excess ΔT (K)")
    ax.set_ylabel(f"Ray height at x = {x_eval_m:g} m (m)")
    ax.set_title(
        f"Sensitivity to surface heating (y₀ = {y0_m:g} m, "
        f"θ₀ = {theta0_deg:g}°)"
    )
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out / FILE_3, dpi=DPI)
    plt.close(fig)
    return rows


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def run_all(output_dir: str | Path = "outputs", config=None) -> list[Path]:
    """Generate the three core figures.

    If ``config`` is supplied, it must be the object returned by
    ``config_io.load_config``.  If omitted, backward-compatible defaults are
    used (the original standalone behaviour).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if config is None:
        params = AtmosphereParams()
        x_max_m = 120.0
        max_step_m = 0.25
        ray_specs = (
            ("ray_01", 0.30, -0.30),
            ("ray_02", 0.30, -0.20),
            ("ray_03", 0.30, 0.0),
        )
        sensitivity_distance_m = 50.0
        sensitivity_values = tuple(float(v) for v in np.arange(0.0, 50.0 + 1e-9, 5.0))
    else:
        params = config.atmosphere
        x_max_m = config.simulation.x_max_m
        max_step_m = config.simulation.max_step_m
        ray_specs = tuple(
            (ray.ray_id, ray.y0_m, ray.theta0_deg)
            for ray in config.simulation.rays
        )
        sensitivity_distance_m = config.sensitivity.distance_m
        sensitivity_values = config.sensitivity.surface_temp_excess_values_K

    prof = _figure_profiles(params, out)
    traj = _figure_trajectories(
        params, out, ray_specs, x_max_m=x_max_m, max_step_m=max_step_m
    )

    first_id, first_y0, first_angle = ray_specs[0]
    sens = _figure_sensitivity(
        params,
        out,
        x_eval_m=sensitivity_distance_m,
        excesses=sensitivity_values,
        y0_m=first_y0,
        theta0_deg=first_angle,
        max_step_m=max_step_m,
    )

    print("=" * 68)
    print("Ray-bending experiment summary")
    print("=" * 68)
    print(
        f"Parameters: T_a = {params.ambient_temp_K} K, "
        f"ΔT = {params.surface_temp_excess_K} K, "
        f"H = {params.scale_height_m} m, n_ref = {params.n_ref}"
    )
    print()
    print("[Fig 1] Profiles (0 to 2 m)")
    print(
        f"  T(0) = {prof['T_surface']:.2f} K, T(2 m) = {prof['T_top']:.2f} K"
    )
    print(
        f"  (n-1)e6 at 0 m = {prof['n_excess_surface']:.2f}, "
        f"at 2 m = {prof['n_excess_top']:.2f}"
    )
    print()
    print(f"[Fig 2] Configured rays (x up to {x_max_m:g} m)")

    def describe(label, path):
        if path.hit_ground:
            print(f"  {label:<34s} hits ground at x = {path.x_m[-1]:7.2f} m")
        else:
            print(
                f"  {label:<34s} no ground hit; y(end) = {path.y_m[-1]:.4f} m, "
                f"min y = {np.min(path.y_m):.4f} m"
            )

    describe(f"uniform control ({first_id})", traj["uniform"])
    for ray_id, path in traj["rays"].items():
        describe(ray_id, path)
    print()
    print(
        f"[Fig 3] Ray height at x = {sensitivity_distance_m:g} m "
        f"vs surface temperature excess ({first_id})"
    )
    print("  ΔT (K)   height (m) / outcome")
    for dT, h, x_end in sens:
        if h is None:
            print(f"  {dT:6.1f}   ground hit at x = {x_end:.2f} m")
        else:
            print(f"  {dT:6.1f}   {h:.6f}")
    print("=" * 68)

    paths = [out / FILE_1, out / FILE_2, out / FILE_3]
    print("Saved:")
    for p in paths:
        print(f"  {p}")
    return paths


if __name__ == "__main__":
    run_all()
