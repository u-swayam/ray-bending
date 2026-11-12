"""Write ray-tracing results to CSV and plain-text summary files.

Public interface:
    write_ray_paths_csv(rays, path="outputs/ray_paths.csv") -> Path
    write_summary_txt(rays, config, path="outputs/summary.txt") -> Path

``rays`` maps ray_id -> RayPath (as returned by ray_solver.trace_ray). Any
mapping works; insertion order is the output order. A list of
(ray_id, RayPath) pairs is accepted too.
"""

from __future__ import annotations

import csv
from dataclasses import fields
from pathlib import Path
from typing import Iterable, Mapping, Tuple, Union

import numpy as np

from config_io import Config

__all__ = ["write_ray_paths_csv", "write_summary_txt"]

DEFAULT_CSV_PATH = Path("outputs") / "ray_paths.csv"
DEFAULT_SUMMARY_PATH = Path("outputs") / "summary.txt"
CSV_COLUMNS = ("ray_id", "x_m", "y_m", "slope")

RaysInput = Union[Mapping[str, object], Iterable[Tuple[str, object]]]


def _normalize_rays(rays: RaysInput) -> list[tuple[str, object]]:
    """Return a validated list of (ray_id, RayPath-like) pairs."""
    items = list(rays.items()) if isinstance(rays, Mapping) else list(rays)
    if not items:
        raise ValueError("rays is empty: nothing to write")

    seen: set[str] = set()
    out = []
    for item in items:
        try:
            ray_id, path = item
        except (TypeError, ValueError):
            raise TypeError(
                "rays must be a mapping {ray_id: RayPath} or an iterable of "
                f"(ray_id, RayPath) pairs; got element {item!r}"
            ) from None
        if not isinstance(ray_id, str) or not ray_id:
            raise ValueError(f"ray_id must be a non-empty string, got {ray_id!r}")
        if ray_id in seen:
            raise ValueError(f"duplicate ray_id {ray_id!r}")
        seen.add(ray_id)

        for attr in ("x_m", "y_m", "slope", "hit_ground"):
            if not hasattr(path, attr):
                raise TypeError(
                    f"ray {ray_id!r}: expected a RayPath (missing attribute {attr!r})"
                )
        x, y, s = (np.asarray(getattr(path, a), dtype=float) for a in ("x_m", "y_m", "slope"))
        if x.ndim != 1 or x.size == 0 or not (x.shape == y.shape == s.shape):
            raise ValueError(
                f"ray {ray_id!r}: x_m, y_m and slope must be equal-length, "
                "non-empty 1-D arrays"
            )
        out.append((ray_id, path))
    return out


def _prepare_path(path: str | Path) -> Path:
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Cannot create output directory {p.parent}: {exc}") from exc
    return p


# --------------------------------------------------------------------- CSV
def write_ray_paths_csv(rays: RaysInput, path: str | Path = DEFAULT_CSV_PATH) -> Path:
    """Write all ray coordinates to one CSV (columns: ray_id, x_m, y_m, slope).

    Rays are written one after another; the ray_id column identifies each row's
    ray. Floats are written with full (repr) precision. Returns the path.
    """
    items = _normalize_rays(rays)
    p = _prepare_path(path)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)
        for ray_id, ray in items:
            for x, y, s in zip(ray.x_m, ray.y_m, ray.slope):
                writer.writerow([ray_id, float(x), float(y), float(s)])
    return p


# ----------------------------------------------------------------- summary
def _find_turning_point(x: np.ndarray, y: np.ndarray, slope: np.ndarray):
    """Estimate a ray's lowest turning point by interpolating slope=0.

    A launch exactly horizontal and then curving upward is treated as a
    turning point at x=0. A ray that simply starts upward is not.
    """
    if x.size < 2:
        return None

    slope_tol = 1e-12
    if abs(float(slope[0])) <= slope_tol and slope[1] > slope_tol:
        return float(x[0]), float(y[0])

    crossings = np.flatnonzero((slope[:-1] < 0.0) & (slope[1:] >= 0.0))
    if crossings.size == 0:
        return None

    i = int(crossings[0])
    denom = float(slope[i + 1] - slope[i])
    if denom == 0.0:
        return float(x[i]), float(y[i])
    fraction = float(np.clip(-slope[i] / denom, 0.0, 1.0))
    x_turn = float(x[i] + fraction * (x[i + 1] - x[i]))
    y_turn = float(y[i] + fraction * (y[i + 1] - y[i]))
    return x_turn, y_turn


def _summarize(ray) -> dict:
    x = np.asarray(ray.x_m, dtype=float)
    y = np.asarray(ray.y_m, dtype=float)
    s = np.asarray(ray.slope, dtype=float)
    hit = bool(ray.hit_ground)
    min_index = int(np.argmin(y))
    turning_point = _find_turning_point(x, y, s)
    return {
        "hit_ground": hit,
        "ground_x": float(x[-1]) if hit else None,
        "min_y": float(np.min(y)),
        "min_y_x": float(x[min_index]),
        "has_turning_point": turning_point is not None,
        "turning_x": turning_point[0] if turning_point is not None else None,
        "turning_y": turning_point[1] if turning_point is not None else None,
        "final_x": float(x[-1]),
        "final_y": float(y[-1]),
        "final_slope": float(s[-1]),
    }


def write_summary_txt(
    rays: RaysInput, config: Config, path: str | Path = DEFAULT_SUMMARY_PATH
) -> Path:
    """Write a human-readable summary: simulation parameters + one block per ray.

    ``config`` is the object returned by config_io.load_config. Returns the path.
    """
    if not isinstance(config, Config):
        raise TypeError("config must be a config_io.Config instance")
    items = _normalize_rays(rays)
    p = _prepare_path(path)

    launch = {r.ray_id: r for r in config.simulation.rays}
    atm = config.atmosphere
    sim = config.simulation

    lines = [
        "RAY BENDING SIMULATION SUMMARY",
        "=" * 60,
        "",
        "Simulation parameters",
        "-" * 60,
        "Atmosphere:",
    ]
    for f in fields(atm):
        lines.append(f"  {f.name} = {getattr(atm, f.name)!r}")
    lines += [
        "Simulation:",
        f"  x_max_m = {sim.x_max_m!r}",
        f"  max_step_m = {sim.max_step_m!r}",
        f"  number_of_rays = {len(items)}",
        "",
        "Ray results",
        "-" * 60,
    ]

    for ray_id, ray in items:
        r = _summarize(ray)
        spec = launch.get(ray_id)
        lines.append(f"Ray ID: {ray_id}")
        if spec is not None:
            lines.append(f"  Launch height y0 (m):          {spec.y0_m:.6f}")
            lines.append(f"  Launch angle theta0 (deg):     {spec.theta0_deg:.6f}")
        lines += [
            f"  Hit ground:                    {'Yes' if r['hit_ground'] else 'No'}",
            "  Ground-impact distance (m):    "
            + (f"{r['ground_x']:.6f}" if r["hit_ground"] else "N/A"),
            f"  Minimum ray height (m):        {r['min_y']:.6f}",
            f"  X position of sampled minimum (m): {r['min_y_x']:.6f}",
            f"  Turning point detected:         {'Yes' if r['has_turning_point'] else 'No'}",
            "  Turning point (x, y) (m):       "
            + (f"({r['turning_x']:.6f}, {r['turning_y']:.6f})"
               if r["has_turning_point"] else "N/A"),
            f"  Final horizontal position (m): {r['final_x']:.6f}",
            f"  Final ray height (m):          {r['final_y']:.6f}",
            f"  Final slope (dy/dx):           {r['final_slope']:.6e}",
            "",
        ]

    with p.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return p