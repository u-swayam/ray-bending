"""Numerical and physical validation of the ray-tracing model.

Checks performed (the optical model and ray equation are NOT modified):

1. Optical invariant.  For n = n(y) the quantity

       I(x) = n(y) * cos(theta) = n(y) / sqrt(1 + p**2),   p = dy/dx

   is conserved along a ray (Snell's law for a stratified medium).  The
   relative drift  eps(x) = |I(x)/I(0) - 1|  measures integration error.

2. Convergence study. The same ray is traced with progressively tighter
   DOP853 error tolerances and maximum-step settings; heights are compared
   against a separate, tighter reference integration.

Public interface:
    optical_invariant(path, params)            -> ndarray
    relative_drift(invariant)                  -> ndarray
    invariant_study(params, ...)               -> InvariantResult
    convergence_study(params, ...)             -> ConvergenceResult
    plot_invariant(result, path)               -> Path
    plot_convergence(result, path)             -> Path
    write_validation_txt(inv, conv, params, path) -> Path
    run_validation(params, ...)                -> ValidationResults

Run standalone:  python validation.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # file output only, no GUI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from atmosphere import AtmosphereParams, refractive_index  # noqa: E402
from ray_solver import RayPath, trace_ray  # noqa: E402

__all__ = [
    "InvariantResult",
    "ConvergenceRow",
    "ConvergenceResult",
    "ValidationResults",
    "optical_invariant",
    "relative_drift",
    "invariant_study",
    "convergence_study",
    "plot_invariant",
    "plot_convergence",
    "write_validation_txt",
    "run_validation",
]

DEFAULT_STEPS_M = (0.25, 0.10, 0.05)
INVARIANT_FORMULA = "I(x) = n(y) * cos(theta) = n(y) / sqrt(1 + p^2),  p = dy/dx"
FILE_INVARIANT = "05_invariant_validation.png"
FILE_CONVERGENCE = "06_convergence.png"
FILE_REPORT = "validation.txt"
DPI = 150


# ------------------------------------------------------------------ results
@dataclass(frozen=True)
class InvariantResult:
    x_m: np.ndarray
    invariant: np.ndarray
    drift: np.ndarray  # epsilon(x) = |I(x)/I(0) - 1|
    max_drift: float
    mean_drift: float
    rms_drift: float
    hit_ground: bool
    y0_m: float
    theta0_deg: float
    x_max_m: float
    max_step_m: float


@dataclass(frozen=True)
class ConvergenceRow:
    step_m: float
    rtol: float
    atol: float
    height_m: float | None  # None if the ray hit the ground before x_eval
    abs_error_m: float | None  # absolute difference from high-accuracy reference
    hit_ground: bool
    x_end_m: float  # where the trace ended (x_eval or the ground impact)
    n_points: int


@dataclass(frozen=True)
class ConvergenceResult:
    x_eval_m: float
    y0_m: float
    theta0_deg: float
    rows: tuple[ConvergenceRow, ...]
    reference_height_m: float | None
    reference_rtol: float
    reference_atol: float
    reference_max_step_m: float

    def successive_differences(self) -> list[tuple[float, float, float | None]]:
        """(coarser_step, finer_step, finer_height - coarser_height).

        The difference is None when either run has no height at x_eval.
        """
        out = []
        for a, b in zip(self.rows[:-1], self.rows[1:]):
            if a.height_m is None or b.height_m is None:
                out.append((a.step_m, b.step_m, None))
            else:
                out.append((a.step_m, b.step_m, b.height_m - a.height_m))
        return out


@dataclass(frozen=True)
class ValidationResults:
    invariant: InvariantResult
    convergence: ConvergenceResult
    files: tuple[Path, ...]


# --------------------------------------------------------------- invariant
def optical_invariant(path: RayPath, params: AtmosphereParams) -> np.ndarray:
    """I = n(y) / sqrt(1 + p^2) at every point of a traced ray.

    n(y) comes from atmosphere.refractive_index; p is the solver's slope.
    """
    n = np.asarray(refractive_index(np.asarray(path.y_m, dtype=float), params))
    return n / np.sqrt(1.0 + np.asarray(path.slope, dtype=float) ** 2)


def relative_drift(invariant: np.ndarray) -> np.ndarray:
    """epsilon(x) = |I(x)/I(0) - 1|."""
    invariant = np.asarray(invariant, dtype=float)
    if invariant.size == 0 or not np.isfinite(invariant[0]) or invariant[0] == 0:
        raise ValueError("invariant must be non-empty with a finite, non-zero I(0)")
    return np.abs(invariant / invariant[0] - 1.0)


def invariant_study(
    params: AtmosphereParams,
    y0_m: float = 0.30,
    theta0_deg: float = -0.30,
    x_max_m: float = 120.0,
    max_step_m: float = 0.25,
) -> InvariantResult:
    """Trace the reference ray and measure the drift of I(x)."""
    path = trace_ray(params, y0_m, theta0_deg, x_max_m, max_step_m)
    inv = optical_invariant(path, params)
    eps = relative_drift(inv)
    return InvariantResult(
        x_m=np.asarray(path.x_m, dtype=float),
        invariant=inv,
        drift=eps,
        max_drift=float(np.max(eps)),
        mean_drift=float(np.mean(eps)),
        rms_drift=float(np.sqrt(np.mean(eps**2))),
        hit_ground=bool(path.hit_ground),
        y0_m=float(y0_m),
        theta0_deg=float(theta0_deg),
        x_max_m=float(x_max_m),
        max_step_m=float(max_step_m),
    )


# ------------------------------------------------------------- convergence
def convergence_study(
    params: AtmosphereParams,
    y0_m: float = 0.30,
    theta0_deg: float = -0.30,
    x_eval_m: float = 50.0,
    steps_m: Sequence[float] = DEFAULT_STEPS_M,
    solver_tolerances: Sequence[tuple[float, float]] | None = None,
) -> ConvergenceResult:
    """Compare solver settings against an independently tighter reference.

    Each entry in ``steps_m`` is paired with an (rtol, atol) pair. If no
    tolerance pairs are supplied, the study uses progressively tighter
    tolerances from 1e-4/1e-7 to 1e-10/1e-12. The reference uses still tighter
    tolerances and a smaller maximum step.
    """
    steps = [float(s) for s in steps_m]
    if len(steps) < 2:
        raise ValueError("steps_m needs at least two step sizes")
    if any(not np.isfinite(s) or s <= 0 for s in steps):
        raise ValueError(f"all step sizes must be finite and > 0, got {steps_m!r}")
    if len(set(steps)) != len(steps):
        raise ValueError("step sizes must be distinct")
    if not np.isfinite(x_eval_m) or x_eval_m <= 0:
        raise ValueError(f"x_eval_m must be finite and > 0, got {x_eval_m!r}")

    if solver_tolerances is None:
        # Geometric progression works for any number of requested runs.
        rtol_values = np.geomspace(1e-2, 1e-8, num=len(steps))
        tolerance_pairs = [(float(r), float(max(1e-14, r * 1e-3))) for r in rtol_values]
    else:
        tolerance_pairs = [(float(r), float(a)) for r, a in solver_tolerances]
        if len(tolerance_pairs) != len(steps):
            raise ValueError("solver_tolerances must have one (rtol, atol) pair per step")
        for rtol, atol in tolerance_pairs:
            if not np.isfinite(rtol) or rtol <= 0 or not np.isfinite(atol) or atol <= 0:
                raise ValueError("all solver tolerances must be finite and > 0")

    reference_rtol = 1e-13
    reference_atol = 1e-15
    reference_step = min(steps) * 0.1
    reference_path = trace_ray(
        params, y0_m, theta0_deg, x_eval_m, reference_step,
        rtol=reference_rtol, atol=reference_atol,
    )
    reference_height = None
    if not reference_path.hit_ground and reference_path.x_m[-1] >= x_eval_m * (1.0 - 1e-12):
        reference_height = float(np.interp(x_eval_m, reference_path.x_m, reference_path.y_m))

    rows = []
    for step, (rtol, atol) in zip(steps, tolerance_pairs):
        path = trace_ray(
            params, y0_m, theta0_deg, x_eval_m, step, rtol=rtol, atol=atol
        )
        x_end = float(path.x_m[-1])
        if path.hit_ground or x_end < x_eval_m * (1.0 - 1e-12):
            height = None
        else:
            height = float(np.interp(x_eval_m, path.x_m, path.y_m))
        error = (
            abs(height - reference_height)
            if height is not None and reference_height is not None
            else None
        )
        rows.append(
            ConvergenceRow(
                step_m=step,
                rtol=rtol,
                atol=atol,
                height_m=height,
                abs_error_m=error,
                hit_ground=bool(path.hit_ground),
                x_end_m=x_end,
                n_points=int(len(path.x_m)),
            )
        )
    return ConvergenceResult(
        x_eval_m=float(x_eval_m),
        y0_m=float(y0_m),
        theta0_deg=float(theta0_deg),
        rows=tuple(rows),
        reference_height_m=reference_height,
        reference_rtol=reference_rtol,
        reference_atol=reference_atol,
        reference_max_step_m=reference_step,
    )


# ------------------------------------------------------------------- plots
def plot_invariant(result: InvariantResult, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(result.x_m, result.drift, color="tab:purple", lw=1.6)
    ax.set_xlabel("Horizontal distance x (m)")
    ax.set_ylabel(r"Relative invariant drift $|I(x)/I(0) - 1|$")
    ax.set_title(
        f"Optical invariant drift (y0 = {result.y0_m:g} m, "
        f"theta0 = {result.theta0_deg:g} deg, max_step = {result.max_step_m:g} m)"
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    ax.grid(alpha=0.3)
    ax.text(
        0.02, 0.95,
        f"max  = {result.max_drift:.3e}\nmean = {result.mean_drift:.3e}\n"
        f"RMS  = {result.rms_drift:.3e}",
        transform=ax.transAxes, va="top", family="monospace", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
    )
    fig.tight_layout()
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def plot_convergence(result: ConvergenceResult, path: str | Path) -> Path:
    """Plot candidate heights and their errors against a high-accuracy run."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    valid = [r for r in result.rows if r.height_m is not None]
    labels = [f"Run {i + 1}" for i, _ in enumerate(result.rows)]
    x_all = np.arange(len(result.rows))

    fig, (ax_h, ax_e) = plt.subplots(2, 1, figsize=(8.2, 7.2))
    valid_indices = [i for i, r in enumerate(result.rows) if r.height_m is not None]
    if valid:
        heights = np.asarray([r.height_m for r in valid], dtype=float)
        ax_h.plot(valid_indices, heights, "o-", color="tab:blue", lw=1.8,
                  label="Numerical runs")
        if result.reference_height_m is not None:
            ax_h.axhline(
                result.reference_height_m,
                color="black", ls="--", lw=1.3,
                label=f"High-accuracy reference = {result.reference_height_m:.12f} m",
            )
        span = float(np.ptp(heights)) if heights.size else 0.0
        pad = max(span * 0.35, 2e-10)
        center = float(np.mean(heights))
        ax_h.set_ylim(center - pad, center + pad)
        ax_h.ticklabel_format(axis="y", useOffset=False, style="plain")
        ax_h.legend(fontsize=8)

    ax_h.set_xticks(x_all, labels)
    ax_h.set_ylabel(f"Ray height at x = {result.x_eval_m:g} m (m)")
    ax_h.set_title("Convergence of the calculated ray height")
    ax_h.grid(alpha=0.3)

    errors = [r.abs_error_m for r in result.rows]
    finite_errors = [e for e in errors if e is not None]
    if finite_errors:
        floor = np.finfo(float).eps * max(abs(result.reference_height_m or 1.0), 1.0)
        plotted = [max(float(e), floor) if e is not None else np.nan for e in errors]
        ax_e.semilogy(x_all, plotted, "s-", color="tab:green", lw=1.8)
        ax_e.axhline(floor, color="gray", ls=":", lw=1.0,
                     label="Approximate double-precision roundoff scale")
        ax_e.legend(fontsize=8)
    else:
        ax_e.text(0.5, 0.5, "No runs reached the comparison distance",
                  ha="center", va="center", transform=ax_e.transAxes)
    ax_e.set_xticks(x_all, labels)
    ax_e.set_ylabel("Absolute error vs reference (m)")
    ax_e.set_xlabel("Solver configuration")
    ax_e.set_title("Error relative to a tighter reference solution")
    ax_e.grid(alpha=0.3, which="both")

    settings_text = "\n".join(
        f"{labels[i]}: hmax={r.step_m:g} m, rtol={r.rtol:g}, atol={r.atol:g}"
        for i, r in enumerate(result.rows)
    )
    fig.text(0.5, 0.005, settings_text, ha="center", va="bottom", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ report
def write_validation_txt(
    inv: InvariantResult,
    conv: ConvergenceResult,
    params: AtmosphereParams,
    path: str | Path,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    L = [
        "RAY-TRACING VALIDATION REPORT",
        "=" * 64,
        "",
        "Atmosphere: "
        f"T_a={params.ambient_temp_K}, dT={params.surface_temp_excess_K}, "
        f"H={params.scale_height_m}, n_ref={params.n_ref}, "
        f"T_ref={params.reference_temp_K}",
        "",
        "1. OPTICAL INVARIANT",
        "-" * 64,
        f"Formula: {INVARIANT_FORMULA}",
        "Relative drift: eps(x) = |I(x)/I(0) - 1|",
        f"Ray: y0 = {inv.y0_m:g} m, theta0 = {inv.theta0_deg:g} deg, "
        f"x_max = {inv.x_max_m:g} m, max_step = {inv.max_step_m:g} m",
        f"Samples: {len(inv.x_m)};  ray ended at x = {inv.x_m[-1]:.4f} m"
        + ("  (hit ground)" if inv.hit_ground else ""),
        f"I(0)                  = {inv.invariant[0]:.15f}",
        f"Maximum relative drift = {inv.max_drift:.6e}",
        f"Mean relative drift    = {inv.mean_drift:.6e}",
        f"RMS relative drift     = {inv.rms_drift:.6e}",
        "",
        "2. SOLVER-TOLERANCE AND STEP-SIZE CONVERGENCE",
        "-" * 64,
        f"Ray: y0 = {conv.y0_m:g} m, theta0 = {conv.theta0_deg:g} deg; "
        f"height compared at x = {conv.x_eval_m:g} m",
        f"Reference height = {conv.reference_height_m:.15f} m"
        if conv.reference_height_m is not None else "Reference height = N/A (reference ray hit ground)",
        f"Reference settings: rtol={conv.reference_rtol:g}, atol={conv.reference_atol:g}, "
        f"max_step={conv.reference_max_step_m:g} m",
        "",
        f"{'run':>5s} {'max_step(m)':>12s} {'rtol':>11s} {'atol':>11s} "
        f"{'height(m)':>20s} {'abs_error(m)':>16s}",
    ]
    for i, r in enumerate(conv.rows, start=1):
        h = "N/A" if r.height_m is None else f"{r.height_m:.15f}"
        e = "N/A" if r.abs_error_m is None else f"{r.abs_error_m:.6e}"
        L.append(
            f"{i:5d} {r.step_m:12.5g} {r.rtol:11.3g} {r.atol:11.3g} "
            f"{h:>20s} {e:>16s}"
        )
    L += [
        "",
        "The candidate runs vary both the adaptive solver tolerances and the",
        "maximum integration step. Errors are measured against the tighter",
        "reference solution above. Values at or below floating-point roundoff",
        "should not be interpreted as physically meaningful differences.",
    ]
    L.append("")
    p.write_text("\n".join(L), encoding="utf-8")
    return p


# ------------------------------------------------------------------ driver
def run_validation(
    params: AtmosphereParams | None = None,
    y0_m: float = 0.30,
    theta0_deg: float = -0.30,
    x_max_m: float = 120.0,
    x_eval_m: float = 50.0,
    steps_m: Sequence[float] = DEFAULT_STEPS_M,
    output_dir: str | Path = "outputs",
    invariant_max_step_m: float = 0.25,
    solver_tolerances: Sequence[tuple[float, float]] | None = None,
) -> ValidationResults:
    """Run both studies, write both figures and validation.txt, print a summary."""
    params = AtmosphereParams() if params is None else params
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    inv = invariant_study(params, y0_m, theta0_deg, x_max_m, invariant_max_step_m)
    conv = convergence_study(
        params, y0_m, theta0_deg, x_eval_m, steps_m, solver_tolerances
    )

    files = (
        plot_invariant(inv, out / FILE_INVARIANT),
        plot_convergence(conv, out / FILE_CONVERGENCE),
        write_validation_txt(inv, conv, params, out / FILE_REPORT),
    )

    print("Validation summary")
    print(f"  invariant drift: max={inv.max_drift:.3e} mean={inv.mean_drift:.3e} "
          f"rms={inv.rms_drift:.3e}")
    for i, r in enumerate(conv.rows, start=1):
        v = "N/A (ground hit)" if r.height_m is None else f"{r.height_m:.12f} m"
        error = "N/A" if r.abs_error_m is None else f"{r.abs_error_m:.3e} m"
        print(
            f"  run {i}: hmax={r.step_m:g} m, rtol={r.rtol:g}, atol={r.atol:g}; "
            f"height at {conv.x_eval_m:g} m = {v}; reference error = {error}"
        )
    for a, b, d in conv.successive_differences():
        print(f"  diff {a:g}->{b:g}: " + ("N/A" if d is None else f"{d:+.3e} m"))
    print("Saved:", *[str(f) for f in files], sep="\n  ")
    return ValidationResults(invariant=inv, convergence=conv, files=files)


if __name__ == "__main__":
    run_validation()