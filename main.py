"""Run the complete configurable ray-bending project.

Default usage from the project directory::

    python main.py

Optional paths::

    python main.py --config input.json --output-dir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config_io import Config, ConfigError, load_config
from additional_experiments import run_additional_experiments
from experiments import run_all
from mirage import MirageError, run_mirage_from_config
from output_writer import write_ray_paths_csv, write_summary_txt
from ray_solver import trace_ray
from validation import run_validation


def run_project(config: Config, output_dir: str | Path = "outputs") -> list[Path]:
    """Run all configured experiments and save figures and numerical outputs."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sim = config.simulation
    first_ray = sim.rays[0]

    print("\n[1/6] Running configured ray-trajectory experiments...")
    figure_paths = run_all(output_dir=out, config=config)

    print("\n[2/6] Tracing configured rays for numerical export...")
    ray_results = {}
    for spec in sim.rays:
        ray_results[spec.ray_id] = trace_ray(
            config.atmosphere,
            y0_m=spec.y0_m,
            theta0_deg=spec.theta0_deg,
            x_max_m=sim.x_max_m,
            max_step_m=sim.max_step_m,
        )

    csv_path = write_ray_paths_csv(ray_results, out / "ray_paths.csv")
    summary_path = write_summary_txt(ray_results, config, out / "summary.txt")
    print(f"Saved ray coordinates: {csv_path}")
    print(f"Saved numerical summary: {summary_path}")

    print("\n[3/6] Checking the optical invariant and numerical convergence...")
    # Preserve the configured maximum step as the coarsest run, then tighten it.
    comparison_distance = config.sensitivity.distance_m
    step_sizes = (
        comparison_distance / 2.5,
        comparison_distance / 5.0,
        comparison_distance / 10.0,
    )
    solver_tolerances = (
        (1e-2, 1e-5),
        (1e-4, 1e-7),
        (1e-8, 1e-11),
    )
    validation_results = run_validation(
        params=config.atmosphere,
        y0_m=first_ray.y0_m,
        theta0_deg=first_ray.theta0_deg,
        x_max_m=sim.x_max_m,
        x_eval_m=config.sensitivity.distance_m,
        steps_m=step_sizes,
        output_dir=out,
        invariant_max_step_m=sim.max_step_m,
        solver_tolerances=solver_tolerances,
    )

    print("\n[4/6] Building the atmospheric-mirage ray diagram...")
    mirage_status_path = out / "mirage_status.txt"
    mirage_path = out / "04_mirage_visualisation.png"
    mirage_created = False
    try:
        run_mirage_from_config(config, output_dir=out)
        mirage_status_path.write_text(
            "Mirage ray diagram generated successfully.\n"
            "Interpretation: geometrical-optics illustration of a source-to-observer "
            "ray and its backward tangent; not a complete photographic mirage model.\n",
            encoding="utf-8",
        )
        mirage_created = mirage_path.is_file()
    except MirageError as exc:
        message = (
            "The configured source-observer geometry did not produce a valid ray.\n\n"
            f"Reason: {exc}\n\n"
            "The other ray-tracing experiments and validation outputs were completed. "
            "Adjust the mirage source height, observer height, distance, or the "
            "mirage solver's angle search range if this figure is required.\n"
        )
        mirage_status_path.write_text(message, encoding="utf-8")
        print(f"Mirage diagram not generated: {exc}")

    print("\n[5/6] Running critical-angle and thermal-gradient comparisons...")
    additional_paths = run_additional_experiments(config, out)

    print("\n[6/6] Integration run complete.")
    output_paths = [
        *figure_paths,
        csv_path,
        summary_path,
        *validation_results.files,
        *additional_paths,
        mirage_status_path,
    ]
    if mirage_created:
        output_paths.append(mirage_path)

    print("\nAll generated outputs:")
    for path in output_paths:
        print(f"  {path}")
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Run the graded-index atmospheric ray-bending simulation."
    )
    parser.add_argument(
        "--config",
        default=str(project_dir / "input.json"),
        help="JSON input file (default: input.json next to main.py)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(project_dir / "outputs"),
        help="directory for figures and numerical outputs (default: outputs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        run_project(config, args.output_dir)
    except (ConfigError, OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
