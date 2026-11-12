"""End-to-end checks for the config-driven project entry point."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from config_io import ConfigError, load_config
from main import run_project

PROJECT_DIR = Path(__file__).resolve().parents[1]


class ConfigDrivenPipelineTests(unittest.TestCase):
    def test_default_input_is_valid(self):
        config = load_config(PROJECT_DIR / "input.json")
        self.assertEqual(len(config.simulation.rays), 3)
        self.assertGreater(config.atmosphere.scale_height_m, 0)

    def test_full_pipeline_creates_expected_outputs(self):
        config = load_config(PROJECT_DIR / "input.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            out = Path(temp_dir) / "outputs"
            paths = run_project(config, out)

            expected = {
                "01_temperature_and_index.png",
                "02_ray_trajectories.png",
                "03_surface_temperature_sensitivity.png",
                "04_mirage_visualisation.png",
                "05_invariant_validation.png",
                "06_convergence.png",
                "07_critical_launch_angle.png",
                "08_heated_uniform_cooled.png",
                "additional_results.txt",
                "ray_paths.csv",
                "summary.txt",
                "validation.txt",
                "mirage_status.txt",
            }
            self.assertTrue(expected.issubset({p.name for p in paths}))
            for name in expected:
                self.assertTrue((out / name).is_file(), name)
                self.assertGreater((out / name).stat().st_size, 0, name)

            with (out / "ray_paths.csv").open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, ["ray_id", "x_m", "y_m", "slope"])
                rows = list(reader)
            self.assertGreater(len(rows), 3)
            self.assertEqual({row["ray_id"] for row in rows}, {"ray_01", "ray_02", "ray_03"})

            summary = (out / "summary.txt").read_text(encoding="utf-8")
            self.assertIn("Ray ID: ray_01", summary)
            self.assertIn("Minimum ray height", summary)
            self.assertIn("X position of sampled minimum", summary)
            self.assertIn("Turning point detected", summary)

            additional = (out / "additional_results.txt").read_text(encoding="utf-8")
            self.assertIn("CRITICAL DOWNWARD LAUNCH ANGLE", additional)
            self.assertIn("THEORETICAL CRITICAL ANGLE", additional.upper())

            validation = (out / "validation.txt").read_text(encoding="utf-8")
            self.assertIn("Maximum relative drift", validation)
            self.assertIn("STEP-SIZE CONVERGENCE", validation)

    def test_invalid_mirage_height_is_rejected_in_config(self):
        source = json.loads((PROJECT_DIR / "input.json").read_text(encoding="utf-8"))
        source["mirage"]["source_y_m"] = 0.0
        with tempfile.TemporaryDirectory() as temp_dir:
            p = Path(temp_dir) / "invalid.json"
            p.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(p)


if __name__ == "__main__":
    unittest.main()
