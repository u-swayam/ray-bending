"""Integration and physics tests (standard-library unittest, no extra deps).

Run:  python -m pytest   (or: python -m unittest discover -s tests -v)
"""

import inspect
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import experiments  # noqa: E402
from atmosphere import AtmosphereParams, dn_dy, refractive_index, temperature  # noqa: E402
from ray_solver import RayPath, trace_ray  # noqa: E402


class InterfaceTests(unittest.TestCase):
    def test_trace_ray_signature(self):
        names = list(inspect.signature(trace_ray).parameters)
        self.assertEqual(names[:4], ["atmosphere_params", "y0_m", "theta0_deg", "x_max_m"])

    def test_raypath_fields(self):
        p = trace_ray(AtmosphereParams(), x_max_m=10.0)
        self.assertIsInstance(p, RayPath)
        self.assertEqual(p.x_m.shape, p.y_m.shape)
        self.assertEqual(p.x_m.shape, p.slope.shape)
        self.assertIsInstance(p.hit_ground, bool)

    def test_run_all_exists(self):
        self.assertTrue(callable(experiments.run_all))


class AtmosphereTests(unittest.TestCase):
    def test_scalar_returns_float(self):
        self.assertIsInstance(temperature(0.5), float)
        self.assertIsInstance(refractive_index(0.5), float)
        self.assertIsInstance(dn_dy(0.5), float)

    def test_array_shape(self):
        y = np.linspace(0, 1, 7)
        self.assertEqual(refractive_index(y).shape, y.shape)

    def test_surface_and_far_field(self):
        p = AtmosphereParams()
        self.assertAlmostEqual(temperature(0.0, p), p.ambient_temp_K + p.surface_temp_excess_K)
        self.assertAlmostEqual(temperature(100.0, p), p.ambient_temp_K, places=6)

    def test_gradient_matches_finite_difference(self):
        p = AtmosphereParams()
        y, h = 0.3, 1e-6
        fd = (refractive_index(y + h, p) - refractive_index(y - h, p)) / (2 * h)
        self.assertAlmostEqual(dn_dy(y, p), fd, delta=1e-9)

    def test_uniform_gradient_zero(self):
        p = replace(AtmosphereParams(), surface_temp_excess_K=0.0)
        self.assertEqual(dn_dy(0.2, p), 0.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            temperature(-1.0)
        with self.assertRaises(ValueError):
            AtmosphereParams(scale_height_m=0.0)


class RayTests(unittest.TestCase):
    def setUp(self):
        self.heated = AtmosphereParams()
        self.uniform = replace(self.heated, surface_temp_excess_K=0.0)

    def test_uniform_ray_is_straight(self):
        p = trace_ray(self.uniform, 0.30, -0.30, 40.0)
        expected = 0.30 + p.x_m * np.tan(np.deg2rad(-0.30))
        np.testing.assert_allclose(p.y_m, expected, atol=1e-9)
        np.testing.assert_allclose(p.slope, np.tan(np.deg2rad(-0.30)), atol=1e-12)

    def test_heated_ray_is_curved(self):
        p = trace_ray(self.heated, 0.30, -0.30, 40.0)
        straight = 0.30 + p.x_m * np.tan(np.deg2rad(-0.30))
        self.assertGreater(np.max(np.abs(p.y_m - straight)), 1e-3)
        self.assertGreater(np.ptp(p.slope), 1e-4)  # slope changes along the path
        self.assertGreater(p.slope[-1], p.slope[0])  # bends upward (mirage)

    def test_ground_hit_terminates_at_ground(self):
        p = trace_ray(self.uniform, 0.30, -0.30, 120.0)
        self.assertTrue(p.hit_ground)
        self.assertEqual(p.y_m[-1], 0.0)
        self.assertAlmostEqual(p.x_m[-1], 0.30 / np.tan(np.deg2rad(0.30)), places=4)
        self.assertLess(p.x_m[-1], 120.0)
        self.assertTrue(np.all(p.y_m[:-1] > 0))

    def test_no_ground_hit_runs_to_end(self):
        p = trace_ray(self.heated, 0.30, 0.0, 120.0)
        self.assertFalse(p.hit_ground)
        self.assertAlmostEqual(p.x_m[-1], 120.0)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            trace_ray(self.heated, y0_m=0.0)
        with self.assertRaises(ValueError):
            trace_ray(self.heated, theta0_deg=90.0)
        with self.assertRaises(TypeError):
            trace_ray("not params")


class FigureTests(unittest.TestCase):
    def test_run_all_creates_three_figures(self):
        with tempfile.TemporaryDirectory() as d:
            paths = experiments.run_all(d)
            self.assertEqual(len(paths), 3)
            for p in paths:
                self.assertTrue(Path(p).is_file())
                self.assertGreater(Path(p).stat().st_size, 10_000)


if __name__ == "__main__":
    unittest.main()