# Ray bending in a vertically graded-index medium

A small computational-optics project that models light-ray propagation above
a heated, flat surface. The refractive index varies with height because the
air temperature varies with height. The model is used to demonstrate
continuous ray bending and a simplified atmospheric-mirage line-of-sight
construction.

## Model

The assumed temperature profile is

\[
T(y)=T_a+\Delta T\,e^{-y/H}.
\]

At approximately constant pressure, the model uses the ideal-gas scaling
\(n-1\propto 1/T\):

\[
n(y)=1+(n_{\rm ref}-1)\frac{T_{\rm ref}}{T(y)}.
\]

With \(p=dy/dx\), Fermat's principle gives

\[
\frac{dy}{dx}=p,\qquad
\frac{dp}{dx}=\frac{1}{n(y)}\frac{dn}{dy}(1+p^2).
\]

The ODE is integrated with SciPy's DOP853 method. A terminal event stops a
ray when it reaches the ground (\(y=0\)). The numerical validation checks the
stratified-medium invariant \(n(y)\cos\theta\) and compares solutions at
several maximum integration step sizes.

## Project files

| File | Purpose |
|---|---|
| `atmosphere.py` | Temperature, refractive index and analytic `dn/dy` |
| `ray_solver.py` | DOP853 ray integrator and ground-impact event |
| `experiments.py` | Core profile, trajectory, temperature-sensitivity and refractive-index-gradient figures |
| `additional_experiments.py` | Critical-angle and heated/uniform/cooled comparisons |
| `config_io.py` | Reads and validates `input.json` |
| `output_writer.py` | Writes ray coordinates to CSV and run summary to TXT |
| `validation.py` | Optical-invariant and solver-tolerance/step-size convergence validation |
| `mirage.py` | Simplified source-observer ray shooting and tangent diagram |
| `main.py` | Runs the complete configured pipeline |
| `input.json` | User-editable simulation parameters |
| `tests/` | Unit and integration tests |

## Installation

Use Python 3.10 or later. From this folder, run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

If PowerShell blocks virtual-environment activation, run the commands with
`py -m pip` in an already activated environment, or use Command Prompt's
`.venv\Scripts\activate.bat`.

## Run the complete project

From the `ray_bending` folder:

```powershell
python main.py
```

The program reads `input.json` automatically. To specify a different input
file or output folder:

```powershell
python main.py --config input.json --output-dir outputs
```

Edit `input.json` to change the atmosphere, ray launch conditions, sensitivity
values, or source-observer geometry. The configuration is validated before
any simulation begins.

## Generated outputs

The program writes these files into `outputs/`:

- `01_temperature_and_index.png` — temperature, refractive-index and vertical refractive-index-gradient profiles.
- `02_ray_trajectories.png` — configured ray paths and a uniform-medium control.
- `03_surface_temperature_sensitivity.png` — ray height versus surface heating.
- `04_mirage_visualisation.png` — simplified source-observer ray and backward tangent, when a valid connecting ray exists.
- `05_invariant_validation.png` — relative drift of \(n(y)\cos\theta\).
- `06_convergence.png` — ray-height errors versus a high-accuracy reference for progressively tighter solver settings.
- `07_critical_launch_angle.png` — numerical turning/ground-impact transition compared with the analytical critical angle.
- `08_heated_uniform_cooled.png` — ray paths for positive, zero and negative refractive-index gradients.
- `ray_paths.csv` — `ray_id`, `x_m`, `y_m`, and `slope` for every configured ray.
- `summary.txt` — launch conditions, ground-hit status, minimum-height location, interpolated turning point and final state for every ray.
- `validation.txt` — invariant drift and tolerance/step-size convergence measurements against a tighter reference.
- `additional_results.txt` — theoretical critical angle, numerical transition bracket and thermal-gradient comparison outcomes.
- `mirage_status.txt` — whether the optional mirage diagram was generated and its interpretation.

A ray that does not hit the ground has `N/A` for ground-impact distance in
`summary.txt`.

## Tests

```powershell
python -m pytest -q
```

## Assumptions and limitations

- Two-dimensional geometrical optics; rays are represented by trajectories.
- Flat ground and a refractive index that depends on height only.
- Exponential temperature excess above the surface.
- Approximately constant pressure, with refractivity scaled as \(1/T\).
- Humidity, wavelength dispersion, turbulence, scattering, ground reflection
  and finite source size are neglected.
- The mirage plot is a simplified geometrical-optics illustration. It is not a
  full simulation of the extended-source image, radiative transfer, or the
  familiar photographic “water on the road” appearance.

The parameters are illustrative and are not a calibrated meteorological model.
