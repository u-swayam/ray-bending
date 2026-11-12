"""Load and validate the JSON input file for the ray-bending project.

Public interface:
    load_config(path) -> Config
    ConfigError              (subclass of ValueError)
    Config, RayConfig, SimulationConfig, SensitivityConfig, MirageConfig

The atmosphere section is turned into the existing ``AtmosphereParams``
object; AtmosphereParams performs its own physical validation and no
atmospheric equations are implemented here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from atmosphere import AtmosphereParams

__all__ = [
    "ConfigError",
    "RayConfig",
    "SimulationConfig",
    "SensitivityConfig",
    "MirageConfig",
    "Config",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when the input file is missing, malformed or has invalid values."""


@dataclass(frozen=True)
class RayConfig:
    ray_id: str
    y0_m: float
    theta0_deg: float


@dataclass(frozen=True)
class SimulationConfig:
    x_max_m: float
    max_step_m: float
    rays: tuple[RayConfig, ...]


@dataclass(frozen=True)
class SensitivityConfig:
    distance_m: float
    surface_temp_excess_values_K: tuple[float, ...]


@dataclass(frozen=True)
class MirageConfig:
    source_y_m: float
    observer_y_m: float
    observer_distance_m: float


@dataclass(frozen=True)
class Config:
    """Validated configuration.

    ``atmosphere`` is a ready-to-use ``AtmosphereParams`` instance.
    """

    atmosphere: AtmosphereParams
    simulation: SimulationConfig
    sensitivity: SensitivityConfig
    mirage: MirageConfig
    source_path: Path | None = None


# ----------------------------------------------------------------- helpers
_ATMOSPHERE_KEYS = (
    "ambient_temp_K",
    "surface_temp_excess_K",
    "scale_height_m",
    "n_ref",
    "reference_temp_K",
)


def _section(root: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    if name not in root:
        raise ConfigError(f"Missing required section '{name}'")
    value = root[name]
    if not isinstance(value, dict):
        raise ConfigError(
            f"Section '{name}' must be a JSON object, got {type(value).__name__}"
        )
    return value


def _check_keys(obj: Mapping[str, Any], where: str, required, optional=()) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ConfigError(f"{where}: missing required key(s): {', '.join(missing)}")
    allowed = set(required) | set(optional)
    unknown = sorted(set(obj) - allowed)
    if unknown:
        raise ConfigError(
            f"{where}: unknown key(s): {', '.join(unknown)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )


def _number(value: Any, where: str) -> float:
    """Return value as a finite float; reject bools, strings, NaN, inf."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{where}: must be a number, got {type(value).__name__} ({value!r})"
        )
    value = float(value)
    if not math.isfinite(value):
        raise ConfigError(f"{where}: must be finite, got {value!r}")
    return value


def _positive(value: Any, where: str) -> float:
    v = _number(value, where)
    if v <= 0:
        raise ConfigError(f"{where}: must be > 0, got {v!r}")
    return v


def _non_negative(value: Any, where: str) -> float:
    v = _number(value, where)
    if v < 0:
        raise ConfigError(f"{where}: must be >= 0, got {v!r}")
    return v


# --------------------------------------------------------------- sections
def _parse_atmosphere(section: Mapping[str, Any]) -> AtmosphereParams:
    # Every key is optional: omitted ones take AtmosphereParams' defaults.
    _check_keys(section, "atmosphere", required=(), optional=_ATMOSPHERE_KEYS)
    kwargs = {
        key: _number(value, f"atmosphere.{key}") for key, value in section.items()
    }
    try:
        return AtmosphereParams(**kwargs)
    except ValueError as exc:
        raise ConfigError(f"atmosphere: {exc}") from exc


def _parse_ray(entry: Any, index: int) -> RayConfig:
    where = f"simulation.rays[{index}]"
    if not isinstance(entry, dict):
        raise ConfigError(f"{where}: must be a JSON object, got {type(entry).__name__}")
    _check_keys(entry, where, required=("ray_id", "y0_m", "theta0_deg"))

    ray_id = entry["ray_id"]
    if not isinstance(ray_id, str) or not ray_id.strip():
        raise ConfigError(f"{where}.ray_id: must be a non-empty string, got {ray_id!r}")

    y0 = _positive(entry["y0_m"], f"{where}.y0_m")
    theta = _number(entry["theta0_deg"], f"{where}.theta0_deg")
    if not -90.0 < theta < 90.0:
        raise ConfigError(
            f"{where}.theta0_deg: must be in (-90, 90) degrees, got {theta!r}"
        )
    return RayConfig(ray_id=ray_id.strip(), y0_m=y0, theta0_deg=theta)


def _parse_simulation(section: Mapping[str, Any]) -> SimulationConfig:
    _check_keys(section, "simulation", required=("x_max_m", "max_step_m", "rays"))
    x_max = _positive(section["x_max_m"], "simulation.x_max_m")
    max_step = _positive(section["max_step_m"], "simulation.max_step_m")

    raw_rays = section["rays"]
    if not isinstance(raw_rays, list) or not raw_rays:
        raise ConfigError("simulation.rays: must be a non-empty list of ray objects")
    rays = tuple(_parse_ray(r, i) for i, r in enumerate(raw_rays))

    seen: set[str] = set()
    for ray in rays:
        if ray.ray_id in seen:
            raise ConfigError(f"simulation.rays: duplicate ray_id {ray.ray_id!r}")
        seen.add(ray.ray_id)
    return SimulationConfig(x_max_m=x_max, max_step_m=max_step, rays=rays)


def _parse_sensitivity(
    section: Mapping[str, Any], atmosphere: AtmosphereParams
) -> SensitivityConfig:
    _check_keys(
        section,
        "sensitivity",
        required=("distance_m", "surface_temp_excess_values_K"),
    )
    distance = _positive(section["distance_m"], "sensitivity.distance_m")

    raw = section["surface_temp_excess_values_K"]
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            "sensitivity.surface_temp_excess_values_K: must be a non-empty list"
        )
    values = []
    for i, item in enumerate(raw):
        where = f"sensitivity.surface_temp_excess_values_K[{i}]"
        v = _number(item, where)
        try:  # reuse the atmosphere model's own physical checks
            replace(atmosphere, surface_temp_excess_K=v)
        except ValueError as exc:
            raise ConfigError(f"{where}: {exc}") from exc
        values.append(v)
    return SensitivityConfig(
        distance_m=distance, surface_temp_excess_values_K=tuple(values)
    )


def _parse_mirage(section: Mapping[str, Any]) -> MirageConfig:
    _check_keys(
        section,
        "mirage",
        required=("source_y_m", "observer_y_m", "observer_distance_m"),
    )
    return MirageConfig(
        source_y_m=_positive(section["source_y_m"], "mirage.source_y_m"),
        observer_y_m=_positive(section["observer_y_m"], "mirage.observer_y_m"),
        observer_distance_m=_positive(
            section["observer_distance_m"], "mirage.observer_distance_m"
        ),
    )


# -------------------------------------------------------------- public API
def load_config(path: str | Path) -> Config:
    """Read, validate and return the configuration in ``path``.

    Raises:
        ConfigError: file missing/unreadable, invalid JSON, wrong structure
            or invalid parameter values. The message names the offending key.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"Input file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            root = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{p}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    except (OSError, UnicodeDecodeError) as exc:
        raise ConfigError(f"Cannot read {p}: {exc}") from exc

    if not isinstance(root, dict):
        raise ConfigError(
            f"{p}: top level must be a JSON object, got {type(root).__name__}"
        )
    _check_keys(
        root, str(p), required=("atmosphere", "simulation", "sensitivity", "mirage")
    )

    atmosphere = _parse_atmosphere(_section(root, "atmosphere"))
    return Config(
        atmosphere=atmosphere,
        simulation=_parse_simulation(_section(root, "simulation")),
        sensitivity=_parse_sensitivity(_section(root, "sensitivity"), atmosphere),
        mirage=_parse_mirage(_section(root, "mirage")),
        source_path=p,
    )