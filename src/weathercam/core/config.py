from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
LOCAL_CONFIG_NAME = "local.yaml"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    dawn_dir: Path
    outputs_dir: Path
    veri_zip: Path | None = None
    ccpd_dir: Path | None = None
    vmmrdb_dir: Path | None = None


@dataclass(frozen=True)
class Config:
    seed: int
    memory_limit_mb: int
    paths: Paths


def _read_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _merge(base, override):
    if not isinstance(base, dict):
        return base
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _optional_path(paths_raw, key: str) -> Path | None:
    value = paths_raw.get(key)
    if value is None:
        return None
    return PROJECT_ROOT / value


def load_config(path=DEFAULT_CONFIG) -> Config:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = _read_yaml(path)
    local_path = path.parent / LOCAL_CONFIG_NAME
    if local_path.is_file() and local_path != path:
        local_raw = _read_yaml(local_path)
        if local_raw is not None:
            if not isinstance(local_raw, dict):
                raise ValueError(f"Local config {local_path} must be a mapping")
            raw = _merge(raw, local_raw)
    try:
        paths_raw = raw["paths"]
        return Config(
            seed=int(raw["seed"]),
            memory_limit_mb=int(raw["memory"]["limit_mb"]),
            paths=Paths(
                data_dir=PROJECT_ROOT / paths_raw["data_dir"],
                dawn_dir=PROJECT_ROOT / paths_raw["dawn_dir"],
                outputs_dir=PROJECT_ROOT / paths_raw["outputs_dir"],
                veri_zip=_optional_path(paths_raw, "veri_zip"),
                ccpd_dir=_optional_path(paths_raw, "ccpd_dir"),
                vmmrdb_dir=_optional_path(paths_raw, "vmmrdb_dir"),
            ),
        )
    except (KeyError, TypeError) as err:
        raise ValueError(f"Config {path} is missing or malformed: {err!r}") from err
