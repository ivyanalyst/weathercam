from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    dawn_dir: Path
    outputs_dir: Path


@dataclass(frozen=True)
class Config:
    seed: int
    memory_limit_mb: int
    paths: Paths


def load_config(path=DEFAULT_CONFIG):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        paths_raw = raw["paths"]
        return Config(
            seed=int(raw["seed"]),
            memory_limit_mb=int(raw["memory"]["limit_mb"]),
            paths=Paths(
                data_dir=PROJECT_ROOT / paths_raw["data_dir"],
                dawn_dir=PROJECT_ROOT / paths_raw["dawn_dir"],
                outputs_dir=PROJECT_ROOT / paths_raw["outputs_dir"],
            ),
        )
    except (KeyError, TypeError) as err:
        raise ValueError(f"Config {path} is missing or malformed: {err!r}") from err
