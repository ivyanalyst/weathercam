from pathlib import Path

import pytest

from weathercam.core.config import PROJECT_ROOT, load_config


DEFAULT_TEXT = (
    "seed: 42\n"
    "memory:\n"
    "  limit_mb: 3000\n"
    "paths:\n"
    "  data_dir: data\n"
    "  dawn_dir: data/DAWN\n"
    "  outputs_dir: outputs\n"
)


def write_configs(tmp_path, local_text=None):
    default = tmp_path / "default.yaml"
    default.write_text(DEFAULT_TEXT, encoding="utf-8")
    if local_text is not None:
        (tmp_path / "local.yaml").write_text(local_text, encoding="utf-8")
    return default


def test_loads_default_values(tmp_path):
    cfg = load_config(write_configs(tmp_path))
    assert cfg.seed == 42
    assert cfg.memory_limit_mb == 3000
    assert cfg.paths.data_dir == PROJECT_ROOT / "data"
    assert cfg.paths.dawn_dir == PROJECT_ROOT / "data" / "DAWN"
    assert cfg.paths.outputs_dir == PROJECT_ROOT / "outputs"


def test_missing_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_missing_key_raises(tmp_path):
    bad = tmp_path / "default.yaml"
    bad.write_text("memory:\n  limit_mb: 3000\npaths:\n  data_dir: data\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_empty_config_raises(tmp_path):
    empty = tmp_path / "default.yaml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(empty)


def test_absolute_path_is_kept(tmp_path):
    text = DEFAULT_TEXT.replace("data_dir: data\n", "data_dir: /mnt/elsewhere/data\n")
    default = tmp_path / "default.yaml"
    default.write_text(text, encoding="utf-8")
    assert load_config(default).paths.data_dir == Path("/mnt/elsewhere/data")


def test_optional_paths_default_to_none(tmp_path):
    paths = load_config(write_configs(tmp_path)).paths
    assert paths.veri_zip is None
    assert paths.ccpd_dir is None
    assert paths.vmmrdb_dir is None


def test_local_sets_optional_paths(tmp_path):
    local = "paths:\n  veri_zip: /mnt/d/VeRi.zip\n  ccpd_dir: /mnt/d/CCPD2019\n  vmmrdb_dir: /mnt/d/VMMRdb\n"
    paths = load_config(write_configs(tmp_path, local)).paths
    assert paths.veri_zip == Path("/mnt/d/VeRi.zip")
    assert paths.ccpd_dir == Path("/mnt/d/CCPD2019")
    assert paths.vmmrdb_dir == Path("/mnt/d/VMMRdb")


def test_local_merges_per_key(tmp_path):
    cfg = load_config(write_configs(tmp_path, "paths:\n  data_dir: /mnt/d/data\n"))
    assert cfg.paths.data_dir == Path("/mnt/d/data")
    assert cfg.paths.dawn_dir == PROJECT_ROOT / "data" / "DAWN"
    assert cfg.paths.outputs_dir == PROJECT_ROOT / "outputs"
    assert cfg.seed == 42
    assert cfg.memory_limit_mb == 3000


def test_local_overrides_scalars(tmp_path):
    cfg = load_config(write_configs(tmp_path, "seed: 7\nmemory:\n  limit_mb: 2000\n"))
    assert cfg.seed == 7
    assert cfg.memory_limit_mb == 2000


def test_local_relative_path_joins_project_root(tmp_path):
    cfg = load_config(write_configs(tmp_path, "paths:\n  veri_zip: data/VeRi.zip\n"))
    assert cfg.paths.veri_zip == PROJECT_ROOT / "data" / "VeRi.zip"


def test_empty_local_file_is_ignored(tmp_path):
    cfg = load_config(write_configs(tmp_path, ""))
    assert cfg.seed == 42
    assert cfg.paths.veri_zip is None


def test_local_not_a_mapping_raises(tmp_path):
    with pytest.raises(ValueError):
        load_config(write_configs(tmp_path, "- a\n- b\n"))
