from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

_CONF_DIR = str(Path(__file__).resolve().parents[2] / "src" / "conf")


def load_cfg(*overrides: str) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=_CONF_DIR):
        cfg = compose(config_name="config", overrides=list(overrides))
    assert isinstance(cfg, DictConfig)
    return cfg
