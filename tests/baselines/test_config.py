from __future__ import annotations

import pytest

from tests.baselines.conftest import load_cfg


@pytest.mark.parametrize("algo", ["sac", "ppo", "td3", "crossq"])
def test_every_algorithm_config_composes(algo: str) -> None:
    cfg = load_cfg(f"algorithm={algo}")
    assert cfg.algorithm.name == algo
    assert cfg.total_steps == 1_000_000
    assert cfg.wandb.entity == "evo_rl"
    assert cfg.wandb.project == "triage_erl"


def test_default_algorithm_is_sac() -> None:
    assert load_cfg().algorithm.name == "sac"


def test_cli_overrides_apply() -> None:
    cfg = load_cfg("algorithm=td3", "seed=7", "env.id=Hopper-v5")
    assert cfg.seed == 7
    assert cfg.env.id == "Hopper-v5"
    assert cfg.algorithm.buffer_size == 200_000
