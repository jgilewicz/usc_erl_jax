from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from sbx import PPO, SAC, TD3, CrossQ
from stable_baselines3.common.noise import NormalActionNoise

from src.baselines.agents import build_agent
from src.baselines.envs import build_vec_env
from tests.baselines.conftest import load_cfg


def test_build_agent_unknown_algo_raises() -> None:
    with pytest.raises(ValueError, match="unknown algo 'dqn'"):
        build_agent(OmegaConf.create({"name": "dqn"}), object(), seed=0)


@pytest.mark.parametrize(
    ("algo", "cls"),
    [("sac", SAC), ("ppo", PPO), ("td3", TD3), ("crossq", CrossQ)],
)
def test_build_agent_builds_requested_algo(algo: str, cls: type) -> None:
    cfg = load_cfg(f"algorithm={algo}")
    env = build_vec_env("Swimmer-v5", 1, seed=0)
    try:
        model = build_agent(cfg.algorithm, env, seed=0)
        assert isinstance(model, cls)
        assert model.seed == 0
    finally:
        env.close()


def test_build_agent_applies_config_hyperparams() -> None:
    cfg = load_cfg("algorithm=sac", "algorithm.batch_size=64")
    env = build_vec_env("Swimmer-v5", 1, seed=0)
    try:
        model = build_agent(cfg.algorithm, env, seed=0)
        assert model.batch_size == 64
        assert model.replay_buffer.buffer_size == cfg.algorithm.buffer_size
    finally:
        env.close()


def test_td3_action_noise_toggles_with_config() -> None:
    env = build_vec_env("Swimmer-v5", 1, seed=0)
    try:
        noisy = build_agent(load_cfg("algorithm=td3").algorithm, env, seed=0)
        assert isinstance(noisy.action_noise, NormalActionNoise)

        quiet = build_agent(
            load_cfg("algorithm=td3", "algorithm.action_noise_std=0").algorithm,
            env,
            seed=0,
        )
        assert quiet.action_noise is None
    finally:
        env.close()
