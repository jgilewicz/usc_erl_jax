from __future__ import annotations

from typing import Any

import numpy as np
from omegaconf import OmegaConf
from sbx import PPO, SAC, TD3, CrossQ
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.noise import NormalActionNoise

ALGOS: tuple[str, ...] = ("sac", "ppo", "td3", "crossq")


def build_agent(
    algo_cfg: Any,
    env: Any,
    *,
    seed: int,
    device: str = "auto",
) -> BaseAlgorithm:
    name = str(algo_cfg.name)
    if name not in ALGOS:
        raise ValueError(
            f"unknown algo {name!r}; available: {', '.join(ALGOS)}"
        )
    kwargs: dict[str, Any] = {
        "policy": "MlpPolicy",
        "env": env,
        "learning_rate": algo_cfg.learning_rate,
        "batch_size": algo_cfg.batch_size,
        "gamma": algo_cfg.gamma,
        "seed": seed,
        "device": device,
        "verbose": 1,
    }
    if name != "ppo":
        kwargs.update(
            buffer_size=algo_cfg.buffer_size,
            learning_starts=algo_cfg.learning_starts,
            train_freq=algo_cfg.train_freq,
            gradient_steps=algo_cfg.gradient_steps,
        )
    if "net_arch" in algo_cfg:
        # OmegaConf ListConfig/DictConfig fail SBX's isinstance(net_arch, list) check, so convert to plain python.
        net_arch = OmegaConf.to_container(algo_cfg.net_arch, resolve=True)
        kwargs["policy_kwargs"] = {"net_arch": net_arch}

    if name == "sac":
        return SAC(**kwargs, tau=algo_cfg.tau, ent_coef=algo_cfg.ent_coef)
    if name == "td3":
        return TD3(
            **kwargs,
            tau=algo_cfg.tau,
            policy_delay=algo_cfg.policy_delay,
            target_policy_noise=algo_cfg.target_policy_noise,
            target_noise_clip=algo_cfg.target_noise_clip,
            action_noise=_action_noise(env, algo_cfg.action_noise_std),
        )
    if name == "crossq":
        return CrossQ(
            **kwargs,
            policy_delay=algo_cfg.policy_delay,
            ent_coef=algo_cfg.ent_coef,
        )
    return PPO(
        **kwargs,
        n_steps=algo_cfg.n_steps,
        n_epochs=algo_cfg.n_epochs,
        gae_lambda=algo_cfg.gae_lambda,
        clip_range=algo_cfg.clip_range,
        ent_coef=algo_cfg.ent_coef,
    )


def _action_noise(env: Any, std: float) -> NormalActionNoise | None:
    if std <= 0:
        return None
    n_actions = int(np.prod(env.action_space.shape))
    return NormalActionNoise(
        mean=np.zeros(n_actions, dtype=np.float32),
        sigma=std * np.ones(n_actions, dtype=np.float32),
    )
