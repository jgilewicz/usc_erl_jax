from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from src.baselines.agents import build_agent
from src.baselines.envs import build_vec_env
from src.baselines.wandb_logging import WandbLoggingCallback


def run_training(cfg: DictConfig) -> float:
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name,
        tags=list(cfg.wandb.tags),
        mode="online" if cfg.wandb.enabled else "disabled",
        config=cast(
            "dict[str, Any]", OmegaConf.to_container(cfg, resolve=True)
        ),
    )
    train_env = build_vec_env(cfg.env.id, cfg.n_envs, cfg.seed)
    eval_env = build_vec_env(cfg.eval_env.id, 1, cfg.seed + 10_000)
    model = build_agent(
        cfg.algorithm, train_env, seed=cfg.seed, device=cfg.device
    )
    eval_cb = EvalCallback(
        eval_env,
        n_eval_episodes=cfg.eval.episodes,
        eval_freq=max(cfg.eval.interval // cfg.n_envs, 1),
        deterministic=True,
        verbose=1,
    )
    try:
        model.learn(
            total_timesteps=cfg.total_steps,
            callback=CallbackList([WandbLoggingCallback(), eval_cb]),
        )
    finally:
        train_env.close()
        eval_env.close()
        run.finish()

    eval_reward = float(eval_cb.last_mean_reward)
    if cfg.result_file is not None:
        Path(cfg.result_file).write_text(
            json.dumps({"eval_reward": eval_reward})
        )
    return eval_reward


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
