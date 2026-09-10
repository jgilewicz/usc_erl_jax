from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList, EvalCallback

from algos.erl import ERLConfig
from algos.erl import evaluate_actor as evaluate_erl_actor
from algos.erl import train as train_erl
from algos.semarl import SEMARLConfig
from algos.semarl import evaluate_actor as evaluate_semarl_actor
from algos.semarl import train as train_semarl
from baselines.agents import build_agent
from baselines.envs import build_vec_env
from baselines.wandb_logging import WandbLoggingCallback


def _erl_kwargs(cfg: DictConfig) -> dict[str, Any]:
    algo_cfg = cfg.algorithm
    horizon = int(algo_cfg.horizon)
    pop_size = int(algo_cfg.pop_size)
    generations = max(cfg.total_steps // (horizon * (pop_size + 1)), 1)
    return dict(
        env_name=cfg.env.id,
        seed=cfg.seed,
        generations=generations,
        horizon=horizon,
        async_env=algo_cfg.async_env,
        pop_size=pop_size,
        embedding_dim=algo_cfg.embedding_dim,
        sigma_init=algo_cfg.sigma_init,
        damp=algo_cfg.damp,
        damp_limit=algo_cfg.damp_limit,
        elitism=algo_cfg.elitism,
        critic_hidden_dims=tuple(algo_cfg.critic_hidden_dims),
        gamma=algo_cfg.gamma,
        tau=algo_cfg.tau,
        policy_noise=algo_cfg.policy_noise,
        noise_clip=algo_cfg.noise_clip,
        policy_freq=algo_cfg.policy_freq,
        exploration_noise=algo_cfg.exploration_noise,
        actor_lr=algo_cfg.actor_lr,
        critic_lr=algo_cfg.critic_lr,
        batch_size=algo_cfg.batch_size,
        buffer_capacity=algo_cfg.buffer_capacity,
        warmup_steps=algo_cfg.warmup_steps,
        train_ratio=algo_cfg.train_ratio,
        rl_to_ea_sync_period=algo_cfg.rl_to_ea_sync_period,
        ea_tau=algo_cfg.ea_tau,
        theta=algo_cfg.theta,
        h_steps=algo_cfg.h_steps,
    )


def _run_erl(cfg: DictConfig) -> float:
    td3_state = train_erl(
        ERLConfig(**_erl_kwargs(cfg)), on_generation=wandb.log
    )
    return evaluate_erl_actor(
        cfg.eval_env.id, td3_state, episodes=cfg.eval.episodes
    )


def _run_semarl(cfg: DictConfig) -> float:
    algo_cfg = cfg.algorithm
    semarl_cfg = SEMARLConfig(
        **_erl_kwargs(cfg),
        h_min=int(algo_cfg.h_min),
        h_max=int(algo_cfg.h_max),
        h_beta=algo_cfg.h_beta,
        td_ema_decay=algo_cfg.td_ema_decay,
    )
    td3_state = train_semarl(semarl_cfg, on_generation=wandb.log)
    return evaluate_semarl_actor(
        cfg.eval_env.id, td3_state, episodes=cfg.eval.episodes
    )


def _run_sb3(cfg: DictConfig) -> float:
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
    return float(eval_cb.last_mean_reward)


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
    try:
        dispatch = {"erl": _run_erl, "semarl": _run_semarl}
        runner = dispatch.get(cfg.algorithm.name, _run_sb3)
        eval_reward = runner(cfg)
    finally:
        run.finish()

    if cfg.result_file is not None:
        Path(cfg.result_file).write_text(
            json.dumps({"eval_reward": eval_reward})
        )
    return eval_reward


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
