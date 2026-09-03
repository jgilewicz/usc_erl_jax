from __future__ import annotations

from typing import Any

import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import KVWriter

# canonical wandb metric -> SB3 logger keys to try, first present one wins.
# off-policy algos use train/actor_loss + train/critic_loss; PPO has neither,
# so its policy- and value-loss stand in.
_CANONICAL: dict[str, tuple[str, ...]] = {
    "total_steps": ("time/total_timesteps",),
    "eval_reward": ("eval/mean_reward",),
    "n_updates": ("train/n_updates",),
    "actor_loss": ("train/actor_loss", "train/pg_loss"),
    "critic_loss": ("train/critic_loss", "train/value_loss"),
}


class WandbOutputFormat(KVWriter):
    def write(
        self,
        key_values: dict[str, Any],
        key_excluded: dict[str, tuple[str, ...]],
        step: int = 0,
    ) -> None:
        payload: dict[str, Any] = {}
        for canonical, sources in _CANONICAL.items():
            for src in sources:
                if src in key_values:
                    payload[canonical] = key_values[src]
                    break
        if payload:
            wandb.log(payload, step=step)

    def close(self) -> None:
        pass


class WandbLoggingCallback(BaseCallback):
    def _on_training_start(self) -> None:
        if wandb.run is None:
            raise RuntimeError(
                "no active wandb run; call wandb.init() before model.learn()"
            )
        formats = self.logger.output_formats
        if not any(isinstance(f, WandbOutputFormat) for f in formats):
            formats.append(WandbOutputFormat())

    def _on_step(self) -> bool:
        return True
