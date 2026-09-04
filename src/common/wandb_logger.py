import os
from typing import Any

import wandb
from omegaconf import DictConfig
from dotenv import load_dotenv


class WandbLogger:
    def __init__(
        self,
        project: str,
        name: str | None = None,
        entity: str | None = None,
        tags: list[str] | None = None,
        config: dict | None = None,
    ):
        self._project = project
        self._name = name
        self._entity = entity
        self._tags = tags or []
        self._config = config or {}

        self._enabled = self._should_enable()
        self._initialized = False
        self.run = None

    def init(self, config: dict | None = None) -> None:
        if not self._enabled:
            print("WandB disabled (no API key).")
            return

        run_config = config if config is not None else self._config

        self.run = wandb.init(
            project=self._project,
            name=self._name,
            entity=self._entity,
            tags=self._tags,
            config=run_config,
        )

        self._initialized = True

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._initialized or self.run is None:
            return
        wandb.log(metrics, step=step)

    def finish(self) -> None:
        if not self._initialized or self.run is None:
            return
        wandb.finish()
        self.run = None
        self._initialized = False

    @staticmethod
    def _should_enable() -> bool:
        load_dotenv()
        return os.getenv("WANDB_API_KEY") is not None

    @classmethod
    def from_cfg(cls, cfg: DictConfig) -> "WandbLogger":
        return cls(
            project=cfg.project,
            name=cfg.name,
            entity=cfg.entity,
            tags=list(cfg.tags) if cfg.tags else None,
        )
