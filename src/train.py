from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # spawned env workers re-import this module as __mp_main__: importing
    # the training stack (JAX, torch, wandb, SBX) here keeps it out of every
    # worker (~0.35 GB RSS each, OOM with probe workers)
    from runner import run_training

    run_training(cfg)


if __name__ == "__main__":
    main()
