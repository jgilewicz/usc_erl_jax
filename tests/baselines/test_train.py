from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.baselines import train
from tests.baselines.conftest import load_cfg


class _FakeRun:
    def __init__(self) -> None:
        self.finished = False

    def finish(self) -> None:
        self.finished = True


class _FakeWandb:
    def __init__(self) -> None:
        self.run: _FakeRun | None = None
        self.logged: list[dict[str, object]] = []
        self.init_kwargs: dict[str, object] = {}

    def init(self, **kwargs: object) -> _FakeRun:
        self.init_kwargs = kwargs
        self.run = _FakeRun()
        return self.run

    def log(self, payload: dict[str, object], step: int = 0) -> None:
        self.logged.append(payload)


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _FakeWandb:
    fake = _FakeWandb()
    monkeypatch.setattr("src.baselines.train.wandb", fake)
    monkeypatch.setattr("src.baselines.wandb_logging.wandb", fake)
    return fake


def test_run_training_logs_canonical_metrics(
    fake_wandb: _FakeWandb, tmp_path: Path
) -> None:
    result_file = tmp_path / "result.json"
    cfg = load_cfg(
        "algorithm=sac",
        "env.id=Swimmer-v5",
        "eval_env.id=Swimmer-v5",
        "total_steps=300",
        "algorithm.learning_starts=100",
        "eval.interval=200",
        "eval.episodes=1",
        f"result_file={result_file}",
    )

    eval_reward = train.run_training(cfg)

    assert fake_wandb.run is not None and fake_wandb.run.finished
    assert fake_wandb.init_kwargs["entity"] == "evo_rl"

    seen: set[str] = set()
    for payload in fake_wandb.logged:
        seen.update(payload)
    assert {"total_steps", "n_updates", "actor_loss", "critic_loss"} <= seen
    assert "eval_reward" in seen

    assert json.loads(result_file.read_text()) == {"eval_reward": eval_reward}


def test_run_training_disabled_wandb_still_runs(fake_wandb: _FakeWandb) -> None:
    cfg = load_cfg(
        "algorithm=td3",
        "total_steps=250",
        "eval.interval=200",
        "eval.episodes=1",
        "wandb.enabled=false",
    )
    train.run_training(cfg)
    assert fake_wandb.init_kwargs["mode"] == "disabled"
