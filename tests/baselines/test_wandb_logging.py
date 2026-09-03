from __future__ import annotations

import pytest

from src.baselines.wandb_logging import WandbLoggingCallback, WandbOutputFormat


class _RecordingWandb:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, object], int]] = []
        self.run = object()

    def log(self, payload: dict[str, object], step: int = 0) -> None:
        self.calls.append((payload, step))


@pytest.fixture
def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> _RecordingWandb:
    fake = _RecordingWandb()
    monkeypatch.setattr("src.baselines.wandb_logging.wandb", fake)
    return fake


def test_write_maps_off_policy_keys(fake_wandb: _RecordingWandb) -> None:
    WandbOutputFormat().write(
        {
            "time/total_timesteps": 2048,
            "train/n_updates": 17,
            "train/actor_loss": -1.5,
            "train/critic_loss": 0.8,
            "train/ent_coef": 0.1,
        },
        {},
        step=2048,
    )
    payload, step = fake_wandb.calls[0]
    assert step == 2048
    assert payload == {
        "total_steps": 2048,
        "n_updates": 17,
        "actor_loss": -1.5,
        "critic_loss": 0.8,
    }


def test_write_maps_ppo_loss_aliases(fake_wandb: _RecordingWandb) -> None:
    WandbOutputFormat().write(
        {"train/pg_loss": -0.2, "train/value_loss": 4.1},
        {},
    )
    payload, _ = fake_wandb.calls[0]
    assert payload == {"actor_loss": -0.2, "critic_loss": 4.1}


def test_write_prefers_explicit_actor_loss_over_ppo_alias(
    fake_wandb: _RecordingWandb,
) -> None:
    WandbOutputFormat().write(
        {"train/actor_loss": 1.0, "train/pg_loss": 2.0},
        {},
    )
    payload, _ = fake_wandb.calls[0]
    assert payload["actor_loss"] == 1.0


def test_write_forwards_eval_reward(fake_wandb: _RecordingWandb) -> None:
    WandbOutputFormat().write({"eval/mean_reward": 123.4}, {})
    payload, _ = fake_wandb.calls[0]
    assert payload == {"eval_reward": 123.4}


def test_write_skips_when_nothing_matches(fake_wandb: _RecordingWandb) -> None:
    WandbOutputFormat().write({"rollout/ep_rew_mean": 5.0}, {})
    assert fake_wandb.calls == []


class _FakeLogger:
    def __init__(self) -> None:
        self.output_formats: list[object] = []


class _FakeModel:
    def __init__(self) -> None:
        self.logger = _FakeLogger()


def _bind(callback: WandbLoggingCallback, model: _FakeModel) -> None:
    callback.model = model  # type: ignore[assignment]


def test_callback_appends_format_once(fake_wandb: _RecordingWandb) -> None:
    model = _FakeModel()
    cb = WandbLoggingCallback()
    _bind(cb, model)
    cb._on_training_start()
    cb._on_training_start()
    formats = model.logger.output_formats
    assert sum(isinstance(f, WandbOutputFormat) for f in formats) == 1


def test_callback_requires_active_run(fake_wandb: _RecordingWandb) -> None:
    fake_wandb.run = None
    cb = WandbLoggingCallback()
    _bind(cb, _FakeModel())
    with pytest.raises(RuntimeError, match="wandb.init"):
        cb._on_training_start()
