import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from src.modules.deep_modules import (
    Actor,
    AdaptiveBeta,
    Critic,
    EvidentialCritic,
    EvidentialModule,
    _activation,
)

STATE_DIM = 11
ACTION_DIM = 3


@pytest.fixture
def key() -> jax.Array:
    return jax.random.key(0)


def test_actor_output_within_action_limit(key: jax.Array) -> None:
    actor = Actor(STATE_DIM, ACTION_DIM, 256, key=key, action_limit=2.0)
    action = actor(jnp.ones(STATE_DIM))
    assert action.shape == (ACTION_DIM,)
    assert jnp.all(jnp.abs(action) <= 2.0)


def test_actor_without_layernorm_runs(key: jax.Array) -> None:
    actor = Actor(STATE_DIM, ACTION_DIM, 64, key=key, use_ln=False)
    assert actor(jnp.ones(STATE_DIM)).shape == (ACTION_DIM,)


def test_actor_batches_under_vmap(key: jax.Array) -> None:
    actor = Actor(STATE_DIM, ACTION_DIM, 32, key=key)
    states = jnp.ones((8, STATE_DIM))
    actions = jax.vmap(actor)(states)
    assert actions.shape == (8, ACTION_DIM)


def test_critic_returns_scalar_value(key: jax.Array) -> None:
    critic = Critic(STATE_DIM, ACTION_DIM, key=key)
    value = critic(jnp.ones(STATE_DIM), jnp.ones(ACTION_DIM))
    assert value.shape == (1,)


def test_critic_dropout_needs_key_and_changes_output(key: jax.Array) -> None:
    critic = Critic(STATE_DIM, ACTION_DIM, key=key, dropout=0.5)
    state, action = jnp.ones(STATE_DIM), jnp.ones(ACTION_DIM)
    d1, d2 = jax.random.split(key)
    out_a = critic(state, action, key=d1)
    out_b = critic(state, action, key=d2)
    assert not jnp.allclose(out_a, out_b)

    inference = eqx.nn.inference_mode(critic)
    assert jnp.allclose(inference(state, action), inference(state, action))


def test_evidential_module_constraints(key: jax.Array) -> None:
    module = EvidentialModule(16, key=key)
    mu, v, alpha, beta = module(jax.random.normal(key, (16,)) * 100)
    assert mu.shape == v.shape == alpha.shape == beta.shape == (1,)
    assert jnp.all(v > 0)
    assert jnp.all(alpha > 1.0)
    assert jnp.all(beta > 0)
    assert jnp.all(jnp.abs(mu) <= 1e4)


def test_evidential_critic_returns_four_params(key: jax.Array) -> None:
    critic = EvidentialCritic(STATE_DIM, ACTION_DIM, key=key)
    outputs = critic(jnp.ones(STATE_DIM), jnp.ones(ACTION_DIM))
    assert len(outputs) == 4
    assert all(o.shape == (1,) for o in outputs)


def test_adaptive_beta_tracks_log_parameter() -> None:
    beta = AdaptiveBeta(init_value=2.0)
    assert jnp.allclose(beta.beta, 2.0)
    updated = eqx.tree_at(lambda b: b.log_beta, beta, jnp.log(jnp.asarray(5.0)))
    assert jnp.allclose(updated.beta, 5.0)


def test_activation_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown activation"):
        _activation("gelu")
