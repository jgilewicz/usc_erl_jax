from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp

Activation = Callable[[jax.Array], jax.Array]


def _activation(name: str) -> Activation:
    match name.lower():
        case "relu":
            return jax.nn.relu
        case "elu":
            return jax.nn.elu
        case "tanh":
            return jax.nn.tanh
        case other:
            raise ValueError(
                f"unknown activation {other!r}; expected 'relu', 'elu' or 'tanh'"
            )


class Critic(eqx.Module):
    state_net: eqx.nn.Sequential
    net: eqx.nn.Sequential

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        key: jax.Array,
        dropout: float = 0.0,
        activation: str = "relu",
        hidden_dims: tuple[int, int] = (400, 300),
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        act = _activation(activation)
        dim1, dim2 = hidden_dims

        self.state_net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(state_dim, dim1, key=k1),
                eqx.nn.LayerNorm(dim1),
                eqx.nn.Lambda(act),
            ]
        )
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim1 + action_dim, dim2, key=k2),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Lambda(act),
                eqx.nn.Dropout(dropout),
                eqx.nn.Linear(dim2, dim2, key=k3),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Lambda(act),
                eqx.nn.Dropout(dropout),
                eqx.nn.Linear(dim2, 1, key=k4),
            ]
        )

    def __call__(
        self,
        state: jax.Array,
        action: jax.Array,
        *,
        key: jax.Array | None = None,
    ) -> jax.Array:
        features = self.state_net(state)
        x = jnp.concatenate([features, action], axis=-1)
        return self.net(x, key=key)


NIG = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


class EvidentialHead(eqx.Module):
    dense: eqx.nn.Linear

    def __init__(self, in_features: int, *, key: jax.Array) -> None:
        self.dense = eqx.nn.Linear(in_features, 4, key=key)

    def __call__(self, x: jax.Array) -> NIG:
        mu, logv, logalpha, logbeta = jnp.split(self.dense(x), 4, axis=-1)
        # floors keep the epistemic variance beta / (v (alpha - 1)) finite
        eps = 1e-4
        v = jax.nn.softplus(logv) + eps
        alpha = jax.nn.softplus(logalpha) + 1.0 + eps
        beta = jax.nn.softplus(logbeta) + eps
        # extreme mu gives NaN actor gradients that destabilise the sim
        return jnp.clip(mu, -1e4, 1e4), v, alpha, beta


class EvidentialCritic(eqx.Module):
    state_net: eqx.nn.Sequential
    net: eqx.nn.Sequential
    head: EvidentialHead

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        key: jax.Array,
        hidden_dims: tuple[int, int] = (400, 300),
    ) -> None:
        k1, k2, k3, k4 = jax.random.split(key, 4)
        dim1, dim2 = hidden_dims
        self.state_net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(state_dim, dim1, key=k1),
                eqx.nn.LayerNorm(dim1),
                eqx.nn.Lambda(jax.nn.relu),
            ]
        )
        self.net = eqx.nn.Sequential(
            [
                eqx.nn.Linear(dim1 + action_dim, dim2, key=k2),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(dim2, dim2, key=k3),
                eqx.nn.LayerNorm(dim2),
                eqx.nn.Lambda(jax.nn.relu),
            ]
        )
        self.head = EvidentialHead(dim2, key=k4)

    def __call__(self, state: jax.Array, action: jax.Array) -> NIG:
        features = self.state_net(state)
        x = jnp.concatenate([features, action], axis=-1)
        return self.head(self.net(x))


class ActorHead(eqx.Module):
    out_layer: eqx.nn.Linear
    action_limit: float = eqx.field(static=True)

    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
        *,
        key: jax.Array,
        action_limit: float = 1.0,
    ) -> None:
        self.action_limit = action_limit
        out = eqx.nn.Linear(embedding_dim, action_dim, key=key)
        wk, bk = jax.random.split(key)
        weight = jax.random.uniform(
            wk, (action_dim, embedding_dim), minval=-3e-3, maxval=3e-3
        )
        bias = jax.random.uniform(bk, (action_dim,), minval=-3e-3, maxval=3e-3)
        self.out_layer = eqx.tree_at(
            lambda m: (m.weight, m.bias), out, (weight, bias)
        )

    def __call__(self, z: jax.Array) -> jax.Array:
        return jnp.tanh(self.out_layer(z)) * self.action_limit


class SharedStateEmbedding(eqx.Module):
    embedding: eqx.nn.Sequential

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 400,
        embedding_dim: int = 4,
        *,
        key: jax.Array,
    ) -> None:
        k1, k2 = jax.random.split(key)
        self.embedding = eqx.nn.Sequential(
            [
                eqx.nn.Linear(state_dim, hidden_dim, key=k1),
                eqx.nn.LayerNorm(hidden_dim),
                eqx.nn.Linear(hidden_dim, embedding_dim, key=k2),
                eqx.nn.LayerNorm(embedding_dim),
            ]
        )

    def __call__(self, state: jax.Array) -> jax.Array:
        return jnp.tanh(self.embedding(state))
