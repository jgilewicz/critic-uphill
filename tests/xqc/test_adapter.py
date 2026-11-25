from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.base import _q1_population_states
from tests.xqc.conftest import ACTION_DIM, OBS_DIM, make_trained_agent
from xqc.agent import XQCAgent


def _states_actions(n: int = 64) -> tuple[jax.Array, jax.Array]:
    k_s, k_a = jax.random.split(jax.random.key(7))
    return (
        jax.random.normal(k_s, (n, OBS_DIM)),
        jax.random.uniform(k_a, (n, ACTION_DIM), minval=-1, maxval=1),
    )


def _exact(a: jax.Array, b: jax.Array) -> None:
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_critics_match_training_code(trained_agent: XQCAgent) -> None:
    states, actions = _states_actions()
    q_members = trained_agent.nets.critic.q_eval(states, actions)
    q1 = jax.vmap(trained_agent.critic1)(states, actions)[:, 0]
    q2 = jax.vmap(trained_agent.critic2)(states, actions)[:, 0]
    _exact(q1, jnp.min(q_members, axis=0))
    _exact(q2, q_members[0])


def test_mean_action_matches_training_code(trained_agent: XQCAgent) -> None:
    states, _ = _states_actions()
    mean, _, _ = trained_agent.nets.actor.forward(states, False)
    z = jax.vmap(trained_agent.embedding)(states)
    action = jax.vmap(trained_agent.actor_head)(z)
    _exact(action, jnp.tanh(mean))


def test_action_limit_scales_actor_and_critic() -> None:
    agent = make_trained_agent(action_limit=0.4, updates=1)
    states, policy_actions = _states_actions()
    mean, _, _ = agent.nets.actor.forward(states, False)
    z = jax.vmap(agent.embedding)(states)
    env_actions = jax.vmap(agent.actor_head)(z)
    _exact(env_actions, jnp.tanh(mean) * 0.4)
    env_scale = policy_actions * 0.4
    q = jax.vmap(agent.critic1)(states, env_scale)[:, 0]
    internal = jnp.min(agent.nets.critic.q_eval(states, env_scale / 0.4), 0)
    _exact(q, internal)


# jitted population path fuses differently, so ulp-level differences remain
def test_population_q_matches_single_head(trained_agent: XQCAgent) -> None:
    states, _ = _states_actions(8)
    head = trained_agent.actor_head
    heads = jax.tree.map(lambda x: jnp.stack([x, x]), head)
    pop_q = _q1_population_states(
        trained_agent.critic1, trained_agent.embedding, heads, states[:2]
    )
    z = jax.vmap(trained_agent.embedding)(states[:2])
    single = jax.vmap(trained_agent.critic1)(states[:2], jax.vmap(head)(z))[
        :, 0
    ]
    np.testing.assert_allclose(pop_q, single, rtol=1e-4, atol=1e-6)


def test_head_flat_roundtrip(trained_agent: XQCAgent) -> None:
    flat = trained_agent.actor_head_flat()
    head = trained_agent.unravel_head(flat)
    _exact(head.mean.kernel, trained_agent.actor_head.mean.kernel)
    assert flat.shape[0] == (256 + 1) * ACTION_DIM


def test_shadow_critic_is_absent(trained_agent: XQCAgent) -> None:
    assert trained_agent.shadow_critic is None
    assert "unavailable" in trained_agent.signal_mapping["shadow"]


@pytest.mark.parametrize("name", ["q1", "q2"])
def test_signal_mapping_names_both_critics(
    trained_agent: XQCAgent, name: str
) -> None:
    assert name in trained_agent.signal_mapping
