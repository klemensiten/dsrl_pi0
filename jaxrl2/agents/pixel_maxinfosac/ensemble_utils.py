from typing import Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.data.dataset import DatasetDict


def flatten_model_obs(observations: Union[DatasetDict, FrozenDict],
                      model_obs_key: str) -> jnp.ndarray:
    obs = observations[model_obs_key]
    return jnp.reshape(obs, (obs.shape[0], -1))


def flatten_actions(actions: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(actions, (actions.shape[0], -1))


def ensemble_inputs(observations: Union[DatasetDict, FrozenDict],
                    actions: jnp.ndarray,
                    model_obs_key: str) -> jnp.ndarray:
    return jnp.concatenate(
        [flatten_model_obs(observations, model_obs_key), flatten_actions(actions)],
        axis=-1,
    )


def ensemble_targets(batch: Union[DatasetDict, FrozenDict],
                     model_obs_key: str,
                     predict_reward: bool,
                     predict_diff: bool) -> jnp.ndarray:
    observations = flatten_model_obs(batch['observations'], model_obs_key)
    next_observations = flatten_model_obs(batch['next_observations'],
                                          model_obs_key)
    if predict_diff:
        targets = next_observations - observations
    else:
        targets = next_observations

    if predict_reward:
        rewards = jnp.reshape(batch['rewards'], (-1, 1))
        targets = jnp.concatenate([targets, rewards], axis=-1)
    return targets
