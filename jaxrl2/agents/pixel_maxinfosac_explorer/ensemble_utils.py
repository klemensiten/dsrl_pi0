from typing import Dict, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict
import numpy as np

from jaxrl2.data.dataset import DatasetDict


MODALITY_ORDER = ('latent', 'image', 'state', 'tactile')


def flatten_model_obs(observations: Union[DatasetDict, FrozenDict],
                      model_obs_key: str) -> jnp.ndarray:
    obs = observations[model_obs_key]
    return jnp.reshape(obs, (obs.shape[0], -1))


def flatten_actions(actions: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(actions, (actions.shape[0], -1))


def flatten_embedding(embedding: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(embedding, (embedding.shape[0], -1))


def ensemble_inputs(states: Union[jnp.ndarray, DatasetDict, FrozenDict],
                    actions: jnp.ndarray,
                    model_obs_key: Optional[str] = None) -> jnp.ndarray:
    if isinstance(states, (dict, FrozenDict)):
        if model_obs_key is None:
            raise ValueError(
                'model_obs_key is required when ensemble_inputs receives '
                'raw observations.')
        states = flatten_model_obs(states, model_obs_key)
    else:
        states = flatten_embedding(states)
    return jnp.concatenate([states, flatten_actions(actions)], axis=-1)


def available_modalities(modalities: Union[DatasetDict, FrozenDict]
                         ) -> Tuple[str, ...]:
    return tuple(key for key in MODALITY_ORDER if key in modalities)


def _flatten_modalities(modalities: Union[DatasetDict, FrozenDict],
                        modality_order: Sequence[str]) -> Dict[str, jnp.ndarray]:
    return {
        key: flatten_embedding(modalities[key])
        for key in modality_order
    }


def modality_output_slices(modalities: Union[DatasetDict, FrozenDict],
                           modality_order: Sequence[str],
                           predict_reward: bool) -> Dict[str, Tuple[int, int]]:
    start = 0
    slices = {}
    for key in modality_order:
        if key not in modalities:
            raise ValueError(f"Ensemble output modality '{key}' is unavailable.")
        dim = int(np.prod(modalities[key].shape[1:]))
        slices[key] = (start, start + dim)
        start += dim
    if predict_reward:
        slices['reward'] = (start, start + 1)
    return slices


def disagreement_weights(
        output_slices: Dict[str, Tuple[int, int]],
        selected_modalities: Optional[Sequence[str]] = None) -> jnp.ndarray:
    output_dim = max(end for _, end in output_slices.values())
    if selected_modalities is None:
        selected_modalities = tuple(
            key for key in output_slices.keys() if key != 'reward')
    else:
        selected_modalities = tuple(selected_modalities)

    missing = tuple(key for key in selected_modalities
                    if key not in output_slices or key == 'reward')
    if missing:
        raise ValueError(
            'Requested ensemble disagreement modalities are unavailable: '
            f'{missing}. Available modalities: '
            f'{tuple(k for k in output_slices.keys() if k != "reward")}.')

    selected_dim = sum(output_slices[key][1] - output_slices[key][0]
                       for key in selected_modalities)
    if selected_dim <= 0:
        raise ValueError('At least one ensemble disagreement dimension is '
                         'required.')

    weights = np.zeros(output_dim, dtype=np.float32)
    scale = float(output_dim) / float(selected_dim)
    for key in selected_modalities:
        start, end = output_slices[key]
        weights[start:end] = scale
    return jnp.asarray(weights)


def ensemble_targets_from_modalities(
        modalities: Union[DatasetDict, FrozenDict],
        next_modalities: Union[DatasetDict, FrozenDict],
        rewards: jnp.ndarray,
        predict_reward: bool,
        predict_diff: bool,
        modality_order: Sequence[str]) -> jnp.ndarray:
    modalities = _flatten_modalities(modalities, modality_order)
    next_modalities = _flatten_modalities(next_modalities, modality_order)
    targets = []
    for key in modality_order:
        target = next_modalities[key]
        if predict_diff:
            target = target - modalities[key]
        targets.append(target)
    if predict_reward:
        targets.append(jnp.reshape(rewards, (-1, 1)))
    return jnp.concatenate(targets, axis=-1)


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
