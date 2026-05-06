from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_maxinfosac.ensemble_utils import (
    ensemble_inputs,
    flatten_actions,
)
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.ensemble_model import DeterministicEnsemble, EnsembleState
from jaxrl2.types import Params, PRNGKey


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, dyn_ent_temp: TrainState,
        ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        mask_expl_critic: bool = False,
        critic_reduction: str = 'min') -> Tuple[TrainState, EnsembleState, Dict[str, float]]:
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    batch_size = batch['actions'].shape[0]
    full_observations = jax.tree_util.tree_map(
        lambda next_obs, obs: jnp.concatenate([next_obs, obs], axis=0),
        batch['next_observations'],
        batch['observations'])
    batch_actions = flatten_actions(batch['actions'])
    full_actions = jnp.concatenate([next_actions, batch_actions], axis=0)

    target_critic_vars = {'params': target_critic.params}
    if getattr(target_critic, 'batch_stats', None) is not None:
        target_critic_vars['batch_stats'] = target_critic.batch_stats
    next_qs_all, expl_next_qs_all, states_all, modalities_all = \
        target_critic.apply_fn(target_critic_vars, full_observations,
                               full_actions)
    next_qs = next_qs_all[:, :batch_size]
    expl_next_qs = expl_next_qs_all[:, :batch_size]
    state = states_all[batch_size:]
    next_state = states_all[:batch_size]
    modalities = jax.tree_util.tree_map(lambda x: x[batch_size:],
                                        modalities_all)
    next_modalities = jax.tree_util.tree_map(lambda x: x[:batch_size],
                                             modalities_all)

    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
        expl_next_q = expl_next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
        expl_next_q = expl_next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    info_gain, new_ens_state = ens.get_info_gain(
        input=ensemble_inputs(state, batch['actions']),
        state=ens_state,
        update_normalizer=False)

    act_ent_coef = temp.apply_fn({'params': temp.params})
    dyn_ent_coef = dyn_ent_temp.apply_fn({'params': dyn_ent_temp.params})
    if backup_entropy:
        entropy_bonus = -act_ent_coef * next_log_probs
        target_q += batch["discount"] * batch['masks'] * entropy_bonus
    else:
        entropy_bonus = jnp.zeros_like(next_log_probs)

    if mask_expl_critic:
        expl_discount = batch["discount"] * batch['masks']
    else:
        expl_discount = batch["discount"]
    target_expl_q = info_gain + expl_discount * expl_next_q

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        critic_vars = {'params': critic_params}
        if getattr(critic, 'batch_stats', None) is not None:
            critic_vars['batch_stats'] = critic.batch_stats
            (qs, expl_qs, _, _), new_model_state = critic.apply_fn(
                critic_vars,
                batch['observations'],
                batch['actions'],
                mutable=['batch_stats'])
        else:
            qs, expl_qs, _, _ = critic.apply_fn(
                critic_vars, batch['observations'], batch['actions'])
            new_model_state = {}
        critic_loss = (((qs - target_q[jnp.newaxis])**2).mean() +
                       ((expl_qs - target_expl_q[jnp.newaxis])**2).mean())
        return critic_loss, {
            'critic_loss': critic_loss,
            'task_critic_loss': ((qs - target_q[jnp.newaxis])**2).mean(),
            'expl_critic_loss': ((expl_qs - target_expl_q[jnp.newaxis])**2).mean(),
            'q': qs.mean(),
            'q_expl': expl_qs.mean(),
            'q1': qs[0].mean(),
            'q2': qs[-1].mean(),
            'q1_expl': expl_qs[0].mean(),
            'q2_expl': expl_qs[-1].mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'next_q_expl_pi': expl_next_qs.mean(),
            'target_q': target_q.mean(),
            'target_expl_q': target_expl_q.mean(),
            'expl_reward': info_gain.mean(),
            'critic_info_gain': info_gain.mean(),
            'critic_entropy_bonus': entropy_bonus.mean(),
            'dyn_ent_temperature': dyn_ent_coef,
            'act_ent_temperature': act_ent_coef,
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'state': jax.lax.stop_gradient(state),
            'next_state': jax.lax.stop_gradient(next_state),
            'modalities': jax.tree_util.tree_map(jax.lax.stop_gradient,
                                                 modalities),
            'next_modalities': jax.tree_util.tree_map(
                jax.lax.stop_gradient, next_modalities),
            'new_model_state': new_model_state,
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_model_state = info.pop('new_model_state')
    if 'batch_stats' in new_model_state:
        new_critic = critic.apply_gradients(
            grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_critic = critic.apply_gradients(grads=grads)

    return new_critic, new_ens_state, info
