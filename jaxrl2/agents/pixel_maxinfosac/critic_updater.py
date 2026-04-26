from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_maxinfosac.ensemble_utils import ensemble_inputs
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.ensemble_model import DeterministicEnsemble, EnsembleState
from jaxrl2.types import Params, PRNGKey


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState, temp: TrainState, dyn_ent_temp: TrainState,
        ens: DeterministicEnsemble, ens_state: EnsembleState,
        batch: DatasetDict,
        discount: float, backup_entropy: bool = False,
        model_obs_key: str = 'state',
        critic_reduction: str = 'min') -> Tuple[TrainState, EnsembleState, Dict[str, float]]:
    dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    info_gain, new_ens_state = ens.get_info_gain(
        input=ensemble_inputs(batch['next_observations'], next_actions,
                              model_obs_key),
        state=ens_state,
        update_normalizer=False)
    act_ent_coef = temp.apply_fn({'params': temp.params})
    dyn_ent_coef = dyn_ent_temp.apply_fn({'params': dyn_ent_temp.params})
    if backup_entropy:
        entropy_bonus = dyn_ent_coef * info_gain - act_ent_coef * next_log_probs
        target_q += batch["discount"] * batch['masks'] * entropy_bonus
    else:
        entropy_bonus = jnp.zeros_like(next_log_probs)

    def critic_loss_fn(
            critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        qs = critic.apply_fn({'params': critic_params}, batch['observations'],
                             batch['actions'])
        critic_loss = ((qs - target_q)**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': qs.mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_qs.mean(),
            'target_q': target_q.mean(),
            'critic_info_gain': info_gain.mean(),
            'critic_entropy_bonus': entropy_bonus.mean(),
            'dyn_ent_temperature': dyn_ent_coef,
            'act_ent_temperature': act_ent_coef,
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'next_log_probs': next_log_probs.mean(),
            
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, new_ens_state, info
