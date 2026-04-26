from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.agents.pixel_maxinfosac.ensemble_utils import ensemble_inputs
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.ensemble_model import DeterministicEnsemble, EnsembleState
from jaxrl2.types import Params, PRNGKey


def update_actor(key: PRNGKey, actor: TrainState, critic: TrainState,
                 temp: TrainState, dyn_ent_temp: TrainState,
                 ens: DeterministicEnsemble, ens_state: EnsembleState,
                 target_actor_params: Params, batch: DatasetDict,
                 model_obs_key: str, cross_norm: bool = False,
                 critic_reduction: str = 'min') -> Tuple[TrainState, EnsembleState, Dict[str, float]]:
    
    key, key_act, key_target = jax.random.split(key, num=3)

    def actor_loss_fn(
            actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            dist, new_model_state = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['observations'], mutable=['batch_stats'])
            if cross_norm:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            else:
                next_dist = actor.apply_fn({'params': actor_params, 'batch_stats': actor.batch_stats}, batch['next_observations'])
            if type(next_dist) == tuple:
                next_dist, new_model_state = next_dist
        else:
            dist = actor.apply_fn({'params': actor_params}, batch['observations'])
            next_dist = actor.apply_fn({'params': actor_params}, batch['next_observations'])
            new_model_state = {}
        
        # For logging only
        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)

        
        actions, log_probs = dist.sample_and_log_prob(seed=key_act)

        if hasattr(critic, 'batch_stats') and critic.batch_stats is not None:
            qs, _ = critic.apply_fn({'params': critic.params, 'batch_stats': critic.batch_stats}, batch['observations'],
                            actions, mutable=['batch_stats'])
        else:    
            qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        
        if critic_reduction == 'min':
            q = qs.min(axis=0)
        elif critic_reduction == 'mean':
            q = qs.mean(axis=0)
        else:
            raise ValueError(f"Invalid critic reduction: {critic_reduction}")

        inp = ensemble_inputs(batch['observations'], actions, model_obs_key)
        if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
            target_dist = actor.apply_fn(
                {'params': target_actor_params, 'batch_stats': actor.batch_stats},
                batch['observations'])
        else:
            target_dist = actor.apply_fn(
                {'params': target_actor_params}, batch['observations'])
        target_actions = target_dist.sample(seed=key_target)
        target_inp = ensemble_inputs(batch['observations'], target_actions,
                                     model_obs_key)
        total_inp = jnp.concatenate([inp, target_inp], axis=0)
        info_gain, new_ens_state = ens.get_info_gain(
            input=total_inp,
            state=ens_state,
            update_normalizer=True)
        info_gain, target_info_gain = (
            info_gain[:actions.shape[0]],
            info_gain[actions.shape[0]:],
        )

        act_ent_coef = temp.apply_fn({'params': temp.params})
        dyn_ent_coef = dyn_ent_temp.apply_fn({'params': dyn_ent_temp.params})
        actor_loss = (act_ent_coef * log_probs - dyn_ent_coef * info_gain - q).mean()

        things_to_log = {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'info_gain': info_gain.mean(),
            'target_info_gain': target_info_gain.mean(),
            'dyn_ent_temperature': dyn_ent_coef,
            'act_ent_temperature': act_ent_coef,
            'q_pi_in_actor': q.mean(),
            'mean_pi_norm': mean_dist_norm.mean(),
            'std_pi_norm': std_dist_norm.mean(),
            'mean_pi_avg': mean_dist.mean(),
            'mean_pi_max': mean_dist.max(),
            'mean_pi_min': mean_dist.min(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_max': std_diag_dist.max(),
            'std_pi_min': std_diag_dist.min(),
        }
        return actor_loss, (new_ens_state, things_to_log, new_model_state)

    grads, (new_ens_state, info, new_model_state) = jax.grad(
        actor_loss_fn, has_aux=True)(actor.params)
    
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, new_ens_state, info
