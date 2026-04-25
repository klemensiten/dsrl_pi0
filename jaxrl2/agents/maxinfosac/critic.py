from typing import Tuple

import jax
import jax.numpy as jnp

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from maxinforl.models.ensemble_model import EnsembleState, DeterministicEnsemble


def target_update(critic: Model, target_critic: Model, tau: float) -> Model:
    new_target_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau, target_critic.params,
        critic.params)

    return target_critic.replace(params=new_target_params)


def update(key: PRNGKey,
           actor: Model,
           critic: Model,
           target_critic: Model,
           temp: Model,
           dyn_entropy_temp: Model,
           ens: DeterministicEnsemble,
           ens_state: EnsembleState,
           batch: Batch,
           discount: float,
           backup_entropy: bool) -> Tuple[Model, EnsembleState, InfoDict]:
    dist = actor(batch.next_observations)
    next_actions = dist.sample(seed=key)
    next_log_probs = dist.log_prob(next_actions)

    # get info gain
    info_gain, new_ens_state = ens.get_info_gain(
        input=jnp.concatenate([batch.next_observations, next_actions], axis=-1),
        state=ens_state, update_normalizer=False)

    next_q1, next_q2 = target_critic(batch.next_observations, next_actions)
    next_q = jnp.minimum(next_q1, next_q2)

    dyn_ent_coef, _ = dyn_entropy_temp()
    act_ent_coef, _ = temp()

    target_q = batch.rewards + discount * batch.masks * next_q

    if backup_entropy:
        total_entropy = dyn_ent_coef * info_gain - act_ent_coef * next_log_probs
        target_q += discount * batch.masks * total_entropy
    else:
        total_entropy = jnp.zeros_like(total_entropy)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        q1, q2 = critic.apply_fn({'params': critic_params}, batch.observations,
                                 batch.actions)
        critic_loss = ((q1 - target_q) ** 2 + (q2 - target_q) ** 2).mean()

        info: InfoDict = {
            'critic_loss': critic_loss,
            'q1': q1.mean(),
            'q2': q2.mean(),
            'dyn_ent_coef': dyn_ent_coef,
            'act_ent_coef': act_ent_coef,
            'critic_info_gain': info_gain.mean(),
            'critic_total_entropy_term': total_entropy.mean(),
        }
        return critic_loss, info

    new_critic, info = critic.apply_gradient(critic_loss_fn)

    return new_critic, new_ens_state, info
