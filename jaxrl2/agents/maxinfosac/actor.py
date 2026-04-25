from typing import Tuple

import jax.numpy as jnp
import jax.random

from jaxrl.datasets import Batch
from jaxrl.networks.common import InfoDict, Model, Params, PRNGKey
from maxinforl.models.ensemble_model import EnsembleState, DeterministicEnsemble


def update(key: PRNGKey,
           actor: Model,
           critic: Model,
           temp: Model,
           target_actor: Model,
           dyn_entropy_temp: Model,
           ens: DeterministicEnsemble,
           ens_state: EnsembleState,
           batch: Batch) -> Tuple[Model, EnsembleState, InfoDict]:
    key, target_key = jax.random.split(key, 2)

    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[EnsembleState, InfoDict]]:
        dist = actor.apply_fn({'params': actor_params}, batch.observations)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)

        # getting info gain objective
        target_actions = target_actor(batch.observations).sample(seed=target_key)
        target_inp = jnp.concatenate([batch.observations, target_actions], axis=-1)
        inp = jnp.concatenate([batch.observations, actions], axis=-1)
        total_inp = jnp.concatenate([inp, target_inp], axis=0)
        info_gain, new_ens_state = ens.get_info_gain(input=total_inp,
                                                     state=ens_state,
                                                     update_normalizer=True)
        info_gain, target_info_gain = info_gain[:actions.shape[0]], info_gain[actions.shape[0]:]
        dyn_ent_coef, _ = dyn_entropy_temp()
        act_ent_coef, _ = temp()
        total_entropy = dyn_ent_coef * info_gain - act_ent_coef * log_probs
        actor_loss = -(total_entropy + q).mean()
        return actor_loss, (new_ens_state, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'info_gain': info_gain.mean(),
            'target_info_gain': target_info_gain.mean(),
            'dyn_ent_coef': dyn_ent_coef,
            'act_ent_coef': act_ent_coef,
            # 'actor_info_gain_nans': jnp.isnan(info_gain).sum(),
            # 'target_actor_info_gain_nans': jnp.isnan(target_info_gain).sum(),
        })

    new_actor, (new_ens_state, info) = actor.apply_gradient(actor_loss_fn)

    return new_actor, new_ens_state, info
