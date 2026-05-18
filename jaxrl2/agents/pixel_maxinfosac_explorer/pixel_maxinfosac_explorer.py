"""Implementations of algorithms for continuous control."""
import matplotlib
matplotlib.use('Agg')
from flax.training import checkpoints
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

import numpy as np
import copy
import functools
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state
from typing import Any

from jaxrl2.agents.agent import Agent, get_batch_stats
from jaxrl2.agents.common import sample_actions_jit
from jaxrl2.data.augmentations import batched_random_crop, color_transform
from jaxrl2.networks.encoders.networks import Encoder, PixelMultiplexer
from jaxrl2.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from jaxrl2.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from jaxrl2.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from jaxrl2.agents.pixel_maxinfosac_explorer.ensemble_utils import (
    available_modalities,
    disagreement_weights,
    ensemble_inputs,
    ensemble_targets_from_modalities,
    flatten_actions,
    modality_output_slices,
)
from jaxrl2.agents.pixel_maxinfosac_explorer.networks import PixelMaxInfoCritic
from jaxrl2.agents.pixel_maxinfosac_explorer.temperature_updater import update_temperature
from jaxrl2.agents.pixel_maxinfosac_explorer.temperature import Temperature
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy
from jaxrl2.networks.ensemble_model import DeterministicEnsemble, EnsembleState
from jaxrl2.types import Params, PRNGKey
from jaxrl2.utils.target_update import soft_target_update


class TrainState(train_state.TrainState):
    batch_stats: Any


def _actor_vars(actor: TrainState, params: Params) -> Dict[str, Any]:
    variables = {'params': params}
    if getattr(actor, 'batch_stats', None) is not None:
        variables['batch_stats'] = actor.batch_stats
    return variables


def _critic_vars(critic: TrainState, params: Params) -> Dict[str, Any]:
    variables = {'params': params}
    if getattr(critic, 'batch_stats', None) is not None:
        variables['batch_stats'] = critic.batch_stats
    return variables


def _reduce_q(qs: jnp.ndarray, critic_reduction: str) -> jnp.ndarray:
    if critic_reduction == 'min':
        return qs.min(axis=0)
    if critic_reduction == 'mean':
        return qs.mean(axis=0)
    raise ValueError(f"Invalid critic reduction: {critic_reduction}")


def _select_q_head(qs: jnp.ndarray, expl_qs: jnp.ndarray,
                   q_head: str) -> jnp.ndarray:
    if q_head == 'task':
        return qs
    if q_head == 'expl':
        return expl_qs
    raise ValueError(f"Invalid q_head: {q_head}")


def _apply_actor(actor: TrainState, params: Params,
                 observations: DatasetDict):
    return actor.apply_fn(_actor_vars(actor, params), observations)


def _augment_batch(rng: PRNGKey, batch: DatasetDict, color_jitter: bool,
                   aug_next: bool, num_cameras: int) -> Tuple[PRNGKey, DatasetDict]:
    aug_pixels = batch['observations']['pixels']
    aug_next_pixels = batch['next_observations']['pixels']
    if batch['observations']['pixels'].squeeze().ndim != 2:
        rng, key = jax.random.split(rng)
        aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_pixels = aug_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

    observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
    batch = batch.copy(add_or_replace={'observations': observations})

    key, rng = jax.random.split(rng)
    if aug_next:
        rng, key = jax.random.split(rng)
        aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
        if color_jitter:
            rng, key = jax.random.split(rng)
            if num_cameras > 1:
                for i in range(num_cameras):
                    aug_next_pixels = aug_next_pixels.at[:,:,:,i*3:(i+1)*3].set((color_transform(key, aug_next_pixels[:,:,:,i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8))
            else:
                aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
        next_observations = batch['next_observations'].copy(
            add_or_replace={'pixels': aug_next_pixels})
        batch = batch.copy(add_or_replace={'next_observations': next_observations})

    return rng, batch


def _critic_context(critic: TrainState,
                    target_critic_params: Params,
                    batch: DatasetDict) -> Tuple[jnp.ndarray, jnp.ndarray,
                                                  DatasetDict, DatasetDict]:
    batch_size = batch['actions'].shape[0]
    full_observations = jax.tree_util.tree_map(
        lambda next_obs, obs: jnp.concatenate([next_obs, obs], axis=0),
        batch['next_observations'],
        batch['observations'])
    batch_actions = flatten_actions(batch['actions'])
    full_actions = jnp.concatenate([batch_actions, batch_actions], axis=0)
    _, _, states_all, modalities_all = critic.apply_fn(
        _critic_vars(critic, target_critic_params),
        full_observations,
        full_actions)
    next_state = states_all[:batch_size]
    state = states_all[batch_size:]
    next_modalities = jax.tree_util.tree_map(lambda x: x[:batch_size],
                                             modalities_all)
    modalities = jax.tree_util.tree_map(lambda x: x[batch_size:],
                                        modalities_all)
    return state, next_state, modalities, next_modalities


def _update_critic_sac(key: PRNGKey, actor: TrainState, critic: TrainState,
                       target_critic_params: Params, temp: TrainState,
                       batch: DatasetDict, discount: float,
                       backup_entropy: bool, critic_reduction: str,
                       q_head: str,
                       mask_target: bool = True) -> Tuple[TrainState, Dict[str, float]]:
    dist = _apply_actor(actor, actor.params, batch['next_observations'])
    next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    target_critic = critic.replace(params=target_critic_params)
    target_qs, target_expl_qs, _, _ = target_critic.apply_fn(
        _critic_vars(target_critic, target_critic.params),
        batch['next_observations'],
        next_actions)
    next_q = _reduce_q(
        _select_q_head(target_qs, target_expl_qs, q_head),
        critic_reduction)

    bootstrap = batch["discount"] * (batch['masks'] if mask_target else 1.0)
    target_q = batch['rewards'] + bootstrap * next_q
    act_ent_coef = temp.apply_fn({'params': temp.params})
    if backup_entropy:
        entropy_bonus = -act_ent_coef * next_log_probs
        target_q += bootstrap * entropy_bonus
    else:
        entropy_bonus = jnp.zeros_like(next_log_probs)

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        critic_vars = _critic_vars(critic, critic_params)
        if getattr(critic, 'batch_stats', None) is not None:
            (qs, expl_qs, _, _), new_model_state = critic.apply_fn(
                critic_vars,
                batch['observations'],
                batch['actions'],
                mutable=['batch_stats'])
        else:
            qs, expl_qs, _, _ = critic.apply_fn(
                critic_vars, batch['observations'], batch['actions'])
            new_model_state = {}

        selected_qs = _select_q_head(qs, expl_qs, q_head)
        critic_loss = ((selected_qs - target_q[jnp.newaxis])**2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'q': selected_qs.mean(),
            'q1': selected_qs[0].mean(),
            'q2': selected_qs[-1].mean(),
            'target_actor_entropy': -next_log_probs.mean(),
            'next_actions_sampled': next_actions.mean(),
            'next_log_probs': next_log_probs.mean(),
            'next_q_pi': next_q.mean(),
            'target_q': target_q.mean(),
            'critic_entropy_bonus': entropy_bonus.mean(),
            'act_ent_temperature': act_ent_coef,
            'next_actions_mean': next_actions.mean(),
            'next_actions_std': next_actions.std(),
            'next_actions_min': next_actions.min(),
            'next_actions_max': next_actions.max(),
            'new_model_state': new_model_state,
        }

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_model_state = info.pop('new_model_state')
    if 'batch_stats' in new_model_state:
        new_critic = critic.apply_gradients(
            grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_critic = critic.apply_gradients(grads=grads)
    return new_critic, info


def _update_actor_sac(key: PRNGKey, actor: TrainState, critic: TrainState,
                      temp: TrainState, batch: DatasetDict,
                      critic_reduction: str,
                      q_head: str) -> Tuple[TrainState, Dict[str, float]]:
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        if getattr(actor, 'batch_stats', None) is not None:
            dist, new_model_state = actor.apply_fn(
                _actor_vars(actor, actor_params),
                batch['observations'],
                mutable=['batch_stats'])
        else:
            dist = actor.apply_fn(
                _actor_vars(actor, actor_params),
                batch['observations'])
            new_model_state = {}

        mean_dist = dist.distribution._loc
        std_diag_dist = dist.distribution._scale_diag
        mean_dist_norm = jnp.linalg.norm(mean_dist, axis=-1)
        std_dist_norm = jnp.linalg.norm(std_diag_dist, axis=-1)
        actions, log_probs = dist.sample_and_log_prob(seed=key)

        qs, expl_qs, _, _ = critic.apply_fn(
            _critic_vars(critic, critic.params),
            batch['observations'],
            actions)
        q = _reduce_q(_select_q_head(qs, expl_qs, q_head), critic_reduction)
        act_ent_coef = temp.apply_fn({'params': temp.params})
        actor_loss = (act_ent_coef * log_probs - q).mean()
        return actor_loss, {
            'actor_loss': actor_loss,
            'entropy': -log_probs.mean(),
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
            'new_model_state': new_model_state,
        }

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_model_state = info.pop('new_model_state')
    if 'batch_stats' in new_model_state:
        new_actor = actor.apply_gradients(
            grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)
    return new_actor, info


def _prefix_info(prefix: str, info: Dict[str, float]) -> Dict[str, float]:
    return {f'{prefix}_{key}': value for key, value in info.items()}


@functools.partial(
    jax.jit,
    static_argnames=(
        'critic_reduction',
        'color_jitter',
        'aug_next',
        'num_cameras',
        'backup_entropy',
        'ens',
        'predict_reward',
        'predict_diff',
        'ensemble_output_modalities',
        'mask_expl_critic',
        'update_agent',
        'update_expl_agent',
        'update_ensemble',
    ))
def _update_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_actor_params: Params,
    target_critic_params: Params,
    temp: TrainState,
    expl_actor: TrainState,
    expl_critic: TrainState,
    expl_target_actor_params: Params,
    expl_target_critic_params: Params,
    expl_temp: TrainState,
    ens: DeterministicEnsemble,
    ens_state: EnsembleState,
    batch: DatasetDict,
    discount: float,
    tau: float,
    target_entropy: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    backup_entropy: bool,
    predict_reward: bool,
    predict_diff: bool,
    ensemble_output_modalities: Tuple[str, ...],
    mask_expl_critic: bool,
    update_agent: bool,
    update_expl_agent: bool,
    update_ensemble: bool,
) -> Tuple[PRNGKey, TrainState, TrainState, Params, Params, TrainState,
           TrainState, TrainState, Params, Params, TrainState, EnsembleState,
           Dict[str, float]]:
    rng, batch = _augment_batch(rng, batch, color_jitter, aug_next,
                                num_cameras)

    model_state, _, model_modalities, model_next_modalities = _critic_context(
        critic, target_critic_params, batch)
    model_input = ensemble_inputs(model_state, batch['actions'])
    expl_rewards, ens_state = ens.get_info_gain(
        input=model_input,
        state=ens_state,
        update_normalizer=True)

    info = {
        'expl_reward': expl_rewards.mean(),
    }

    new_actor = actor
    new_critic = critic
    new_target_actor_params = target_actor_params
    new_target_critic_params = target_critic_params
    new_temp = temp
    if update_agent:
        rng, key = jax.random.split(rng)
        target_critic = critic.replace(params=target_critic_params)
        new_critic, critic_info = _update_critic_sac(
            key, actor, critic, target_critic.params, temp, batch, discount,
            backup_entropy, critic_reduction, q_head='task',
            mask_target=True)
        new_target_critic_params = soft_target_update(
            new_critic.params, target_critic_params, tau)

        rng, key = jax.random.split(rng)
        new_actor, actor_info = _update_actor_sac(
            key, actor, new_critic, temp, batch, critic_reduction,
            q_head='task')
        new_target_actor_params = soft_target_update(
            new_actor.params, target_actor_params, tau)
        new_temp, alpha_info = update_temperature(
            temp, actor_info['entropy'], target_entropy)
        info = {
            **info,
            **_prefix_info('task', critic_info),
            **_prefix_info('task', actor_info),
            **_prefix_info('task_alpha', alpha_info),
        }

    new_expl_actor = expl_actor
    new_expl_critic = expl_critic
    new_expl_target_actor_params = expl_target_actor_params
    new_expl_target_critic_params = expl_target_critic_params
    new_expl_temp = expl_temp
    if update_expl_agent:
        expl_batch = batch.copy(add_or_replace={'rewards': expl_rewards})
        rng, key = jax.random.split(rng)
        new_expl_critic, expl_critic_info = _update_critic_sac(
            key, expl_actor, expl_critic, expl_target_critic_params,
            expl_temp, expl_batch, discount, backup_entropy,
            critic_reduction, q_head='expl',
            mask_target=mask_expl_critic)
        new_expl_target_critic_params = soft_target_update(
            new_expl_critic.params, expl_target_critic_params, tau)

        rng, key = jax.random.split(rng)
        new_expl_actor, expl_actor_info = _update_actor_sac(
            key, expl_actor, new_expl_critic, expl_temp, expl_batch,
            critic_reduction, q_head='expl')
        new_expl_target_actor_params = soft_target_update(
            new_expl_actor.params, expl_target_actor_params, tau)
        new_expl_temp, expl_alpha_info = update_temperature(
            expl_temp, expl_actor_info['entropy'], target_entropy)
        info = {
            **info,
            **_prefix_info('expl', expl_critic_info),
            **_prefix_info('expl', expl_actor_info),
            **_prefix_info('expl_alpha', expl_alpha_info),
        }

    if update_ensemble:
        model_output = ensemble_targets_from_modalities(
            model_modalities,
            model_next_modalities,
            batch['rewards'],
            predict_reward,
            predict_diff,
            ensemble_output_modalities,
        )
        ens_state, (loss, mse) = ens.update(
            input=model_input,
            output=model_output,
            state=ens_state,
        )
        normalizer_state = ens_state.ensemble_normalizer_state
        info = {
            **info,
            'ens_nll': loss,
            'ens_mse': mse,
            'dyn_model_loss': loss,
            'dyn_model_mse': mse,
            'ens_inp_mean': normalizer_state.input_normalizer_state.mean.mean(),
            'ens_inp_std': normalizer_state.input_normalizer_state.std.mean(),
            'ens_out_mean': normalizer_state.output_normalizer_state.mean.mean(),
            'ens_out_std': normalizer_state.output_normalizer_state.std.mean(),
            'ens_info_gain_mean': normalizer_state.info_gain_normalizer_state.mean.mean(),
            'ens_info_gain_std': normalizer_state.info_gain_normalizer_state.std.mean(),
        }

    return (rng, new_actor, new_critic, new_target_actor_params,
            new_target_critic_params, new_temp, new_expl_actor,
            new_expl_critic, new_expl_target_actor_params,
            new_expl_target_critic_params, new_expl_temp, ens_state, info)


class PixelMaxinfoSACExplorer(Agent):

    def __init__(self,
                 seed: int,
                 observations: Union[jnp.ndarray, DatasetDict],
                 actions: jnp.ndarray,
                 actor_lr: float = 3e-4,
                 critic_lr: float = 3e-4,
                 temp_lr: float = 3e-4,
                 decay_steps: Optional[int] = None,
                 hidden_dims: Sequence[int] = (256, 256),
                 cnn_features: Sequence[int] = (32, 32, 32, 32),
                 cnn_strides: Sequence[int] = (2, 1, 1, 1),
                 cnn_padding: str = 'VALID',
                 latent_dim: int = 50,
                 discount: float = 0.99,
                 tau: float = 0.005,
                 critic_reduction: str = 'mean',
                 dropout_rate: Optional[float] = None,
                 encoder_type='resnet_34_v1',
                 encoder_norm='group',
                 color_jitter = True,
                 use_spatial_softmax=True,
                 softmax_temperature=1,
                 aug_next=True,
                 use_bottleneck=True,
                 init_temperature: float = 1.0,
                 dyn_ent_lr: float = 3e-4,
                 init_dyn_ent_temperature: float = 1.0,
                 model_lr: float = 3e-4,
                 model_wd: float = 0.0,
                 model_hidden_dims: Sequence[int] = (256, 256),
                 num_model_heads: int = 5,
                 model_noise_var: float = 1.0,
                 predict_reward: bool = True,
                 predict_diff: bool = True,
                 backup_entropy: bool = True,
                 model_obs_key: str = "state",
                 obs_dim: int = 64,
                 ensemble_disagreement_modalities: Optional[Sequence[str]] = None,
                 mask_expl_critic: bool = False,
                 tactile_hidden_dims: Sequence[int] = (256, 256),
                 mask_touch: bool = False,
                 num_qs: int = 2,
                 target_entropy: float = None,
                 action_magnitude: float = 1.0,
                 num_cameras: int = 1,
                 explore_until: int = 300000,
                 agent_update_period: int = 1,
                 expl_agent_update_period: int = 1,
                 ensemble_update_period: int = 1,
                 ):
        """
        An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905
        """

        self.aug_next=aug_next
        self.color_jitter = color_jitter
        self.num_cameras = num_cameras

        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.backup_entropy = backup_entropy
        self.model_obs_key = model_obs_key
        self.model_noise_var = model_noise_var
        self.predict_reward = predict_reward
        self.predict_diff = predict_diff
        self.mask_expl_critic = mask_expl_critic
        self.explore_until = explore_until
        self.agent_update_period = agent_update_period
        self.expl_agent_update_period = expl_agent_update_period
        self.ensemble_update_period = ensemble_update_period
        self._step = 0
        self._exploration_steps = 0

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, expl_actor_key, critic_key, expl_critic_key, \
            temp_key, expl_temp_key, ensemble_key = jax.random.split(rng, 8)

        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            print('using impala')
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            print('using impala small')
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        else:
            raise ValueError('encoder type not found!')

        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        policy_def = LearnedStdTanhNormalPolicy(hidden_dims, self.action_dim, dropout_rate=dropout_rate, low=-action_magnitude, high=action_magnitude)

        actor_def = PixelMultiplexer(encoder=encoder_def,
                                     network=policy_def,
                                     latent_dim=latent_dim,
                                     use_bottleneck=use_bottleneck
                                     )
        print(actor_def)
        actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init['batch_stats'] if 'batch_stats' in actor_def_init else None

        actor = TrainState.create(apply_fn=actor_def.apply,
                                  params=actor_params,
                                  tx=optax.adam(learning_rate=actor_lr),
                                  batch_stats=actor_batch_stats)
        target_actor_params = copy.deepcopy(actor_params)

        expl_actor_def_init = actor_def.init(expl_actor_key, observations)
        expl_actor_params = expl_actor_def_init['params']
        expl_actor_batch_stats = expl_actor_def_init['batch_stats'] if 'batch_stats' in expl_actor_def_init else None
        expl_actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=expl_actor_params,
            tx=optax.adam(learning_rate=actor_lr),
            batch_stats=expl_actor_batch_stats)
        expl_target_actor_params = copy.deepcopy(expl_actor_params)

        critic_def = PixelMaxInfoCritic(
            encoder=encoder_def,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            obs_dim=obs_dim,
            num_qs=num_qs,
            model_obs_key=model_obs_key,
            use_bottleneck=use_bottleneck,
            tactile_hidden_dims=tactile_hidden_dims,
            tactile_cnn_features=cnn_features,
            tactile_cnn_strides=cnn_strides,
            tactile_cnn_padding=cnn_padding,
            mask_touch=mask_touch)
        print(critic_def)
        critic_def_init = critic_def.init(critic_key, observations, actions)
        self._critic_init_params = critic_def_init['params']

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init['batch_stats'] if 'batch_stats' in critic_def_init else None
        critic = TrainState.create(apply_fn=critic_def.apply,
                                   params=critic_params,
                                   tx=optax.adam(learning_rate=critic_lr),
                                   batch_stats=critic_batch_stats
                                   )
        target_critic_params = copy.deepcopy(critic_params)

        expl_critic_def_init = critic_def.init(expl_critic_key, observations,
                                               actions)
        expl_critic_params = expl_critic_def_init['params']
        expl_critic_batch_stats = expl_critic_def_init['batch_stats'] if 'batch_stats' in expl_critic_def_init else None
        expl_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=expl_critic_params,
            tx=optax.adam(learning_rate=critic_lr),
            batch_stats=expl_critic_batch_stats)
        expl_target_critic_params = copy.deepcopy(expl_critic_params)

        # TODO: Debug block
        critic_vars = {'params': critic_params}
        if critic_batch_stats is not None:
            critic_vars['batch_stats'] = critic_batch_stats
        _, _, sample_state, sample_modalities = critic_def.apply(
            critic_vars, observations, actions)
        ensemble_output_modalities = available_modalities(sample_modalities)
        if not ensemble_output_modalities:
            raise ValueError(
                'PixelMaxInfoSAC could not build any ensemble output modalities from the critic.')
        output_slices = modality_output_slices(
            sample_modalities,
            ensemble_output_modalities,
            predict_reward=predict_reward)
        
        temp_def = Temperature(init_temperature)
        temp_params = temp_def.init(temp_key)['params']
        temp = TrainState.create(apply_fn=temp_def.apply,
                                 params=temp_params,
                                 tx=optax.adam(learning_rate=temp_lr),
                                 batch_stats=None)

        expl_temp_def = Temperature(init_temperature)
        expl_temp_params = expl_temp_def.init(expl_temp_key)['params']
        expl_temp = TrainState.create(apply_fn=expl_temp_def.apply,
                                      params=expl_temp_params,
                                      tx=optax.adam(learning_rate=temp_lr),
                                      batch_stats=None)

        model_output_dim = max(end for _, end in output_slices.values())
        model_optimizer = optax.adamw(learning_rate=model_lr, weight_decay=model_wd)
        ensemble = DeterministicEnsemble(
            model_kwargs={'hidden_dims': tuple(model_hidden_dims) + (model_output_dim,)},
            optimizer=model_optimizer,
            num_heads=num_model_heads)
        model_input = ensemble_inputs(sample_state, actions)
        ens_state = ensemble.init(key=ensemble_key, input=model_input)
        ensemble.set_disg_weights(
            disagreement_weights(output_slices,
                                 ensemble_disagreement_modalities))

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_actor_params = target_actor_params
        self._target_critic_params = target_critic_params
        self._temp = temp
        self._expl_actor = expl_actor
        self._expl_critic = expl_critic
        self._expl_target_actor_params = expl_target_actor_params
        self._expl_target_critic_params = expl_target_critic_params
        self._expl_temp = expl_temp
        self._ensemble = ensemble
        self._ens_state = ens_state
        self._ensemble_output_modalities = ensemble_output_modalities
        self._ensemble_output_slices = output_slices
        self._ensemble_disagreement_modalities = (
            None if ensemble_disagreement_modalities is None
            else tuple(ensemble_disagreement_modalities))
        if target_entropy is None or target_entropy == 'auto':
            self.target_entropy = -self.action_dim / 2
        else:
            self.target_entropy = float(target_entropy)
        print(f'target_entropy: {self.target_entropy}')
        print(self.critic_reduction)

    @property
    def collecting_exploration(self) -> bool:
        return self._step <= self.explore_until

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        actor = self._expl_actor if self.collecting_exploration else self._actor
        rng, actions = sample_actions_jit(
            self._rng,
            actor.apply_fn,
            actor.params,
            observations,
            get_batch_stats(actor))
        self._rng = rng
        if self.collecting_exploration:
            self._exploration_steps += 1
        return np.asarray(actions)

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        self._step += 1
        update_agent = self._step % self.agent_update_period == 0
        update_expl_agent = (
            self.collecting_exploration and
            self._step % self.expl_agent_update_period == 0)
        update_ensemble = self._step % self.ensemble_update_period == 0

        new_rng, new_actor, new_critic, new_target_actor, new_target_critic, \
            new_temp, new_expl_actor, new_expl_critic, \
            new_expl_target_actor, new_expl_target_critic, new_expl_temp, \
            new_ens_state, info = _update_jit(
                self._rng, self._actor, self._critic,
                self._target_actor_params, self._target_critic_params,
                self._temp, self._expl_actor, self._expl_critic,
                self._expl_target_actor_params,
                self._expl_target_critic_params, self._expl_temp,
                self._ensemble, self._ens_state, batch,
                self.discount, self.tau, self.target_entropy,
                self.critic_reduction, self.color_jitter, self.aug_next,
                self.num_cameras, self.backup_entropy,
                self.predict_reward, self.predict_diff,
                self._ensemble_output_modalities,
                self.mask_expl_critic,
                update_agent, update_expl_agent, update_ensemble
            )

        self._rng = new_rng
        self._actor = new_actor
        self._critic = new_critic
        self._target_actor_params = new_target_actor
        self._target_critic_params = new_target_critic
        self._temp = new_temp
        self._expl_actor = new_expl_actor
        self._expl_critic = new_expl_critic
        self._expl_target_actor_params = new_expl_target_actor
        self._expl_target_critic_params = new_expl_target_critic
        self._expl_temp = new_expl_temp
        self._ens_state = new_ens_state
        return {
            **info,
            'policy_phase': jnp.asarray(int(self.collecting_exploration)),
            'exploration_steps': jnp.asarray(self._exploration_steps),
            'expl_update_active': jnp.asarray(int(update_expl_agent)),
            'learner_step': jnp.asarray(self._step),
        }

    def perform_eval(self, variant, i, wandb_logger, eval_buffer, eval_buffer_iterator, eval_env):
        from examples.train_utils_sim import make_multiple_value_reward_visulizations
        make_multiple_value_reward_visulizations(self, variant, i, eval_buffer, wandb_logger)

    def make_value_reward_visulization(self, variant, trajs):
        num_traj = len(trajs['rewards'])
        traj_images = []

        for itraj in range(num_traj):
            observations = trajs['observations'][itraj]
            next_observations = trajs['next_observations'][itraj]
            actions = trajs['actions'][itraj]
            rewards = trajs['rewards'][itraj]
            masks = trajs['masks'][itraj]

            q_pred = []

            for t in range(0, len(actions)):
                action = actions[t][None]
                obs_pixels = observations['pixels'][t]
                next_obs_pixels = next_observations['pixels'][t]

                obs_dict = {'pixels': obs_pixels[None]}
                for k, v in observations.items():
                    if 'pixels' not in k:
                        obs_dict[k] = v[t][None]
                next_obs_dict = {'pixels': next_obs_pixels[None]}
                for k, v in next_observations.items():
                    if 'pixels' not in k:
                        next_obs_dict[k] = v[t][None]

                q_value = get_value(action, obs_dict, self._critic)
                q_pred.append(q_value)

            traj_images.append(make_visual(q_pred, rewards, masks, observations['pixels']))
        print('finished reward value visuals.')
        return np.concatenate(traj_images, 0)

    @property
    def _save_dict(self):
        save_dict = {
            'critic': self._critic,
            'target_actor_params': self._target_actor_params,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'temp': self._temp,
            'expl_actor': self._expl_actor,
            'expl_critic': self._expl_critic,
            'expl_target_actor_params': self._expl_target_actor_params,
            'expl_target_critic_params': self._expl_target_critic_params,
            'expl_temp': self._expl_temp,
            'ens_state': self._ens_state,
            'step': self._step,
            'exploration_steps': self._exploration_steps,
        }
        return save_dict

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_actor_params = output_dict['target_actor_params']
        self._target_critic_params = output_dict['target_critic_params']
        self._temp = output_dict['temp']
        self._expl_actor = output_dict.get('expl_actor', self._expl_actor)
        self._expl_critic = output_dict.get('expl_critic', self._expl_critic)
        self._expl_target_actor_params = output_dict.get(
            'expl_target_actor_params', self._expl_target_actor_params)
        self._expl_target_critic_params = output_dict.get(
            'expl_target_critic_params', self._expl_target_critic_params)
        self._expl_temp = output_dict.get('expl_temp', self._expl_temp)
        self._ens_state = output_dict['ens_state']
        self._step = int(output_dict.get('step', self._step))
        self._exploration_steps = int(
            output_dict.get('exploration_steps', self._exploration_steps))
        print('restored from ', dir)
        


@functools.partial(jax.jit)
def get_value(action, observation, critic):
    input_collections = {'params': critic.params}
    if getattr(critic, 'batch_stats', None) is not None:
        input_collections['batch_stats'] = critic.batch_stats
    q_pred, _, _, _ = critic.apply_fn(input_collections, observation, action)
    return q_pred


def np_unstack(array, axis):
    arr = np.split(array, array.shape[axis], axis)
    arr = [a.squeeze() for a in arr]
    return arr

def make_visual(q_estimates, rewards, masks, images):

    q_estimates_np = np.stack(q_estimates, 0).squeeze()
    fig, axs = plt.subplots(4, 1, figsize=(8, 12))
    canvas = FigureCanvas(fig)
    plt.xlim([0, len(q_estimates_np)])

    assert len(images.shape) == 5
    images = images[..., -1]  # only taking the most recent image of the stack
    assert images.shape[-1] == 3

    interval = max(1, images.shape[0] // 4)
    sel_images = images[::interval]
    sel_images = np.concatenate(np_unstack(sel_images, 0), 1)

    axs[0].imshow(sel_images)
    if len(q_estimates_np.shape) == 2:
        for i in range(q_estimates_np.shape[1]):
            axs[1].plot(q_estimates_np[:, i], linestyle='--', marker='o')
    else:
        axs[1].plot(q_estimates_np, linestyle='--', marker='o')
    axs[1].set_ylabel('q values')
    axs[2].plot(rewards, linestyle='--', marker='o')
    axs[2].set_ylabel('rewards')
    axs[2].set_xlim([0, len(rewards)])
    
    axs[3].plot(masks, linestyle='--', marker='d')
    axs[3].set_ylabel('masks')
    axs[3].set_xlim([0, len(masks)])

    plt.tight_layout()

    canvas.draw()  # draw the canvas, cache the renderer
    out_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    out_image = out_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)
    return out_image


PixelMaxinfoSACLearner = PixelMaxinfoSACExplorer
PixelSACLearner = PixelMaxinfoSACExplorer
