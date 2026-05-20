from typing import Dict, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init, xavier_init
from jaxrl2.networks.values import StateActionEnsemble


def _flatten(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(x, (x.shape[0], -1))


def _max_pool_to_dim(x: jnp.ndarray, output_dim: int) -> jnp.ndarray:
    x = _flatten(x)
    feature_dim = x.shape[-1]
    if feature_dim == output_dim:
        return x
    if feature_dim < output_dim:
        return jnp.pad(x, ((0, 0), (0, output_dim - feature_dim)))

    pool_size = (feature_dim + output_dim - 1) // output_dim
    padded_dim = pool_size * output_dim
    pad = padded_dim - feature_dim
    if pad:
        x = jnp.pad(x, ((0, 0), (0, pad)), constant_values=-jnp.inf)
    x = jnp.reshape(x, (x.shape[0], output_dim, pool_size))
    return jnp.max(x, axis=-1)


def _touch_mask(x: jnp.ndarray) -> jnp.ndarray:
    axes = tuple(range(1, x.ndim))
    return (jnp.max(jnp.abs(x), axis=axes) > 0)[..., jnp.newaxis]


class PixelMaxInfoCritic(nn.Module):
    encoder: nn.Module
    hidden_dims: Sequence[int]
    latent_dim: int
    obs_dim: int = 64
    num_qs: int = 2
    model_obs_key: str = "state"
    use_bottleneck: bool = True
    tactile_hidden_dims: Sequence[int] = (256, 256)
    tactile_cnn_features: Sequence[int] = (32, 32, 32, 32)
    tactile_cnn_strides: Sequence[int] = (2, 1, 1, 1)
    tactile_cnn_padding: str = "VALID"
    mask_touch: bool = False

    @nn.compact
    def __call__(
            self,
            observations: Dict[str, jnp.ndarray],
            actions: jnp.ndarray,
            training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, FrozenDict]:
        observations = FrozenDict(observations)

        image_features = self.encoder(observations['pixels'], training)

        if self.use_bottleneck:
            latent = nn.Dense(
                self.latent_dim, kernel_init=xavier_init(), name='latent')(
                    image_features)
            latent = nn.LayerNorm(name='latent_ln')(latent)
            latent = nn.tanh(latent)
        else:
            latent = image_features

        image_embed = _max_pool_to_dim(image_features, self.obs_dim)

        modalities = {
            'latent': latent,
            'image': image_embed,
        }
        if self.model_obs_key in observations:
            modalities['state'] = _flatten(observations[self.model_obs_key])
        if 'tactile' in observations:
            modalities['tactile'] = TactileEncoder(
                obs_dim=self.obs_dim,
                hidden_dims=self.tactile_hidden_dims,
                cnn_features=self.tactile_cnn_features,
                cnn_strides=self.tactile_cnn_strides,
                cnn_padding=self.tactile_cnn_padding,
                mask_touch=self.mask_touch,
                name='tactile_encoder',
            )(observations['tactile'], training=training)

        critic_replacements = {'pixels': latent}
        if 'tactile' in modalities:
            critic_replacements['tactile'] = modalities['tactile']
        critic_observations = observations.copy(
            add_or_replace=critic_replacements)
        critic = StateActionEnsemble(
            self.hidden_dims, num_qs=self.num_qs, name='critic')(
                critic_observations, actions, training=training)
        expl_critic = StateActionEnsemble(
            self.hidden_dims, num_qs=self.num_qs, name='expl_critic')(
                critic_observations, actions, training=training)
        return critic, expl_critic, latent, FrozenDict(modalities)


class TactileEncoder(nn.Module):
    obs_dim: int = 64
    hidden_dims: Sequence[int] = (256, 256)
    cnn_features: Sequence[int] = (32, 32, 32, 32)
    cnn_strides: Sequence[int] = (2, 1, 1, 1)
    cnn_padding: str = "VALID"
    mask_touch: bool = False

    @nn.compact
    def __call__(self,
                 tactile: jnp.ndarray,
                 training: bool = False) -> jnp.ndarray:
        tactile = tactile.astype(jnp.float32)
        touch = _touch_mask(tactile)

        if tactile.ndim == 5:
            x = jnp.reshape(tactile, (*tactile.shape[:-2], -1))
        elif tactile.ndim == 4 and tactile.shape[1] == 3 and tactile.shape[-1] != 3:
            x = jnp.moveaxis(tactile, 1, -1)
        else:
            x = tactile

        if x.ndim == 4:
            assert len(self.cnn_features) == len(self.cnn_strides)
            for features, stride in zip(self.cnn_features, self.cnn_strides):
                x = nn.Conv(
                    features,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    kernel_init=default_init(),
                    padding=self.cnn_padding)(x)
                x = nn.relu(x)
            x = _flatten(x)
        else:
            x = MLP(self.hidden_dims)(_flatten(tactile), training=training)

        tactile_embed = nn.Dense(
            self.obs_dim, kernel_init=xavier_init(), name='tactile_embed')(x)
        tactile_embed = nn.LayerNorm(name='tactile_embed_ln')(tactile_embed)
        tactile_embed = nn.tanh(tactile_embed)
        if self.mask_touch:
            tactile_embed = touch.astype(tactile_embed.dtype) * tactile_embed
        return tactile_embed


class PixelTactileMultiplexer(nn.Module):
    encoder: nn.Module
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool = True
    obs_dim: int = 64
    tactile_hidden_dims: Sequence[int] = (256, 256)
    tactile_cnn_features: Sequence[int] = (32, 32, 32, 32)
    tactile_cnn_strides: Sequence[int] = (2, 1, 1, 1)
    tactile_cnn_padding: str = "VALID"
    mask_touch: bool = False

    @nn.compact
    def __call__(
            self,
            observations: Dict[str, jnp.ndarray],
            actions: jnp.ndarray = None,
            training: bool = False
    ):
        observations = FrozenDict(observations)

        x = self.encoder(observations['pixels'], training)
        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        replacements = {'pixels': x}
        if 'tactile' in observations:
            replacements['tactile'] = TactileEncoder(
                obs_dim=self.obs_dim,
                hidden_dims=self.tactile_hidden_dims,
                cnn_features=self.tactile_cnn_features,
                cnn_strides=self.tactile_cnn_strides,
                cnn_padding=self.tactile_cnn_padding,
                mask_touch=self.mask_touch,
                name='tactile_encoder',
            )(observations['tactile'], training=training)

        encoded_observations = observations.copy(add_or_replace=replacements)
        if actions is None:
            return self.network(encoded_observations, training=training)
        return self.network(encoded_observations, actions, training=training)
