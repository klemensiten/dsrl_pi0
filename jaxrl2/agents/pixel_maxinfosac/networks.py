from typing import Dict, Sequence, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init, xavier_init
from jaxrl2.networks.values import StateActionEnsemble


def _flatten(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.reshape(x, (x.shape[0], -1))


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

    def _encode_tactile(self,
                        tactile: jnp.ndarray,
                        training: bool = False) -> jnp.ndarray:
        tactile = tactile.astype(jnp.float32)
        touch = _touch_mask(tactile)

        # Simpler than in Maxinfodrqv2 since we don't replace the DSRL image encoder
        if tactile.ndim >= 4:
            assert len(self.tactile_cnn_features) == len(
                self.tactile_cnn_strides)
            x = jnp.reshape(tactile, (*tactile.shape[:-2], -1))
            for features, stride in zip(self.tactile_cnn_features,
                                        self.tactile_cnn_strides):
                x = nn.Conv(
                    features,
                    kernel_size=(3, 3),
                    strides=(stride, stride),
                    kernel_init=default_init(),
                    padding=self.tactile_cnn_padding)(x)
                x = nn.relu(x)
            x = _flatten(x)
        else:
            x = MLP(self.tactile_hidden_dims)(
                _flatten(tactile), training=training)

        tactile_embed = nn.Dense(
            self.obs_dim, kernel_init=xavier_init(), name='tactile_embed')(x)
        tactile_embed = nn.LayerNorm(name='tactile_embed_ln')(tactile_embed)
        tactile_embed = nn.tanh(tactile_embed)
        if self.mask_touch:
            tactile_embed = touch.astype(tactile_embed.dtype) * tactile_embed
        return tactile_embed

    @nn.compact
    def __call__(
            self,
            observations: Dict[str, jnp.ndarray],
            actions: jnp.ndarray,
            training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, FrozenDict]:
        observations = FrozenDict(observations)

        image_features = self.encoder(observations['pixels'], training)

        # compute image latent same as in PixelMultiplexer for the critic
        # TODO: also add tactile modalities?
        if self.use_bottleneck:
            latent = nn.Dense(
                self.latent_dim, kernel_init=xavier_init(), name='latent')(
                    image_features)
            latent = nn.LayerNorm(name='latent_ln')(latent)
            latent = nn.tanh(latent)
        else:
            latent = image_features

        # compact image target for the ensemble
        # TODO: this was max pooling for MaxinfoDrQ. Change?
        image_embed = nn.Dense(
            self.obs_dim, kernel_init=xavier_init(), name='image_embed')(
                image_features)
        image_embed = nn.LayerNorm(name='image_embed_ln')(image_embed)
        image_embed = nn.tanh(image_embed)

        modalities = {
            'latent': latent,
            'image': image_embed,
        }
        if self.model_obs_key in observations:
            state_embed = nn.Dense(
                self.obs_dim, kernel_init=xavier_init(), name='state_embed')(
                    _flatten(observations[self.model_obs_key]))
            state_embed = nn.LayerNorm(name='state_embed_ln')(state_embed)
            modalities['state'] = nn.tanh(state_embed)
        if 'tactile' in observations:
            # currently a placeholder
            modalities['tactile'] = self._encode_tactile(
                observations['tactile'], training=training)

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
