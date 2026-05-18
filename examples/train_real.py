#! /usr/bin/env python
import inspect
import os
import jax
from jaxrl2.agents.pixel_maxinfosac.pixel_maxinfosac_learner import PixelMaxinfoSACLearner
from jaxrl2.agents.pixel_maxinfosac_explorer.pixel_maxinfosac_explorer import PixelMaxinfoSACExplorer
from jaxrl2.agents.pixel_sac.pixel_sac_learner import PixelSACLearner
from jaxrl2.utils.general_utils import add_batch_dim
import numpy as np
import logging

import gymnasium as gym
from gym.spaces import Dict, Box

from jaxrl2.data import ReplayBuffer
from jaxrl2.utils.wandb_logger import WandBLogger, create_exp_name
import tempfile
from functools import partial
from examples.train_utils_real import trajwise_alternating_training_loop
import tensorflow as tf
from jax.experimental.compilation_cache import compilation_cache
from openpi_client import websocket_client_policy as _websocket_client_policy
from droid.robot_env import RobotEnv

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


LEARNER_BY_ALGORITHM = {
    'pixel_sac': PixelSACLearner,
    'pixel_maxinfosac': PixelMaxinfoSACLearner,
    'pixel_maxinfosac_explorer': PixelMaxinfoSACExplorer,
}


def _normalize_ensemble_disagreement_modalities(kwargs):
    modalities = kwargs.get('ensemble_disagreement_modalities')
    if not isinstance(modalities, str):
        return

    modalities = modalities.strip()
    if not modalities:
        kwargs['ensemble_disagreement_modalities'] = None
        return

    kwargs['ensemble_disagreement_modalities'] = tuple(
        modality.strip()
        for modality in modalities.split(',')
        if modality.strip())


def _make_agent(variant, sample_obs, sample_action):
    algorithm = str(variant.algorithm).strip().lower()
    if algorithm not in LEARNER_BY_ALGORITHM:
        supported = ', '.join(sorted(LEARNER_BY_ALGORITHM))
        raise ValueError(
            f"Unsupported algorithm '{variant.algorithm}'. "
            f"Supported algorithms: {supported}.")

    learner_cls = LEARNER_BY_ALGORITHM[algorithm]
    kwargs = dict(variant['train_kwargs'])
    _normalize_ensemble_disagreement_modalities(kwargs)

    valid_kwargs = set(inspect.signature(learner_cls.__init__).parameters)
    valid_kwargs.discard('self')
    learner_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in valid_kwargs
    }
    dropped_kwargs = sorted(set(kwargs) - set(learner_kwargs))
    if dropped_kwargs:
        print(f"Dropping unsupported {algorithm} train kwargs: {dropped_kwargs}")

    print(f"Using algorithm: {algorithm}")
    return learner_cls(variant.seed, sample_obs, sample_action,
                       **learner_kwargs)


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension.

    Args:
        batch: A pytree of arrays.
        sharding: A jax Sharding object with shape (num_devices,).
    """
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )

class DummyEnv(gym.ObservationWrapper):

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        obs_dict = {}
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if variant.add_states:
            state_dim = 8 + 2024 # 8 is the proprioceptive state's dim, 2024 is the image representation's dim
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        self.observation_space = Dict(obs_dict)
        self.action_space = Box(low=-1, high=1, shape=(1, 32,), dtype=np.float32) # 32 is the noise action space of pi 0

def main(variant):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert variant.batch_size % num_devices == 0
    logging.info('num devices', num_devices)
    logging.info('batch size', variant.batch_size)
    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")
    
    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps
        
    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)
   
    outputdir = os.path.join(os.environ['EXP'], expname)
    variant.outputdir = outputdir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)        

    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp()
    wandb_logger = WandBLogger(variant.prefix != '', variant, variant.wandb_project, experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name)

    
    agent_dp = _websocket_client_policy.WebsocketClientPolicy(
        host=os.environ['remote_host'],
        port=os.environ['remote_port']
    )
    logging.info(f"Server metadata: {agent_dp.get_server_metadata()}")
    
    logging.info("initializing environment...")
    env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
    eval_env = env
    logging.info("created the droid env!")
    
    assert os.environ.get('LEFT_CAMERA_ID') is not None
    assert os.environ.get('RIGHT_CAMERA_ID') is not None
    assert os.environ.get('WRIST_CAMERA_ID') is not None
    
    robot_config = dict(
        camera_to_use='right',
        left_camera_id=os.environ['LEFT_CAMERA_ID'],
        right_camera_id=os.environ['RIGHT_CAMERA_ID'],
        wrist_camera_id=os.environ['WRIST_CAMERA_ID'],
        max_timesteps=200
    )
    
    dummy_env = DummyEnv(variant)
    sample_obs = add_batch_dim(dummy_env.observation_space.sample())
    sample_action = add_batch_dim(dummy_env.action_space.sample())
    logging.info('sample obs shapes', [(k, v.shape) for k, v in sample_obs.items()])
    logging.info('sample action shape', sample_action.shape)
    
    agent = _make_agent(variant, sample_obs, sample_action)
    
    if variant.restore_path != '':
        logging.info('restoring from', variant.restore_path)
        agent.restore_checkpoint(variant.restore_path)

    online_buffer_size = 2 * variant.max_steps  // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp, robot_config=robot_config)
