from tqdm import tqdm
import numpy as np
import wandb
import jax
from openpi_client import image_tools
import math
import PIL
from PIL import Image

DSRL_ACTION_MODES = ('noise', 'delta', 'both')
DEFAULT_PI0_ACTION_HORIZON = 50
DEFAULT_PI0_NOISE_DIM = 32
DEFAULT_RESIDUAL_DELTA_FRACTION = 0.1
DEFAULT_TACTILE_SHAPE = (32, 64, 3)
TACTILE_SHAPE_BY_GRIPPER = {
}

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def obs_to_img(obs, variant):
    '''
    Convert raw observation to resized image for DSRL actor/critic
    '''
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'aloha_cube':
        curr_image = obs["pixels"]["top"]
    else:
        raise NotImplementedError()
    if variant.resize_image > 0: 
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image

def obs_to_pi_zero_input(obs, variant):
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs_to_gripper_qpos(obs, variant),
                            )
                        ),
                        "prompt": str(variant.task_description),
                    }
    elif variant.env == 'aloha_cube':
        img = np.ascontiguousarray(obs["pixels"]["top"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        obs_pi_zero = {
            "state": obs["agent_pos"],
            "images": {"cam_high": np.transpose(img, (2,0,1))}
        }
    else:
        raise NotImplementedError()
    return obs_pi_zero

def obs_to_qpos(obs, variant):
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs_to_gripper_qpos(obs, variant),
            )
        )
    elif variant.env == 'aloha_cube':
        qpos = obs["agent_pos"]
    else:
        raise NotImplementedError()
    return qpos

def get_libero_state_dim(variant):
    return 6 + int(getattr(variant, 'gripper_state_dim', 2))

def _parse_int_sequence(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in ('none', 'auto'):
            return None
        value = value.replace(',', ' ').split()
    return tuple(int(v) for v in value)

def obs_to_gripper_qpos(obs, variant):
    qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    target_dim = int(getattr(variant, 'gripper_state_dim', 2))
    indices = _parse_int_sequence(getattr(variant, 'gripper_state_indices', None))

    if indices is None and qpos.size == 6 and target_dim == 2:
        # Robotiq85 exposes 6 mimic joints; use the two outer knuckle roots as
        # a compact left/right proxy so Pi0 still receives an 8D LIBERO state.
        indices = (0, 3)

    if indices is not None:
        selected = qpos[np.asarray(indices, dtype=np.int32)]
        if selected.size != target_dim:
            raise ValueError(
                f'gripper_state_indices selected {selected.size} values, '
                f'but gripper_state_dim is {target_dim}.')
        return selected

    if qpos.size == target_dim:
        return qpos
    if qpos.size > target_dim:
        raise ValueError(
            f'robot0_gripper_qpos has {qpos.size} values, but '
            f'gripper_state_dim is {target_dim}. Set '
            f'--gripper_state_indices to choose the values fed to Pi0/DSRL.')

    return np.pad(qpos, (0, target_dim - qpos.size))

def get_tactile_shape(variant):
    tactile_shape = getattr(variant, 'tactile_shape', DEFAULT_TACTILE_SHAPE)
    if isinstance(tactile_shape, str):
        tactile_shape = tuple(
            int(dim) for dim in tactile_shape.replace(',', ' ').split())
    else:
        tactile_shape = tuple(int(dim) for dim in tactile_shape)
    touch_gripper_type = getattr(variant, 'touch_gripper_type', None)
    if (
            tactile_shape == DEFAULT_TACTILE_SHAPE
            and touch_gripper_type in TACTILE_SHAPE_BY_GRIPPER):
        tactile_shape = TACTILE_SHAPE_BY_GRIPPER[touch_gripper_type]
    if len(tactile_shape) != 3:
        raise ValueError(
            f'tactile_shape must be three image-style dims (H, W, C); '
            f'got {tactile_shape}')
    if tactile_shape[-1] != 3:
        raise ValueError(
            f'tactile_shape channel dim must be 3; got {tactile_shape}')
    return tactile_shape

def obs_to_tactile(obs, variant):
    if variant.env != 'libero':
        raise NotImplementedError('Tactile observations are only wired for LIBERO.')
    if 'tactile' in obs:
        tactile = obs['tactile']
    elif 'tactile_left' in obs and 'tactile_right' in obs:
        tactile = np.concatenate(
            (obs['tactile_left'], obs['tactile_right']),
            axis=-1,
        )
    else:
        raise KeyError('LIBERO observation does not contain tactile data.')

    tactile = np.asarray(tactile, dtype=np.float32)
    if tactile.ndim == 3 and tactile.shape[0] == 3:
        tactile = np.moveaxis(tactile, 0, -1)
    expected_shape = get_tactile_shape(variant)
    if tactile.shape != expected_shape:
        raise ValueError(
            f'Unexpected tactile observation shape: {tactile.shape}; '
            f'expected {expected_shape}. Set --tactile_shape H W C to match '
            f'the active gripper.')
    return tactile

def tactile_to_heatmap(tactile, max_value, output_size):
    tactile = np.asarray(tactile, dtype=np.float32)
    values = np.linalg.norm(tactile, axis=-1)
    scale = max(float(max_value), 1e-8)
    values = np.clip(values / scale, 0.0, 1.0)

    heatmap = np.zeros((*values.shape, 3), dtype=np.float32)
    heatmap[..., 0] = np.clip(3.0 * values - 1.5, 0.0, 1.0)
    heatmap[..., 1] = np.clip(3.0 * values - 0.5, 0.0, 1.0)
    heatmap[..., 2] = np.clip(4.0 * values, 0.0, 1.0) * (1.0 - np.clip(2.0 * values - 1.0, 0.0, 1.0))
    heatmap = (255.0 * heatmap).astype(np.uint8)

    if heatmap.shape[1] >= 2:
        split = heatmap.shape[1] // 2
        heatmap[:, split - 1:split + 1] = 255

    return np.asarray(
        Image.fromarray(heatmap).resize(output_size, Image.Resampling.NEAREST),
        dtype=np.uint8,
    )

def tactile_video_from_list(tactile_list, output_size):
    if not tactile_list:
        return None
    max_value = max(
        float(np.linalg.norm(np.asarray(tactile), axis=-1).max())
        for tactile in tactile_list
    )
    frames = [
        tactile_to_heatmap(tactile, max_value, output_size)
        for tactile in tactile_list
    ]
    return np.stack(frames).transpose(0, 3, 1, 2)

def obs_to_agent_input(obs, variant, curr_image=None):
    if curr_image is None:
        curr_image = obs_to_img(obs, variant)
    obs_dict = {
        'pixels': curr_image[np.newaxis, ..., np.newaxis],
    }
    if variant.add_states:
        qpos = obs_to_qpos(obs, variant)
        obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
    if getattr(variant, 'add_tactile', 0):
        tactile = obs_to_tactile(obs, variant)
        obs_dict['tactile'] = tactile[np.newaxis, ..., np.newaxis]
    return obs_dict

def get_dsrl_action_mode(variant):
    mode = str(getattr(variant, 'dsrl_action_mode', 'noise')).strip().lower()
    if mode not in DSRL_ACTION_MODES:
        raise ValueError(
            f"Unsupported dsrl_action_mode '{mode}'. "
            f"Expected one of {DSRL_ACTION_MODES}.")
    return mode

def get_pi0_action_horizon(agent_dp=None):
    return int(getattr(agent_dp, 'action_horizon', DEFAULT_PI0_ACTION_HORIZON))

def get_pi0_noise_dim(agent_dp=None):
    return int(getattr(agent_dp, 'action_dim', DEFAULT_PI0_NOISE_DIM))

def get_default_actual_action_dim(variant):
    if variant.env == 'libero':
        return 7
    if variant.env == 'aloha_cube':
        return 14
    raise NotImplementedError()

def get_actual_action_dim(variant):
    action_dim = getattr(variant, 'actual_action_dim', None)
    if action_dim is None:
        return get_default_actual_action_dim(variant)
    return int(action_dim)

def _action_bounds_candidates(env):
    action_space = getattr(env, 'action_space', None)
    if action_space is not None and hasattr(action_space, 'low') and hasattr(action_space, 'high'):
        yield action_space.low, action_space.high

    inner_env = getattr(env, 'env', None)
    if inner_env is not None and hasattr(inner_env, 'action_spec'):
        try:
            yield inner_env.action_spec
        except Exception:
            pass

    if hasattr(env, 'action_spec'):
        try:
            yield env.action_spec
        except Exception:
            pass

def infer_actual_action_dim(variant, env=None):
    default_dim = get_default_actual_action_dim(variant)
    if env is None:
        return default_dim

    for low, high in _action_bounds_candidates(env):
        low = np.asarray(low).reshape(-1)
        high = np.asarray(high).reshape(-1)
        if low.size == high.size == default_dim:
            return default_dim
    return default_dim

def get_env_action_bounds(env, actual_action_dim):
    for low, high in _action_bounds_candidates(env):
        low = np.asarray(low, dtype=np.float32).reshape(-1)
        high = np.asarray(high, dtype=np.float32).reshape(-1)
        if low.size == high.size == actual_action_dim:
            return low, high

    low = np.full((actual_action_dim,), np.nan, dtype=np.float32)
    high = np.full((actual_action_dim,), np.nan, dtype=np.float32)
    return low, high

def action_bounds_to_config_list(bounds):
    bounds = np.asarray(bounds, dtype=np.float32).reshape(-1)
    return [
        float(bound) if np.isfinite(bound) else None
        for bound in bounds
    ]

def compute_residual_delta_bounds(action_low, action_high, residual_delta_fraction):
    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)
    residual_delta_fraction = float(residual_delta_fraction)
    fallback = np.full_like(action_low, residual_delta_fraction, dtype=np.float32)
    finite = np.isfinite(action_low) & np.isfinite(action_high)
    positive_range = action_high > action_low
    scaled = residual_delta_fraction * (action_high - action_low)
    return np.where(finite & positive_range, scaled, fallback).astype(np.float32)

def get_residual_delta_bounds(variant, actual_action_dim=None):
    residual_delta_bounds = getattr(variant, 'residual_delta_bounds', None)
    if residual_delta_bounds is not None:
        return np.asarray(residual_delta_bounds, dtype=np.float32)

    if actual_action_dim is None:
        actual_action_dim = get_actual_action_dim(variant)
    fallback = float(getattr(
        variant,
        'residual_delta_fraction',
        DEFAULT_RESIDUAL_DELTA_FRACTION,
    ))
    return np.full((actual_action_dim,), fallback, dtype=np.float32)

def get_dsrl_action_width(variant, pi0_noise_dim=DEFAULT_PI0_NOISE_DIM,
                          actual_action_dim=None):
    mode = get_dsrl_action_mode(variant)
    if actual_action_dim is None:
        actual_action_dim = get_actual_action_dim(variant)
    if mode == 'noise':
        return int(pi0_noise_dim)
    if mode == 'delta':
        return int(actual_action_dim)
    if mode == 'both':
        return int(pi0_noise_dim) + int(actual_action_dim)
    raise AssertionError(f'Unhandled DSRL action mode: {mode}')

def get_dsrl_action_shape(variant, pi0_noise_dim=DEFAULT_PI0_NOISE_DIM,
                          actual_action_dim=None):
    return (1, get_dsrl_action_width(
        variant,
        pi0_noise_dim=pi0_noise_dim,
        actual_action_dim=actual_action_dim,
    ))

def split_dsrl_action(variant, learner_action, pi0_noise_dim=DEFAULT_PI0_NOISE_DIM,
                      actual_action_dim=None):
    mode = get_dsrl_action_mode(variant)
    if actual_action_dim is None:
        actual_action_dim = get_actual_action_dim(variant)

    expected_width = get_dsrl_action_width(
        variant,
        pi0_noise_dim=pi0_noise_dim,
        actual_action_dim=actual_action_dim,
    )
    learner_action = np.asarray(learner_action, dtype=np.float32).reshape(1, expected_width)

    if mode == 'noise':
        return learner_action, None
    if mode == 'delta':
        return None, learner_action
    if mode == 'both':
        return (
            learner_action[:, :pi0_noise_dim],
            learner_action[:, pi0_noise_dim:],
        )
    raise AssertionError(f'Unhandled DSRL action mode: {mode}')

def _compose_dsrl_action(variant, noise_chunk, delta_normalized,
                         pi0_noise_dim, actual_action_dim):
    mode = get_dsrl_action_mode(variant)
    if mode == 'noise':
        return np.asarray(noise_chunk, dtype=np.float32).reshape(1, pi0_noise_dim)
    if mode == 'delta':
        return np.asarray(delta_normalized, dtype=np.float32).reshape(1, actual_action_dim)
    if mode == 'both':
        noise_chunk = np.asarray(noise_chunk, dtype=np.float32).reshape(1, pi0_noise_dim)
        delta_normalized = np.asarray(delta_normalized, dtype=np.float32).reshape(1, actual_action_dim)
        return np.concatenate([noise_chunk, delta_normalized], axis=-1)
    raise AssertionError(f'Unhandled DSRL action mode: {mode}')

def _noise_chunk_to_horizon(noise_chunk, action_horizon):
    noise_chunk = jax.numpy.asarray(noise_chunk)
    if noise_chunk.shape[0] >= action_horizon:
        return noise_chunk[:action_horizon][None]

    repeat = jax.numpy.repeat(
        noise_chunk[-1:, :],
        action_horizon - noise_chunk.shape[0],
        axis=0,
    )
    return jax.numpy.concatenate([noise_chunk, repeat], axis=0)[None]

def make_dsrl_components(variant, learner_action, key, action_chunk_shape,
                         action_horizon, pi0_noise_dim, actual_action_dim,
                         random_delta=False, zero_delta=False,
                         full_horizon_noise=False):
    mode = get_dsrl_action_mode(variant)
    noise_key, delta_key = jax.random.split(key)
    noise_chunk = None
    full_noise = None
    delta_normalized = None

    if learner_action is None:
        if mode in ('noise', 'both'):
            if full_horizon_noise:
                full_noise = jax.random.normal(
                    noise_key,
                    (1, action_horizon, pi0_noise_dim),
                )
                noise_chunk = full_noise[0, :action_chunk_shape[0], :]
            else:
                noise_chunk = jax.random.normal(noise_key, (action_chunk_shape[0], pi0_noise_dim))
        if mode in ('delta', 'both'):
            if zero_delta:
                delta_normalized = np.zeros((action_chunk_shape[0], actual_action_dim), dtype=np.float32)
            elif random_delta:
                delta_normalized = jax.random.uniform(
                    delta_key,
                    (action_chunk_shape[0], actual_action_dim),
                    minval=-1.0,
                    maxval=1.0,
                )
            else:
                delta_normalized = np.zeros((action_chunk_shape[0], actual_action_dim), dtype=np.float32)
        learner_action = _compose_dsrl_action(
            variant,
            noise_chunk,
            delta_normalized,
            pi0_noise_dim,
            actual_action_dim,
        )
    else:
        learner_action = np.reshape(learner_action, action_chunk_shape)
        noise_chunk, delta_normalized = split_dsrl_action(
            variant,
            learner_action,
            pi0_noise_dim=pi0_noise_dim,
            actual_action_dim=actual_action_dim,
        )

    if full_noise is not None:
        noise = full_noise
    elif noise_chunk is None:
        noise = jax.random.normal(noise_key, (1, action_horizon, pi0_noise_dim))
    else:
        noise = _noise_chunk_to_horizon(noise_chunk, action_horizon)

    return np.asarray(learner_action, dtype=np.float32), noise, delta_normalized

def apply_residual_delta(variant, actions, delta_normalized):
    actions = np.asarray(actions, dtype=np.float32)
    action_dim = actions.shape[-1]

    if delta_normalized is not None:
        delta_normalized = np.asarray(delta_normalized, dtype=np.float32)
        delta_normalized = np.clip(delta_normalized, -1.0, 1.0)
        residual_delta_bounds = get_residual_delta_bounds(
            variant,
            actual_action_dim=action_dim,
        )
        delta = delta_normalized * residual_delta_bounds
        actions = actions + delta

    action_low = getattr(variant, 'actual_action_low', None)
    action_high = getattr(variant, 'actual_action_high', None)
    if action_low is None or action_high is None:
        return actions

    action_low = np.asarray(action_low, dtype=np.float32)
    action_high = np.asarray(action_high, dtype=np.float32)
    finite = np.isfinite(action_low) & np.isfinite(action_high)
    if not np.any(finite):
        return actions

    clipped_actions = np.array(actions, copy=True)
    clipped_actions[..., finite] = np.clip(
        clipped_actions[..., finite],
        action_low[finite],
        action_high[finite],
    )
    return clipped_actions

def trajwise_alternating_training_loop(variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
                                       perform_control_evals=True, shard_fn=None, agent_dp=None):
    replay_buffer_iterator = replay_buffer.get_iterator(variant.batch_size)
    if shard_fn is not None:
        replay_buffer_iterator = map(shard_fn, replay_buffer_iterator)

    total_env_steps = 0
    i = 0
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)
    
    with tqdm(total=variant.max_steps, initial=0) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj(variant, agent, env, i, agent_dp)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer(variant, traj, online_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            print('total env steps:', total_env_steps)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"])*variant.multi_grad_step

            if len(online_replay_buffer) > variant.start_online_updates:
                for _ in range(num_gradsteps):
                    # perform first visualization before updating
                    if i == 0:
                        print('performing evaluation for initial checkpoint')
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    # online perform update once we have some amount of online trajs
                    batch = next(replay_buffer_iterator)
                    update_info = agent.update(batch)

                    pbar.update()
                    i += 1
                        

                    if i % variant.log_interval == 0:
                        update_info = {k: jax.device_get(v) for k, v in update_info.items()}
                        for k, v in update_info.items():
                            if v.ndim == 0:
                                wandb_logger.log({f'training/{k}': v}, step=i)
                            elif v.ndim <= 2:
                                wandb_logger.log_histogram(f'training/{k}', v, i)
                        # wandb_logger.log({'replay_buffer_size': len(online_replay_buffer)}, i)
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval(agent, eval_env, i, variant, wandb_logger, agent_dp)
                        if hasattr(agent, 'perform_eval'):
                            agent.perform_eval(variant, i, wandb_logger, replay_buffer, replay_buffer_iterator, eval_env)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        agent.save_checkpoint(variant.outputdir, i, variant.checkpoint_interval)

            
def add_online_data_to_buffer(variant, traj, online_replay_buffer):

    discount_horizon = variant.query_freq
    actions = np.array(traj['actions']) # (T, chunk_size, action_dim )
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        # remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon
        )
        online_replay_buffer.insert(insert_dict)
    online_replay_buffer.increment_traj_counter()

def collect_traj(variant, agent, env, i, agent_dp=None):
    query_frequency = variant.query_freq
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    action_horizon = get_pi0_action_horizon(agent_dp)
    pi0_noise_dim = get_pi0_noise_dim(agent_dp)
    actual_action_dim = get_actual_action_dim(variant)

    agent._rng, rng = jax.random.split(agent._rng)
    
    if 'libero' in variant.env:
        obs = env.reset()
    elif 'aloha' in variant.env:
        obs, _ = env.reset()
    
    image_list = [] # for visualization
    rewards = []
    action_list = []
    obs_list = []

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        obs_dict = obs_to_agent_input(obs, variant, curr_image=curr_image)

        if t % query_frequency == 0:

            assert agent_dp is not None
            # we then use the noise to sample the action from diffusion model
            rng, key = jax.random.split(rng)
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            if i == 0:
                # For initial collection, explore with random pi0 noise and
                # random residuals when residual modes are enabled.
                learner_action, noise, delta_normalized = make_dsrl_components(
                    variant,
                    None,
                    key,
                    agent.action_chunk_shape,
                    action_horizon,
                    pi0_noise_dim,
                    actual_action_dim,
                    random_delta=True,
                )
            else:
                # SAC predicts the configured DSRL action: noise, residual,
                # or the concatenation of both.
                learner_action = agent.sample_actions(obs_dict)
                learner_action, noise, delta_normalized = make_dsrl_components(
                    variant,
                    learner_action,
                    key,
                    agent.action_chunk_shape,
                    action_horizon,
                    pi0_noise_dim,
                    actual_action_dim,
                )
            
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
            actions = apply_residual_delta(variant, actions, delta_normalized)
            action_list.append(learner_action)
            obs_list.append(obs_dict)
     
        action_t = actions[t % query_frequency]
        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t)
        elif 'aloha' in variant.env:
            obs, reward, terminated, truncated, _ = env.step(action_t)
            done = terminated or truncated
            
        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # add last observation
    curr_image = obs_to_img(obs, variant)
    obs_dict = obs_to_agent_input(obs, variant, curr_image=curr_image)
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # per episode
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards!=None])
    is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    
    '''
    We use sparse -1/0 reward to train the SAC agent.
    '''
    if is_success:
        query_steps = len(action_list)
        rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
        masks = np.concatenate([np.ones(query_steps - 1), [0]])
    else:
        query_steps = len(action_list)
        rewards = -np.ones(query_steps)
        masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1 
    }

def perform_control_eval(agent, env, i, variant, wandb_logger, agent_dp=None):
    query_frequency = variant.query_freq
    print('query frequency', query_frequency)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    action_horizon = get_pi0_action_horizon(agent_dp)
    pi0_noise_dim = get_pi0_noise_dim(agent_dp)
    actual_action_dim = get_actual_action_dim(variant)
    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    rng = jax.random.PRNGKey(variant.seed+456)

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
        elif 'aloha' in variant.env:
            obs, _ = env.reset()
            
        image_list = [] # for visualization
        log_eval_tactile = 'libero' in variant.env and bool(
            getattr(variant, 'use_touch', 0))
        tactile_list = [] if log_eval_tactile else None
        rewards = []
        

        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)
            curr_tactile = obs_to_tactile(obs, variant) if log_eval_tactile else None

            if t % query_frequency == 0:
                obs_dict = obs_to_agent_input(obs, variant, curr_image=curr_image)

                rng, key = jax.random.split(rng)
                assert agent_dp is not None
                
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                
                
                if i == 0:
                    # Initial evaluation keeps residuals at zero so it
                    # evaluates the base pi0 policy cleanly.
                    _, noise, delta_normalized = make_dsrl_components(
                        variant,
                        None,
                        key,
                        agent.action_chunk_shape,
                        action_horizon,
                        pi0_noise_dim,
                        actual_action_dim,
                        zero_delta=True,
                        full_horizon_noise=True,
                    )
                else:
                    if hasattr(agent, 'collecting_exploration'):
                        learner_action = agent.eval_actions(obs_dict)
                    else:
                        learner_action = agent.sample_actions(obs_dict)
                    _, noise, delta_normalized = make_dsrl_components(
                        variant,
                        learner_action,
                        key,
                        agent.action_chunk_shape,
                        action_horizon,
                        pi0_noise_dim,
                        actual_action_dim,
                    )
                    
                actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]
                actions = apply_residual_delta(variant, actions, delta_normalized)
              
            action_t = actions[t % query_frequency]
            
            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t)
            elif 'aloha' in variant.env:
                obs, reward, terminated, truncated, _ = env.step(action_t)
                done = terminated or truncated
                
            rewards.append(reward)
            image_list.append(curr_image)
            if log_eval_tactile:
                tactile_list.append(curr_tactile)
            if done:
                break

        # per episode
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        is_success = (reward == env_max_reward)
        success_rates.append(is_success)
                
        print(f'Rollout {rollout_id} : {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        wandb_logger.log({f'eval_video/{rollout_id}': wandb.Video(video, fps=50)}, step=i)
        if log_eval_tactile:
            height, width = image_list[0].shape[:2]
            tactile_video = tactile_video_from_list(tactile_list, (width, height))
            wandb_logger.log({
                f'eval_tactile_video/{rollout_id}': wandb.Video(
                    tactile_video, fps=50)
            }, step=i)


    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

def make_multiple_value_reward_visulizations(agent, variant, i, replay_buffer, wandb_logger):
    trajs = replay_buffer.get_random_trajs(3)
    images = agent.make_value_reward_visulization(variant, trajs)
    wandb_logger.log({'reward_value_images': wandb.Image(images)}, step=i)
  
