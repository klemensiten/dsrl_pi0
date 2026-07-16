#!/usr/bin/env python
"""Render supplement-style LIBERO rollouts from saved DSRL checkpoints."""
import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


GENERAL_DEFAULTS = {
    "seed": 42,
    "launch_group_id": "",
    "eval_episodes": 10,
    "env": "libero",
    "log_interval": 1000,
    "eval_interval": 5000,
    "checkpoint_interval": -1,
    "batch_size": 16,
    "max_steps": int(1e6),
    "add_states": 1,
    "add_tactile": 0,
    "use_touch": 0,
    "touch_gripper_type": "Robotiq85TactileGripper",
    "tactile_shape": (32, 64, 3),
    "gripper_state_dim": 2,
    "gripper_state_indices": None,
    "wandb_project": "cql_sim_online",
    "start_online_updates": 1000,
    "algorithm": "pixel_sac",
    "prefix": "",
    "suffix": "",
    "multi_grad_step": 1,
    "resize_image": -1,
    "query_freq": -1,
    "libero_suite": "libero_90",
    "libero_task_id": 57,
    "libero_robot": "Panda",
}


TRAIN_KWARGS_DEFAULTS = {
    "actor_lr": 1e-4,
    "critic_lr": 3e-4,
    "temp_lr": 3e-4,
    "dyn_ent_lr": 3e-4,
    "init_dyn_ent_temperature": 1.0,
    "model_lr": 3e-4,
    "model_wd": 0.0,
    "model_hidden_dims": (256, 256),
    "num_model_heads": 5,
    "model_noise_var": 1.0,
    "predict_reward": True,
    "predict_diff": True,
    "backup_entropy": True,
    "model_obs_key": "state",
    "obs_dim": 64,
    "ensemble_disagreement_modalities": "",
    "mask_expl_critic": False,
    "tactile_hidden_dims": (256, 256),
    "mask_touch": False,
    "hidden_dims": (128, 128, 128),
    "cnn_features": (32, 32, 32, 32),
    "cnn_strides": (2, 1, 1, 1),
    "cnn_padding": "VALID",
    "latent_dim": 50,
    "discount": 0.999,
    "tau": 0.005,
    "critic_reduction": "mean",
    "dropout_rate": 0.0,
    "aug_next": 1,
    "use_bottleneck": True,
    "encoder_type": "small",
    "encoder_norm": "group",
    "use_spatial_softmax": True,
    "softmax_temperature": -1,
    "target_entropy": "auto",
    "num_qs": 10,
    "action_magnitude": 1.0,
    "num_cameras": 1,
    "explore_until": 300000,
    "agent_update_period": 1,
    "expl_agent_update_period": 1,
    "ensemble_update_period": 1,
}


PRESET_CHOICES = ("sac", "ablate", "explorer")
FALLBACK_PI0_ACTION_HORIZON = 50
FALLBACK_PI0_NOISE_DIM = 32
RUN_DIR_RE = re.compile(
    r"^(?P<prefix>.+)_"
    r"\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_"
    r"\d{4}--s-(?P<seed>\d+)"
    r"(?:_(?P<suffix>.+))?$"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--render_resolution", type=int, default=512)
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=480)
    parser.add_argument("--side_panel_width", type=int, default=None)
    parser.add_argument("--episode_steps", type=int, default=400)
    parser.add_argument(
        "--init_state_id",
        type=int,
        default=0,
        help="LIBERO fixed init-state index used when --use_libero_init_state is set.",
    )
    parser.add_argument(
        "--settle_steps",
        type=int,
        default=5,
        help="Zero-action settling steps after applying a fixed LIBERO init state.",
    )
    parser.add_argument(
        "--use_libero_init_state",
        action="store_true",
        help=(
            "Opt into LIBERO fixed init states. By default the renderer matches "
            "training eval and uses raw env.reset()."
        ),
    )
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--include_wrist", action="store_true")
    parser.add_argument("--no_tactile_panel", action="store_true")
    parser.add_argument(
        "--deterministic_dsrl",
        action="store_true",
        help=(
            "Override training eval action selection and force agent.eval_actions(). "
            "Training eval samples SAC actions unless the agent has collecting_exploration."
        ),
    )
    parser.add_argument(
        "--collection_dsrl",
        action="store_true",
        help=(
            "Render collection behavior with agent.sample_actions(). For explorer "
            "checkpoints this uses the exploration actor while step <= explore_until."
        ),
    )
    parser.add_argument(
        "--policy_seed",
        type=int,
        default=None,
        help=(
            "Render-only override for the DSRL sampling RNG. Use different "
            "values to sample different stochastic actions from one checkpoint."
        ),
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--mujoco_gl",
        default=None,
        help="Set MUJOCO_GL before importing LIBERO/MuJoCo, e.g. egl or osmesa.",
    )
    parser.add_argument(
        "--pi0_checkpoint",
        default="s3://openpi-assets/checkpoints/pi0_libero",
        help="Pi0 checkpoint path passed through openpi.shared.download.",
    )

    parser.add_argument("--launcher_preset", choices=PRESET_CHOICES, default=None)
    parser.add_argument(
        "--project_name",
        default=None,
        help="Project name used when rebuilding launcher presets.",
    )

    _add_optional_general_flags(parser)
    _add_optional_train_flags(parser)
    return parser.parse_args()


def _add_optional_general_flags(parser):
    for key, default in GENERAL_DEFAULTS.items():
        kwargs = {"default": None}
        if isinstance(default, tuple):
            kwargs.update(nargs="+", type=type(default[0]))
        elif isinstance(default, bool):
            kwargs.update(type=int)
        elif default is None:
            kwargs.update(nargs="*", type=int)
        else:
            kwargs.update(type=type(default))
        parser.add_argument(f"--{key}", **kwargs)


def _add_optional_train_flags(parser):
    for key, default in TRAIN_KWARGS_DEFAULTS.items():
        if key in GENERAL_DEFAULTS:
            continue
        kwargs = {"default": None}
        if isinstance(default, tuple):
            kwargs.update(nargs="+", type=type(default[0]))
        elif isinstance(default, bool):
            kwargs.update(type=int)
        else:
            kwargs.update(type=type(default))
        parser.add_argument(f"--{key}", **kwargs)


def _provided_flag_overrides(args):
    keys = set(GENERAL_DEFAULTS) | set(TRAIN_KWARGS_DEFAULTS)
    overrides = {}
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def infer_checkpoint_context(args):
    """Fill seed/suffix/preset from checkpoint/run directory names when possible."""
    run_name = _checkpoint_run_name(args.checkpoint_dir)
    match = RUN_DIR_RE.match(run_name)
    if not match:
        return

    prefix = match.group("prefix")
    suffix = match.group("suffix")
    seed = int(match.group("seed"))

    if getattr(args, "seed", None) is None:
        args.seed = seed
    if getattr(args, "suffix", None) is None and suffix is not None:
        args.suffix = suffix
    if args.launcher_preset is None:
        args.launcher_preset = _preset_from_prefix(prefix)

    print(
        "Inferred from checkpoint path: "
        f"launcher_preset={args.launcher_preset}, seed={args.seed}, "
        f"suffix={args.suffix}"
    )


def _checkpoint_run_name(checkpoint_dir):
    path = Path(checkpoint_dir)
    if path.name.startswith("checkpoint") and path.parent.name:
        return path.parent.name
    return path.name


def resolve_checkpoint_dir(args):
    """Allow --checkpoint_dir to be either a checkpoint folder or a run folder."""
    path = Path(args.checkpoint_dir)
    if path.name.startswith("checkpoint"):
        return
    if not path.exists() or not path.is_dir():
        return

    # Legacy Flax checkpoints are files, while Orbax checkpoints are
    # directories. Support both layouts.
    candidates = [
        child for child in path.iterdir()
        if child.name.startswith("checkpoint")
        and (child.is_file() or child.is_dir())
    ]
    if not candidates:
        return

    latest = max(candidates, key=_checkpoint_step)
    args.checkpoint_dir = latest
    print(f"Using latest checkpoint under run directory: {latest}")


def _checkpoint_step(path):
    match = re.search(r"(\d+)$", path.name)
    if match:
        return int(match.group(1))
    return -1


def _preset_from_prefix(prefix):
    if prefix == "dsrl_pi0_libero":
        return "sac"
    if prefix == "dsrl_pi0_libero_maxinfo":
        return "ablate"
    if prefix == "dsrl_pi0_libero_maxinfo_explorer":
        return "explorer"
    return None


def _load_preset_flags(args):
    if args.launcher_preset == "sac":
        from examples.scripts import launcher_sac as launcher
    elif args.launcher_preset == "ablate":
        from examples.scripts import launcher_ablate as launcher
    elif args.launcher_preset == "explorer":
        from examples.scripts import launcher_explorer as launcher
    else:
        raise ValueError(f"Unknown launcher preset: {args.launcher_preset}")

    project_name = args.project_name or launcher.PROJECT_NAME
    candidates = list(launcher.build_flags(project_name))
    overrides = _provided_flag_overrides(args)

    selector_keys = (
        "seed",
        "suffix",
        "add_tactile",
        "libero_suite",
        "libero_task_id",
        "touch_gripper_type",
        "ensemble_disagreement_modalities",
    )
    for key in selector_keys:
        if key in overrides:
            candidates = [
                flags for flags in candidates
                if _normalize_for_compare(flags.get(key)) == _normalize_for_compare(overrides[key])
            ]

    if len(candidates) != 1:
        preview = "\n".join(
            _candidate_summary(flags) for flags in candidates[:12]
        )
        if len(candidates) > 12:
            preview += f"\n... {len(candidates) - 12} more"
        raise ValueError(
            "Launcher preset selection did not resolve to exactly one run. "
            f"Matched {len(candidates)} candidates. Add --seed and --suffix "
            "or a more specific filter.\n"
            f"{preview}"
        )

    flags = copy.deepcopy(candidates[0])
    for key, value in overrides.items():
        flags[key] = value
    return flags


def _normalize_for_compare(value):
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return value


def _candidate_summary(flags):
    return (
        f"seed={flags.get('seed')} suffix={flags.get('suffix')} "
        f"add_tactile={flags.get('add_tactile')} "
        f"modalities={flags.get('ensemble_disagreement_modalities', '')!r}"
    )


def _explicit_flags(args):
    flags = copy.deepcopy(GENERAL_DEFAULTS)
    flags.update(_provided_flag_overrides(args))
    return flags


def build_variant(args):
    from jaxrl2.utils.general_utils import AttrDict

    flags = _load_preset_flags(args) if args.launcher_preset else _explicit_flags(args)

    data = {}
    data.update(copy.deepcopy(GENERAL_DEFAULTS))
    data.update(copy.deepcopy(TRAIN_KWARGS_DEFAULTS))
    data.update(copy.deepcopy(flags))
    data["dsrl_action_mode"] = "noise"

    train_kwargs = {
        key: copy.deepcopy(data[key])
        for key in TRAIN_KWARGS_DEFAULTS
    }
    data["train_kwargs"] = train_kwargs

    if data["env"] != "libero":
        raise ValueError("This renderer currently supports only --env libero.")
    if int(data["query_freq"]) <= 0:
        raise ValueError(
            "--query_freq must be positive. Launcher presets set it to 20; "
            "explicit configs must pass the training value."
        )
    if int(data["resize_image"]) <= 0:
        raise ValueError(
            "--resize_image must be positive so the DSRL pixel encoder input "
            "matches training."
        )

    return AttrDict(data)


def configure_environment(args):
    if args.mujoco_gl is not None:
        os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if os.environ.get("MUJOCO_GL") == "egl":
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    xla_flags = os.environ.get("XLA_FLAGS", "")
    triton_flag = "--xla_gpu_triton_gemm_any=True"
    if triton_flag not in xla_flags:
        os.environ["XLA_FLAGS"] = f"{xla_flags} {triton_flag}".strip()


def validate_render_args(args):
    if args.rollouts <= 0:
        raise ValueError("--rollouts must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.render_resolution <= 0:
        raise ValueError("--render_resolution must be positive.")
    if args.video_width <= 0 or args.video_height <= 0:
        raise ValueError("--video_width and --video_height must be positive.")
    if args.side_panel_width is not None and args.side_panel_width <= 0:
        raise ValueError("--side_panel_width must be positive when set.")
    if args.episode_steps <= 0:
        raise ValueError("--episode_steps must be positive.")
    if args.init_state_id < 0:
        raise ValueError("--init_state_id must be non-negative.")
    if args.settle_steps < 0:
        raise ValueError("--settle_steps must be non-negative.")
    if args.policy_seed is not None and args.policy_seed < 0:
        raise ValueError("--policy_seed must be non-negative when set.")
    if args.collection_dsrl and args.deterministic_dsrl:
        raise ValueError(
            "--collection_dsrl and --deterministic_dsrl cannot both be set."
        )


def import_runtime():
    from examples.sim_path_bootstrap import bootstrap_sim_paths

    bootstrap_sim_paths()

    import imageio.v2 as imageio
    import jax
    import tensorflow as tf
    from jaxrl2.utils.general_utils import add_batch_dim
    from libero.libero import benchmark
    from openpi.policies import policy_config
    from openpi.shared import download
    from openpi.training import config as openpi_config

    from examples.train_sim import (
        DummyEnv,
        _get_libero_env,
        _make_agent,
        _resolve_touch_gripper_type,
    )
    from examples import train_utils_sim

    DEFAULT_PI0_ACTION_HORIZON = getattr(
        train_utils_sim,
        "DEFAULT_PI0_ACTION_HORIZON",
        FALLBACK_PI0_ACTION_HORIZON,
    )
    DEFAULT_PI0_NOISE_DIM = getattr(
        train_utils_sim,
        "DEFAULT_PI0_NOISE_DIM",
        FALLBACK_PI0_NOISE_DIM,
    )

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError:
        pass

    return {
        "add_batch_dim": add_batch_dim,
        "benchmark": benchmark,
        "download": download,
        "imageio": imageio,
        "jax": jax,
        "openpi_config": openpi_config,
        "policy_config": policy_config,
        "DummyEnv": DummyEnv,
        "_get_libero_env": _get_libero_env,
        "_make_agent": _make_agent,
        "_resolve_touch_gripper_type": _resolve_touch_gripper_type,
        "DEFAULT_PI0_ACTION_HORIZON": DEFAULT_PI0_ACTION_HORIZON,
        "DEFAULT_PI0_NOISE_DIM": DEFAULT_PI0_NOISE_DIM,
        "get_pi0_action_horizon": getattr(
            train_utils_sim, "get_pi0_action_horizon",
            lambda agent_dp=None: int(getattr(
                agent_dp, "action_horizon", DEFAULT_PI0_ACTION_HORIZON))),
        "get_pi0_noise_dim": getattr(
            train_utils_sim, "get_pi0_noise_dim",
            lambda agent_dp=None: int(getattr(
                agent_dp, "action_dim", DEFAULT_PI0_NOISE_DIM))),
        "obs_to_agent_input": _required_attr(train_utils_sim, "obs_to_agent_input"),
        "obs_to_img": _required_attr(train_utils_sim, "obs_to_img"),
        "obs_to_pi_zero_input": _required_attr(
            train_utils_sim, "obs_to_pi_zero_input"),
        "obs_to_tactile": _required_attr(train_utils_sim, "obs_to_tactile"),
        "tactile_to_heatmap": _required_attr(train_utils_sim, "tactile_to_heatmap"),
    }


def _required_attr(module, name):
    if not hasattr(module, name):
        raise ImportError(
            f"{module.__name__} is missing {name!r}. The render script needs "
            "the same train_utils_sim rollout helpers used by training. "
            "Update the Euler checkout or copy the current helper implementation."
        )
    return getattr(module, name)


def build_runtime_state(args, variant, rt):
    benchmark_dict = rt["benchmark"].get_benchmark_dict()
    if variant.libero_suite not in benchmark_dict:
        raise ValueError(
            f"Unsupported LIBERO suite '{variant.libero_suite}'. "
            f"Supported suites: {sorted(benchmark_dict)}"
        )
    task_suite = benchmark_dict[variant.libero_suite]()
    task = task_suite.get_task(variant.libero_task_id)
    init_states = None
    if args.use_libero_init_state:
        init_states = task_suite.get_task_init_states(variant.libero_task_id)
    libero_robot = getattr(variant, "libero_robot", "Panda")
    touch_gripper_type = rt["_resolve_touch_gripper_type"](
        libero_robot,
        getattr(variant, "touch_gripper_type", "Robotiq85TactileGripper"),
    )
    variant.libero_robot = libero_robot
    variant.touch_gripper_type = touch_gripper_type
    env, task_description = rt["_get_libero_env"](
        task,
        args.render_resolution,
        variant.seed,
        libero_robot=libero_robot,
        use_touch=bool(getattr(variant, "use_touch", 0)),
        touch_gripper_type=touch_gripper_type,
    )

    variant.task_description = task_description
    variant.env_max_reward = 1
    variant.max_timesteps = int(args.episode_steps)
    variant.dsrl_action_mode = "noise"

    config = rt["openpi_config"].get_config("pi0_libero")
    pi0_checkpoint_dir = rt["download"].maybe_download(args.pi0_checkpoint)
    agent_dp = rt["policy_config"].create_trained_policy(config, pi0_checkpoint_dir)
    variant.pi0_noise_dim = int(
        getattr(agent_dp, "action_dim", rt["DEFAULT_PI0_NOISE_DIM"])
    )
    variant.pi0_action_horizon = int(
        getattr(agent_dp, "action_horizon", rt["DEFAULT_PI0_ACTION_HORIZON"])
    )

    dummy_env = rt["DummyEnv"](variant)
    sample_obs = rt["add_batch_dim"](dummy_env.observation_space.sample())
    sample_action = rt["add_batch_dim"](dummy_env.action_space.sample())
    agent = rt["_make_agent"](variant, sample_obs, sample_action)

    return (
        env,
        task_description,
        init_states,
        dummy_env,
        sample_obs,
        sample_action,
        agent,
        agent_dp,
    )


def restore_agent(args, agent):
    if not args.checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.checkpoint_dir}")
    agent.restore_checkpoint(str(args.checkpoint_dir))


def apply_policy_seed(args, agent, rt):
    if args.policy_seed is None:
        return
    agent._rng = rt["jax"].random.PRNGKey(int(args.policy_seed))


def render_rollout(args, variant, env, init_states, agent, agent_dp, rt, rollout_id):
    query_frequency = int(variant.query_freq)
    max_timesteps = int(variant.max_timesteps)
    action_horizon = rt["get_pi0_action_horizon"](agent_dp)
    pi0_noise_dim = rt["get_pi0_noise_dim"](agent_dp)

    obs, init_state_index = reset_rollout_env(
        args, env, init_states, variant, rollout_id)
    records = []
    rewards = []
    actions = None

    for t in range(max_timesteps):
        curr_image = rt["obs_to_img"](obs, variant)
        record = capture_record(args, variant, obs, rt)

        if t % query_frequency == 0:
            obs_dict = rt["obs_to_agent_input"](obs, variant, curr_image=curr_image)
            obs_pi_zero = rt["obs_to_pi_zero_input"](obs, variant)

            learner_action = select_eval_learner_action(args, agent, obs_dict)
            noise = learner_action_to_noise(
                learner_action,
                agent.action_chunk_shape,
                action_horizon,
                pi0_noise_dim,
            )
            actions = agent_dp.infer(obs_pi_zero, noise=noise)["actions"]

        action_t = actions[t % query_frequency]
        obs, reward, done, _ = env.step(action_t)
        rewards.append(reward)
        records.append(record)
        if done:
            break

    rewards = np.asarray(rewards, dtype=np.float32)
    episode_return = float(np.sum(rewards))
    highest_reward = float(np.max(rewards)) if rewards.size else 0.0
    success = bool(rewards.size and rewards[-1] == variant.env_max_reward)

    return {
        "records": records,
        "episode_len": len(records),
        "episode_return": episode_return,
        "highest_reward": highest_reward,
        "success": success,
        "init_state_index": init_state_index,
    }


def select_eval_learner_action(args, agent, obs_dict):
    if args.collection_dsrl:
        return agent.sample_actions(obs_dict)

    if args.deterministic_dsrl:
        return agent.eval_actions(obs_dict)

    if hasattr(agent, "collecting_exploration"):
        return agent.eval_actions(obs_dict)

    return agent.sample_actions(obs_dict)


def reset_rollout_env(args, env, init_states, variant, rollout_id):
    obs = env.reset()
    if not args.use_libero_init_state:
        return obs, None

    if init_states is None or len(init_states) == 0:
        raise ValueError("No LIBERO init states are available for this task.")

    init_state_index = (args.init_state_id + rollout_id) % len(init_states)
    obs = env.set_init_state(init_states[init_state_index])

    zero_action = zero_env_action(env)
    for _ in range(args.settle_steps):
        obs, _, done, _ = env.step(zero_action)
        if done:
            break
    return obs, init_state_index


def zero_env_action(env):
    action_dim = getattr(getattr(env, "env", None), "action_dim", None)
    if action_dim is not None:
        return np.zeros(action_dim, dtype=np.float32)

    for candidate in (
        getattr(env, "action_space", None),
        getattr(getattr(env, "env", None), "action_space", None),
    ):
        if candidate is not None and hasattr(candidate, "shape"):
            return np.zeros(candidate.shape, dtype=np.float32)

    for owner in (env, getattr(env, "env", None)):
        action_spec = getattr(owner, "action_spec", None)
        if action_spec is not None:
            low, _ = action_spec
            return np.zeros_like(np.asarray(low, dtype=np.float32))

    return np.zeros(7, dtype=np.float32)


def learner_action_to_noise(learner_action, action_chunk_shape, action_horizon,
                            pi0_noise_dim):
    noise_chunk = np.asarray(learner_action, dtype=np.float32).reshape(
        action_chunk_shape)
    if noise_chunk.shape[-1] != pi0_noise_dim:
        raise ValueError(
            f"DSRL action width {noise_chunk.shape[-1]} does not match Pi0 "
            f"noise dim {pi0_noise_dim}. This renderer expects noise-only "
            "checkpoints."
        )
    if noise_chunk.shape[0] >= action_horizon:
        return noise_chunk[:action_horizon][None]

    repeat = np.repeat(
        noise_chunk[-1:, :],
        action_horizon - noise_chunk.shape[0],
        axis=0,
    )
    return np.concatenate([noise_chunk, repeat], axis=0)[None]


def capture_record(args, variant, obs, rt):
    camera_key = f"{args.camera}_image"
    if camera_key not in obs:
        raise KeyError(f"Observation is missing camera key '{camera_key}'.")

    record = {
        "agentview": np.ascontiguousarray(obs[camera_key][::-1, ::-1]),
        "wrist": None,
        "tactile": None,
    }

    if args.include_wrist:
        wrist_key = "robot0_eye_in_hand_image"
        if wrist_key not in obs:
            raise KeyError(f"Observation is missing wrist camera key '{wrist_key}'.")
        record["wrist"] = np.ascontiguousarray(obs[wrist_key][::-1, ::-1])

    if should_render_tactile(args, variant):
        record["tactile"] = rt["obs_to_tactile"](obs, variant)

    return record


def should_render_tactile(args, variant):
    return (
        not args.no_tactile_panel
        and bool(getattr(variant, "use_touch", 0))
        and bool(getattr(variant, "add_tactile", 0))
    )


def compose_video_frames(args, records, rt):
    width = int(args.video_width)
    height = int(args.video_height)
    side_width = args.side_panel_width or max(1, width // 2)
    has_wrist = any(record["wrist"] is not None for record in records)
    has_tactile = any(record["tactile"] is not None for record in records)
    side_count = int(has_wrist) + int(has_tactile)
    out_width = width + (side_width if side_count else 0)

    max_tactile = 1e-8
    if has_tactile:
        max_tactile = max(
            float(np.linalg.norm(record["tactile"], axis=-1).max())
            for record in records
            if record["tactile"] is not None
        )

    frames = []
    for record in records:
        canvas = Image.new("RGB", (out_width, height), (0, 0, 0))
        main = _resize_rgb(record["agentview"], (width, height), Image.Resampling.BILINEAR)
        canvas.paste(main, (0, 0))

        side_y = 0
        if has_wrist:
            panel_height = height // side_count
            wrist = _resize_rgb(
                record["wrist"],
                (side_width, panel_height),
                Image.Resampling.BILINEAR,
            )
            canvas.paste(wrist, (width, side_y))
            side_y += panel_height

        if has_tactile:
            panel_height = height - side_y
            tactile = rt["tactile_to_heatmap"](
                record["tactile"],
                max_tactile,
                (side_width, panel_height),
            )
            tactile = Image.fromarray(tactile)
            canvas.paste(tactile, (width, side_y))

        frames.append(np.asarray(canvas, dtype=np.uint8))

    return frames


def _resize_rgb(frame, size, resampling):
    frame = np.asarray(frame, dtype=np.uint8)
    return Image.fromarray(frame).resize(size, resampling)


def output_path_for_rollout(output_path, rollout_id, num_rollouts, policy_seed):
    policy_seed_part = f"policyseed{policy_seed}"
    if output_path.suffix.lower() == ".mp4":
        stem = output_path.stem
        if num_rollouts == 1:
            return output_path.with_name(
                f"{stem}_{policy_seed_part}{output_path.suffix}"
            )
        return output_path.with_name(
            f"{stem}_{policy_seed_part}_rollout{rollout_id:03d}{output_path.suffix}"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / f"{policy_seed_part}_rollout_{rollout_id:03d}.mp4"


def write_rollout_outputs(args, variant, rollout, rt, rollout_id):
    video_path = output_path_for_rollout(
        args.output_path,
        rollout_id,
        args.rollouts,
        effective_policy_seed(args, variant),
    )
    video_path.parent.mkdir(parents=True, exist_ok=True)

    frames = compose_video_frames(args, rollout["records"], rt)
    rt["imageio"].mimsave(video_path, frames, fps=args.fps, macro_block_size=1)

    summary = rollout_summary(args, variant, rollout, video_path, rollout_id)
    summary_path = video_path.with_suffix(".json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return video_path, summary_path


def rollout_summary(args, variant, rollout, video_path, rollout_id):
    return {
        "rollout_id": rollout_id,
        "video_path": str(video_path),
        "checkpoint_dir": str(args.checkpoint_dir),
        "episode_len": rollout["episode_len"],
        "episode_return": rollout["episode_return"],
        "highest_reward": rollout["highest_reward"],
        "success": rollout["success"],
        "fps": args.fps,
        "render_resolution": args.render_resolution,
        "video_width": args.video_width,
        "video_height": args.video_height,
        "camera": args.camera,
        "init_state_id": args.init_state_id,
        "init_state_index": rollout["init_state_index"],
        "libero_init_state": bool(args.use_libero_init_state),
        "matches_training_eval_reset": not args.use_libero_init_state,
        "settle_steps": args.settle_steps,
        "include_wrist": bool(args.include_wrist),
        "include_tactile_panel": should_render_tactile(args, variant),
        "dsrl_action_selection": dsrl_action_selection(args, variant),
        "policy_seed": effective_policy_seed(args, variant),
        "policy_seed_override": args.policy_seed,
        "variant": serializable_variant(variant),
    }


def effective_policy_seed(args, variant):
    if args.policy_seed is not None:
        return int(args.policy_seed)
    return int(variant.seed)


def dsrl_action_selection(args, variant):
    if args.collection_dsrl:
        return "sample_actions_collection"
    if args.deterministic_dsrl:
        return "eval_actions_forced"
    if str(getattr(variant, "algorithm", "")).strip().lower() == "pixel_maxinfosac_explorer":
        return "eval_actions_training_eval_explorer"
    return "sample_actions_training_eval"


def serializable_variant(variant):
    skip = {"train_kwargs"}
    data = {
        key: _jsonable(value)
        for key, value in variant.items()
        if key not in skip
    }
    data["train_kwargs"] = {
        key: _jsonable(value)
        for key, value in variant["train_kwargs"].items()
    }
    return data


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def print_dry_run_summary(variant, sample_obs, sample_action, task_description):
    obs_shapes = {key: list(value.shape) for key, value in sample_obs.items()}
    print("Dry-run reconstruction complete.")
    print(f"  algorithm: {variant.algorithm}")
    print(f"  task: {variant.libero_suite}/{variant.libero_task_id}")
    print(f"  description: {task_description}")
    print(f"  add_tactile/use_touch: {variant.add_tactile}/{variant.use_touch}")
    print("  DSRL steering: Pi0 noise")
    print(f"  sample obs shapes: {obs_shapes}")
    print(f"  sample action shape: {list(sample_action.shape)}")


def main():
    args = parse_args()
    validate_render_args(args)
    infer_checkpoint_context(args)
    resolve_checkpoint_dir(args)
    configure_environment(args)
    rt = import_runtime()
    variant = build_variant(args)

    env = None
    try:
        env, task_description, init_states, _, sample_obs, sample_action, agent, agent_dp = (
            build_runtime_state(args, variant, rt)
        )
        if args.dry_run:
            print_dry_run_summary(variant, sample_obs, sample_action, task_description)
            return 0

        restore_agent(args, agent)
        apply_policy_seed(args, agent, rt)

        for rollout_id in range(args.rollouts):
            rollout = render_rollout(
                args, variant, env, init_states, agent, agent_dp, rt, rollout_id
            )
            video_path, summary_path = write_rollout_outputs(
                args, variant, rollout, rt, rollout_id
            )
            print(
                f"Rollout {rollout_id}: success={rollout['success']} "
                f"return={rollout['episode_return']:.3f} "
                f"len={rollout['episode_len']} video={video_path} "
                f"summary={summary_path}"
            )
    finally:
        if env is not None:
            env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
