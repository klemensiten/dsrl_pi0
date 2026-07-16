#!/usr/bin/env python
"""Evaluate one Panda-trained DSRL policy on matched Panda and Piper rollouts.

The trace keeps the learned DSRL noise and Pi0's decoded 7D actions separate.
This makes it possible to locate whether cross-embodiment divergence first
appears in the DSRL actor, Pi0, or environment execution.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

from examples import render_libero_rollout as render


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=10)
    parser.add_argument("--episode_steps", type=int, default=400)
    parser.add_argument("--init_state_id", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--libero_suite", default="libero_90")
    parser.add_argument("--libero_task_id", type=int, default=58)
    parser.add_argument("--query_freq", type=int, default=20)
    parser.add_argument("--resize_image", type=int, default=64)
    parser.add_argument("--algorithm", default="pixel_maxinfosac")
    parser.add_argument("--add_tactile", type=int, default=0)
    parser.add_argument("--use_touch", type=int, default=1)
    parser.add_argument("--tactile_shape", nargs=3, type=int, default=(32, 64, 3))
    parser.add_argument("--ensemble_disagreement_modalities", default="image")
    parser.add_argument("--launcher_preset", choices=render.PRESET_CHOICES)
    parser.add_argument("--project_name")
    parser.add_argument("--pi0_checkpoint", default="s3://openpi-assets/checkpoints/pi0_libero")
    parser.add_argument("--render_resolution", type=int, default=256)
    parser.add_argument("--mujoco_gl", default="egl")
    return parser.parse_args()


def render_namespace(args, robot):
    """Create the namespace expected by the shared checkpoint reconstruction."""
    values = vars(args).copy()
    values.update({
        "libero_robot": robot,
        "touch_gripper_type": f"{robot}Gripper",
        "use_libero_init_state": True,
        "deterministic_dsrl": True,
        "collection_dsrl": False,
        "policy_seed": args.seed,
        "dry_run": False,
        "fps": 50,
        "video_width": 640,
        "video_height": 480,
        "side_panel_width": None,
        "camera": "agentview",
        "include_wrist": False,
        "no_tactile_panel": True,
        "output_path": args.output_dir,
    })
    # Optional renderer flags must exist; None means use checkpoint/preset defaults.
    for key in set(render.GENERAL_DEFAULTS) | set(render.TRAIN_KWARGS_DEFAULTS):
        values.setdefault(key, None)
    return argparse.Namespace(**values)


def rollout(args, variant, env, init_states, agent, agent_dp, rt, rollout_id,
            replay=None, replay_mode="live"):
    if replay_mode not in ("live", "noise", "actions"):
        raise ValueError(f"Unknown replay mode: {replay_mode}")
    if replay_mode != "live" and replay is None:
        raise ValueError(f"Replay data is required for mode {replay_mode}")
    obs, init_index = render.reset_rollout_env(args, env, init_states, variant, rollout_id)
    horizon = rt["get_pi0_action_horizon"](agent_dp)
    noise_dim = rt["get_pi0_noise_dim"](agent_dp)
    actions = None
    rewards = []
    queries = []
    steps = []

    max_steps = int(args.episode_steps)
    if replay is not None:
        # Compare the same command horizon; do not invent commands after the
        # source Panda episode has terminated.
        max_steps = min(max_steps, len(replay["steps"]))

    for t in range(max_steps):
        if t % int(variant.query_freq) == 0:
            query_id = t // int(variant.query_freq)
            image = rt["obs_to_img"](obs, variant)
            actor_obs = rt["obs_to_agent_input"](obs, variant, curr_image=image)
            pi0_obs = rt["obs_to_pi_zero_input"](obs, variant)
            if replay_mode == "live":
                learner_action = agent.eval_actions(actor_obs)
                noise = render.learner_action_to_noise(
                    learner_action, agent.action_chunk_shape, horizon, noise_dim)
                actions = np.asarray(
                    agent_dp.infer(pi0_obs, noise=noise)["actions"], dtype=np.float32)
            else:
                source = replay["queries"][query_id]
                learner_action = source["dsrl_action"].copy()
                noise = source["pi0_noise"].copy()
                if replay_mode == "noise":
                    # Exact Panda noise, decoded against the current Piper observation.
                    actions = np.asarray(
                        agent_dp.infer(pi0_obs, noise=noise)["actions"], dtype=np.float32)
                else:
                    # Bypass both the DSRL actor and Pi0: execute Panda's exact 7D chunk.
                    actions = source["pi0_actions"].copy()
            queries.append({
                "step": t,
                "dsrl_action": np.asarray(learner_action, dtype=np.float32),
                "pi0_noise": np.asarray(noise, dtype=np.float32),
                "pi0_actions": actions.copy(),
                "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32),
                "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            })

        action = np.asarray(actions[t % int(variant.query_freq)], dtype=np.float32)
        before_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
        obs, reward, done, _ = env.step(action)
        steps.append({
            "step": t,
            "command": action,
            "eef_pos_before": before_pos,
            "eef_pos_after": np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            "gripper_qpos_after": np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
            "reward": float(reward),
        })
        rewards.append(float(reward))
        if done:
            break

    return {
        "init_state_index": init_index,
        "success": bool(rewards and rewards[-1] == variant.env_max_reward),
        "return": float(sum(rewards)),
        "replay_mode": replay_mode,
        "queries": queries,
        "steps": steps,
    }


def stack(items, key):
    return np.stack([np.asarray(item[key]) for item in items])


def save_trace(path, result):
    queries, steps = result["queries"], result["steps"]
    np.savez_compressed(
        path,
        query_steps=np.asarray([q["step"] for q in queries], dtype=np.int32),
        dsrl_actions=stack(queries, "dsrl_action"),
        pi0_noise=stack(queries, "pi0_noise"),
        pi0_actions=stack(queries, "pi0_actions"),
        query_eef_pos=stack(queries, "eef_pos"),
        query_eef_quat=stack(queries, "eef_quat"),
        query_gripper_qpos=np.asarray([q["gripper_qpos"] for q in queries], dtype=object),
        commands=stack(steps, "command"),
        eef_pos_before=stack(steps, "eef_pos_before"),
        eef_pos_after=stack(steps, "eef_pos_after"),
        gripper_qpos_after=np.asarray([s["gripper_qpos_after"] for s in steps], dtype=object),
        rewards=np.asarray([s["reward"] for s in steps], dtype=np.float32),
    )


def main():
    args = parse_args()
    if args.rollouts <= 0 or args.query_freq <= 0:
        raise ValueError("--rollouts and --query_freq must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render.configure_environment(args)
    rt = render.import_runtime()
    summaries = []

    panda_results = []
    conditions = (
        ("Panda", "live"),
        ("Piper", "live"),
        ("Piper", "noise"),
        ("Piper", "actions"),
    )
    for robot, replay_mode in conditions:
        robot_args = render_namespace(args, robot)
        render.resolve_checkpoint_dir(robot_args)
        variant = render.build_variant(robot_args)
        env = None
        try:
            env, _, init_states, _, _, _, agent, agent_dp = render.build_runtime_state(
                robot_args, variant, rt)
            render.restore_agent(robot_args, agent)
            render.apply_policy_seed(robot_args, agent, rt)
            for rollout_id in range(args.rollouts):
                replay = None if robot == "Panda" else panda_results[rollout_id]
                result = rollout(
                    robot_args, variant, env, init_states, agent, agent_dp, rt,
                    rollout_id, replay=replay, replay_mode=replay_mode)
                if robot == "Panda":
                    panda_results.append(result)
                condition = f"{robot.lower()}_{replay_mode}"
                trace_path = args.output_dir / f"{condition}_rollout_{rollout_id:03d}.npz"
                save_trace(trace_path, result)
                summary = {
                    "robot": robot,
                    "condition": condition,
                    "replay_mode": replay_mode,
                    "rollout_id": rollout_id,
                    "init_state_index": result["init_state_index"],
                    "success": result["success"],
                    "return": result["return"],
                    "steps": len(result["steps"]),
                    "queries": len(result["queries"]),
                    "trace": str(trace_path),
                }
                summaries.append(summary)
                print(json.dumps(summary, sort_keys=True))
        finally:
            if env is not None:
                env.close()

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, sort_keys=True))
    print(f"Wrote comparison summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
