#!/usr/bin/env python
"""Compare LIBERO task-58 controller axis response for Panda and Piper.

This is a controller sanity test. It applies one normalized command axis at a
time and records how the world-frame gripper site actually moves. The most
important check for the Piper swing-up issue is whether a negative z command
makes ``gripper0_grip_site`` move down under the active controller.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.sim_path_bootstrap import bootstrap_sim_paths


ROBOT_GRIPPERS = {
    "Panda": "PandaGripper",
    "Piper": "PiperGripper",
}

COMMAND_AXES = (
    ("x_pos", 0, 1.0),
    ("x_neg", 0, -1.0),
    ("y_pos", 1, 1.0),
    ("y_neg", 1, -1.0),
    ("z_pos", 2, 1.0),
    ("z_neg", 2, -1.0),
    ("rx_pos", 3, 1.0),
    ("rx_neg", 3, -1.0),
    ("ry_pos", 4, 1.0),
    ("ry_neg", 4, -1.0),
    ("rz_pos", 5, 1.0),
    ("rz_neg", 5, -1.0),
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_suite", default="libero_90")
    parser.add_argument("--libero_task_id", type=int, default=58)
    parser.add_argument("--robots", nargs="+", default=("Panda", "Piper"))
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=("OSC_POSE", "OSC_POSITION"),
        help="Controllers to compare. OSC_POSITION skips orientation axes.",
    )
    parser.add_argument(
        "--axes",
        nargs="+",
        default=None,
        help=(
            "Optional axis names to render, e.g. z_neg z_pos. "
            "Defaults to all axes."
        ),
    )
    parser.add_argument("--output_dir", default="/tmp/libero_task58_axis_response")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--steps_per_axis", type=int, default=20)
    parser.add_argument(
        "--intro_frames_per_axis",
        type=int,
        default=5,
        help="Still frames to write before each axis command starts.",
    )
    parser.add_argument(
        "--hold_frames_per_axis",
        type=int,
        default=5,
        help="Still frames to write after each axis command finishes.",
    )
    parser.add_argument(
        "--trail_plot_range",
        type=float,
        default=0.6,
        help="Meters shown from center to edge in the video motion-trace panel.",
    )
    parser.add_argument("--action_magnitude", type=float, default=1.0)
    parser.add_argument(
        "--osc_position_output_max",
        type=float,
        default=0.15,
        help=(
            "Cartesian position output limit for OSC controllers. robosuite's "
            "default is 0.05, which can be hard to see in videos."
        ),
    )
    parser.add_argument(
        "--osc_orientation_output_max",
        type=float,
        default=None,
        help="Optional orientation output limit for OSC_POSE rotation axes.",
    )
    parser.add_argument("--open_gripper_action", type=float, default=-1.0)
    parser.add_argument("--mujoco_gl", default="osmesa")
    return parser.parse_args()


def make_action(action_dim, command_index, command_value, gripper_value):
    if command_index >= action_dim - 1:
        return None
    action = np.zeros(action_dim, dtype=np.float32)
    action[command_index] = command_value
    action[-1] = gripper_value
    return action


def grip_pos(env):
    site_id = env.sim.model.site_name2id("gripper0_grip_site")
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def robot_qpos(env):
    indexes = getattr(env.robots[0], "_ref_joint_pos_indexes", None)
    if indexes is None:
        return []
    return np.asarray(env.sim.data.qpos[indexes], dtype=np.float64).tolist()


def robot_qvel(env):
    indexes = getattr(env.robots[0], "_ref_joint_vel_indexes", None)
    if indexes is None:
        return []
    return np.asarray(env.sim.data.qvel[indexes], dtype=np.float64).tolist()


def grip_vel(env):
    return np.asarray(
        env.sim.data.get_site_xvelp("gripper0_grip_site"),
        dtype=np.float64,
    ).copy()


def array_or_none(value):
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return None
    return array.copy()


def json_array(value):
    if value is None:
        return ""
    return json.dumps(np.asarray(value, dtype=np.float64).tolist())


def vector_norm(value):
    if value is None:
        return None
    return float(np.linalg.norm(value))


def vector_component(value, index):
    if value is None or len(value) <= index:
        return None
    return float(value[index])


def controller_tracking_state(env, actual_grip_pos):
    robot = env.robots[0]
    controller = getattr(robot, "controller", None)

    goal_pos = array_or_none(getattr(controller, "goal_pos", None))
    controller_ee_pos = array_or_none(getattr(controller, "ee_pos", None))
    controller_ee_vel = array_or_none(getattr(controller, "ee_pos_vel", None))
    raw_torques = array_or_none(getattr(controller, "torques", None))
    applied_torques = array_or_none(getattr(robot, "torques", None))

    actuator_indexes = getattr(robot, "_ref_joint_actuator_indexes", None)
    data_ctrl = None
    if actuator_indexes is not None:
        data_ctrl = np.asarray(
            env.sim.data.ctrl[actuator_indexes],
            dtype=np.float64,
        ).copy()

    torque_low = None
    torque_high = None
    if hasattr(robot, "torque_limits"):
        torque_low, torque_high = (
            array_or_none(limit) for limit in robot.torque_limits
        )

    goal_error = None
    controller_goal_error = None
    if goal_pos is not None and goal_pos.size >= 3:
        goal_error = goal_pos[:3] - actual_grip_pos[:3]
        if controller_ee_pos is not None and controller_ee_pos.size >= 3:
            controller_goal_error = goal_pos[:3] - controller_ee_pos[:3]

    torque_clip_delta = None
    torque_saturated = False
    torque_saturated_count = 0
    torque_saturation_fraction = 0.0
    torque_limit_margin = None
    if (
        raw_torques is not None
        and applied_torques is not None
        and raw_torques.shape == applied_torques.shape
    ):
        torque_clip_delta = raw_torques - applied_torques
        torque_saturated = bool(np.any(np.abs(torque_clip_delta) > 1e-6))
        torque_saturated_count = int(np.count_nonzero(np.abs(torque_clip_delta) > 1e-6))
        torque_saturation_fraction = float(torque_saturated_count / raw_torques.size)

    if (
        applied_torques is not None
        and torque_low is not None
        and torque_high is not None
        and applied_torques.shape == torque_low.shape == torque_high.shape
    ):
        torque_limit_margin = np.minimum(
            applied_torques - torque_low,
            torque_high - applied_torques,
        )

    return {
        "controller_goal_pos": goal_pos,
        "controller_ee_pos": controller_ee_pos,
        "controller_ee_vel": controller_ee_vel,
        "goal_error_xyz": goal_error,
        "controller_goal_error_xyz": controller_goal_error,
        "raw_torques": raw_torques,
        "applied_torques": applied_torques,
        "data_ctrl": data_ctrl,
        "torque_low": torque_low,
        "torque_high": torque_high,
        "torque_clip_delta": torque_clip_delta,
        "torque_limit_margin": torque_limit_margin,
        "torque_saturated": torque_saturated,
        "torque_saturated_count": torque_saturated_count,
        "torque_saturation_fraction": torque_saturation_fraction,
    }


def render_frame(renderer, data, camera, lines, trail, plot_range):
    renderer.update_scene(data, camera=camera)
    robot = Image.fromarray(np.asarray(renderer.render(), dtype=np.uint8)).convert("RGB")
    panel = Image.new("RGB", robot.size, (18, 18, 18))
    image = Image.new("RGB", (robot.width + panel.width, robot.height), (0, 0, 0))
    image.paste(robot, (0, 0))
    image.paste(panel, (robot.width, 0))
    draw = ImageDraw.Draw(image)

    line_height = 14
    panel_left = robot.width
    panel_height = 8 + line_height * len(lines)
    draw.rectangle(
        [panel_left, 0, image.width, panel_height],
        fill=(0, 0, 0),
    )
    for i, line in enumerate(lines):
        draw.text((panel_left + 6, 4 + line_height * i), line, fill=(255, 255, 255))

    plot_margin = 22
    plot_left = panel_left + plot_margin
    plot_top = min(panel_height + plot_margin, image.height - plot_margin - 1)
    plot_right = max(plot_left + 1, image.width - plot_margin)
    plot_bottom = max(plot_top + 1, image.height - plot_margin)
    plot_width = max(1, plot_right - plot_left)
    plot_height = max(1, plot_bottom - plot_top)
    mid_x = plot_left + plot_width // 2
    mid_y = plot_top + plot_height // 2

    draw.rectangle(
        [plot_left, plot_top, plot_right, plot_bottom],
        outline=(95, 95, 95),
    )
    draw.line([(mid_x, plot_top), (mid_x, plot_bottom)], fill=(70, 70, 70))
    draw.line([(plot_left, mid_y), (plot_right, mid_y)], fill=(70, 70, 70))
    draw.text((plot_left, plot_top - 16), "xy trail", fill=(220, 220, 220))
    draw.text(
        (plot_left, plot_bottom + 4),
        f"range +/-{plot_range:.2f} m",
        fill=(180, 180, 180),
    )

    scale = 0.5 * min(plot_width, plot_height) / max(float(plot_range), 1e-6)
    points = []
    for delta in trail:
        x = mid_x + int(float(delta[0]) * scale)
        y = mid_y - int(float(delta[1]) * scale)
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(80, 220, 255), width=3)
    if points:
        draw.ellipse(
            [points[0][0] - 4, points[0][1] - 4, points[0][0] + 4, points[0][1] + 4],
            fill=(120, 120, 120),
        )
        draw.ellipse(
            [points[-1][0] - 5, points[-1][1] - 5, points[-1][0] + 5, points[-1][1] + 5],
            fill=(255, 90, 90),
        )

        z = float(trail[-1][2])
        z_center = panel_left + panel.width - 18
        z_zero = mid_y
        z_tip = z_zero - int(z * scale)
        draw.line([(z_center, plot_top), (z_center, plot_bottom)], fill=(95, 95, 95))
        draw.line([(z_center - 8, z_zero), (z_center + 8, z_zero)], fill=(130, 130, 130))
        draw.line([(z_center, z_zero), (z_center, z_tip)], fill=(255, 220, 80), width=5)
        draw.text((z_center - 36, plot_top), "z", fill=(220, 220, 220))

    return np.asarray(image, dtype=np.uint8)


def write_video(path, frames, fps):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def controller_config(env):
    controller = getattr(env.robots[0], "controller", None)
    config = getattr(controller, "controller_config", None)
    if config is None:
        config = getattr(env.robots[0], "controller_config", None)
    if config is None:
        return {}
    return jsonable(dict(config))


def make_controller_config(robosuite, controller, args):
    config = robosuite.load_controller_config(default_controller=controller)
    if controller in ("OSC_POSE", "OSC_POSITION"):
        output_max = np.asarray(config["output_max"], dtype=np.float64).copy()
        output_min = np.asarray(config["output_min"], dtype=np.float64).copy()
        position_limit = float(args.osc_position_output_max)
        output_max[:3] = position_limit
        output_min[:3] = -position_limit
        if args.osc_orientation_output_max is not None and output_max.size >= 6:
            orientation_limit = float(args.osc_orientation_output_max)
            output_max[3:6] = orientation_limit
            output_min[3:6] = -orientation_limit
        config["output_max"] = output_max.tolist()
        config["output_min"] = output_min.tolist()
    return config


def create_env(ControlEnv, robosuite, bddl_file, robot, controller, args):
    if robot not in ROBOT_GRIPPERS:
        raise ValueError(
            f"Unknown robot '{robot}'. Known robots: {sorted(ROBOT_GRIPPERS)}"
        )
    controller_configs = make_controller_config(robosuite, controller, args)
    return ControlEnv(
        str(bddl_file),
        robots=[robot],
        controller=controller,
        gripper_types=ROBOT_GRIPPERS[robot],
        controller_configs=controller_configs,
        use_camera_obs=False,
        has_offscreen_renderer=False,
        hard_reset=False,
    )


def run_axis_case(
    env,
    renderer,
    data,
    args,
    robot,
    controller,
    axis_name,
    axis_index,
    axis_sign,
    video_frames,
    initial_state,
):
    obs = env.regenerate_obs_from_state(initial_state)
    action_dim = int(env.env.action_dim)
    zero_action = np.zeros(action_dim, dtype=np.float32)
    zero_action[-1] = args.open_gripper_action

    for _ in range(args.settle_steps):
        obs, _, _, _ = env.step(zero_action)

    start_pos = grip_pos(env)
    command_value = float(axis_sign * args.action_magnitude)
    action = make_action(
        action_dim,
        axis_index,
        command_value,
        args.open_gripper_action,
    )
    if action is None:
        return [], {
            "robot": robot,
            "controller": controller,
            "axis": axis_name,
            "action_dim": action_dim,
            "skipped": True,
            "reason": "axis is not available for this action dimension",
        }

    rows = []
    final_reward = None
    final_done = False
    trail = [np.zeros(3, dtype=np.float64)]
    for _ in range(args.intro_frames_per_axis):
        video_frames.append(
            render_frame(
                renderer,
                data,
                args.camera,
                [
                    f"{robot} | {controller} | {axis_name} | start",
                    (
                        f"action[{axis_index}]={command_value:+.2f} "
                        f"gripper={args.open_gripper_action:+.2f}"
                    ),
                    "delta xyz +0.0000 +0.0000 +0.0000",
                ],
                trail,
                args.trail_plot_range,
            )
        )

    for step in range(args.steps_per_axis):
        obs, reward, done, _ = env.step(action)
        final_reward = reward
        final_done = bool(done)
        pos = grip_pos(env)
        vel = grip_vel(env)
        delta = pos - start_pos
        tracking = controller_tracking_state(env, pos)
        goal_pos = tracking["controller_goal_pos"]
        goal_delta = None if goal_pos is None else goal_pos[:3] - start_pos[:3]
        goal_error = tracking["goal_error_xyz"]
        controller_goal_error = tracking["controller_goal_error_xyz"]
        torque_clip_norm = vector_norm(tracking["torque_clip_delta"])
        torque_clip_max_abs = (
            None
            if tracking["torque_clip_delta"] is None
            else float(np.max(np.abs(tracking["torque_clip_delta"])))
        )
        torque_limit_margin_min = (
            None
            if tracking["torque_limit_margin"] is None
            else float(np.min(tracking["torque_limit_margin"]))
        )
        trail.append(delta.copy())
        video_frames.append(
            render_frame(
                renderer,
                data,
                args.camera,
                [
                    f"{robot} | {controller} | {axis_name} | step {step}",
                    (
                        f"action[{axis_index}]={command_value:+.2f} "
                        f"gripper={args.open_gripper_action:+.2f}"
                    ),
                    (
                        f"delta xyz "
                        f"{delta[0]:+.4f} {delta[1]:+.4f} {delta[2]:+.4f}"
                    ),
                    (
                        "goal err "
                        if goal_error is None
                        else (
                            f"goal err "
                            f"{goal_error[0]:+.4f} "
                            f"{goal_error[1]:+.4f} "
                            f"{goal_error[2]:+.4f}"
                        )
                    ),
                ],
                trail,
                args.trail_plot_range,
            )
        )
        rows.append(
            {
                "robot": robot,
                "controller": controller,
                "axis": axis_name,
                "step": step,
                "action_dim": action_dim,
                "action": json.dumps(action.tolist()),
                "grip_x": float(pos[0]),
                "grip_y": float(pos[1]),
                "grip_z": float(pos[2]),
                "grip_vx": float(vel[0]),
                "grip_vy": float(vel[1]),
                "grip_vz": float(vel[2]),
                "delta_x": float(delta[0]),
                "delta_y": float(delta[1]),
                "delta_z": float(delta[2]),
                "delta_norm": float(np.linalg.norm(delta)),
                "controller_goal_x": vector_component(goal_pos, 0),
                "controller_goal_y": vector_component(goal_pos, 1),
                "controller_goal_z": vector_component(goal_pos, 2),
                "goal_delta_x": vector_component(goal_delta, 0),
                "goal_delta_y": vector_component(goal_delta, 1),
                "goal_delta_z": vector_component(goal_delta, 2),
                "goal_error_x": vector_component(goal_error, 0),
                "goal_error_y": vector_component(goal_error, 1),
                "goal_error_z": vector_component(goal_error, 2),
                "goal_error_norm": vector_norm(goal_error),
                "controller_goal_error_x": vector_component(controller_goal_error, 0),
                "controller_goal_error_y": vector_component(controller_goal_error, 1),
                "controller_goal_error_z": vector_component(controller_goal_error, 2),
                "controller_goal_error_norm": vector_norm(controller_goal_error),
                "controller_ee_pos": json_array(tracking["controller_ee_pos"]),
                "controller_ee_vel": json_array(tracking["controller_ee_vel"]),
                "raw_torques": json_array(tracking["raw_torques"]),
                "applied_torques": json_array(tracking["applied_torques"]),
                "data_ctrl": json_array(tracking["data_ctrl"]),
                "torque_low": json_array(tracking["torque_low"]),
                "torque_high": json_array(tracking["torque_high"]),
                "torque_clip_delta": json_array(tracking["torque_clip_delta"]),
                "torque_clip_norm": torque_clip_norm,
                "torque_clip_max_abs": torque_clip_max_abs,
                "torque_limit_margin": json_array(tracking["torque_limit_margin"]),
                "torque_limit_margin_min": torque_limit_margin_min,
                "torque_saturated": tracking["torque_saturated"],
                "torque_saturated_count": tracking["torque_saturated_count"],
                "torque_saturation_fraction": tracking["torque_saturation_fraction"],
                "robot_qpos": json.dumps(robot_qpos(env)),
                "robot_qvel": json.dumps(robot_qvel(env)),
                "reward": None if final_reward is None else float(final_reward),
                "done": final_done,
            }
        )
        if done:
            break

    final_pos = grip_pos(env)
    delta = final_pos - start_pos
    final_tracking = controller_tracking_state(env, final_pos)
    final_goal_pos = final_tracking["controller_goal_pos"]
    final_goal_delta = (
        None if final_goal_pos is None else final_goal_pos[:3] - start_pos[:3]
    )
    final_goal_error = final_tracking["goal_error_xyz"]
    for _ in range(args.hold_frames_per_axis):
        video_frames.append(
            render_frame(
                renderer,
                data,
                args.camera,
                [
                    f"{robot} | {controller} | {axis_name} | final",
                    (
                        f"action[{axis_index}]={command_value:+.2f} "
                        f"gripper={args.open_gripper_action:+.2f}"
                    ),
                    (
                        f"delta xyz "
                        f"{delta[0]:+.4f} {delta[1]:+.4f} {delta[2]:+.4f}"
                    ),
                    (
                        "goal err "
                        if final_goal_error is None
                        else (
                            f"goal err "
                            f"{final_goal_error[0]:+.4f} "
                            f"{final_goal_error[1]:+.4f} "
                            f"{final_goal_error[2]:+.4f}"
                        )
                    ),
                ],
                trail,
                args.trail_plot_range,
            )
        )
    dominant_axis = ["x", "y", "z"][int(np.argmax(np.abs(delta)))]
    goal_error_norms = [
        row["goal_error_norm"] for row in rows if row["goal_error_norm"] is not None
    ]
    torque_clip_norms = [
        row["torque_clip_norm"] for row in rows if row["torque_clip_norm"] is not None
    ]
    torque_clip_max_abs_values = [
        row["torque_clip_max_abs"]
        for row in rows
        if row["torque_clip_max_abs"] is not None
    ]
    torque_saturation_fractions = [
        row["torque_saturation_fraction"]
        for row in rows
        if row["torque_saturation_fraction"] is not None
    ]
    torque_saturated_steps = sum(1 for row in rows if row["torque_saturated"])
    summary = {
        "robot": robot,
        "controller": controller,
        "axis": axis_name,
        "action_dim": action_dim,
        "skipped": False,
        "start_grip_pos": start_pos.tolist(),
        "final_grip_pos": final_pos.tolist(),
        "delta_xyz": delta.tolist(),
        "delta_norm": float(np.linalg.norm(delta)),
        "final_controller_goal_pos": (
            None if final_goal_pos is None else final_goal_pos[:3].tolist()
        ),
        "final_goal_delta_xyz": (
            None if final_goal_delta is None else final_goal_delta.tolist()
        ),
        "final_goal_error_xyz": (
            None if final_goal_error is None else final_goal_error.tolist()
        ),
        "final_goal_error_norm": vector_norm(final_goal_error),
        "max_goal_error_norm": (
            None if not goal_error_norms else float(max(goal_error_norms))
        ),
        "mean_goal_error_norm": (
            None if not goal_error_norms else float(np.mean(goal_error_norms))
        ),
        "max_torque_clip_norm": (
            None if not torque_clip_norms else float(max(torque_clip_norms))
        ),
        "max_torque_clip_abs": (
            None
            if not torque_clip_max_abs_values
            else float(max(torque_clip_max_abs_values))
        ),
        "torque_saturated_steps": int(torque_saturated_steps),
        "any_torque_saturated": bool(torque_saturated_steps > 0),
        "max_torque_saturation_fraction": (
            None
            if not torque_saturation_fractions
            else float(max(torque_saturation_fractions))
        ),
        "mean_torque_saturation_fraction": (
            None
            if not torque_saturation_fractions
            else float(np.mean(torque_saturation_fractions))
        ),
        "dominant_axis": dominant_axis,
        "steps_executed": len(rows),
        "command_index": axis_index,
        "command_value": command_value,
        "final_reward": None if final_reward is None else float(final_reward),
        "final_done": final_done,
    }
    if axis_name == "z_neg":
        summary["z_neg_moved_down"] = bool(delta[2] < 0.0)
    if axis_name == "z_pos":
        summary["z_pos_moved_up"] = bool(delta[2] > 0.0)
    return rows, summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "robot",
        "controller",
        "axis",
        "step",
        "action_dim",
        "action",
        "grip_x",
        "grip_y",
        "grip_z",
        "grip_vx",
        "grip_vy",
        "grip_vz",
        "delta_x",
        "delta_y",
        "delta_z",
        "delta_norm",
        "controller_goal_x",
        "controller_goal_y",
        "controller_goal_z",
        "goal_delta_x",
        "goal_delta_y",
        "goal_delta_z",
        "goal_error_x",
        "goal_error_y",
        "goal_error_z",
        "goal_error_norm",
        "controller_goal_error_x",
        "controller_goal_error_y",
        "controller_goal_error_z",
        "controller_goal_error_norm",
        "controller_ee_pos",
        "controller_ee_vel",
        "raw_torques",
        "applied_torques",
        "data_ctrl",
        "torque_low",
        "torque_high",
        "torque_clip_delta",
        "torque_clip_norm",
        "torque_clip_max_abs",
        "torque_limit_margin",
        "torque_limit_margin_min",
        "torque_saturated",
        "torque_saturated_count",
        "torque_saturation_fraction",
        "robot_qpos",
        "robot_qvel",
        "reward",
        "done",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    bootstrap_sim_paths()

    import mujoco
    import robosuite
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    if args.libero_suite not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO suite '{args.libero_suite}'. "
            f"Supported suites: {sorted(benchmark_dict)}"
        )

    task_suite = benchmark_dict[args.libero_suite]()
    task = task_suite.get_task(args.libero_task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    output_dir = Path(args.output_dir)
    all_rows = []
    summaries = []
    controller_configs = {}
    video_paths = {}
    if args.axes is None:
        axis_commands = COMMAND_AXES
    else:
        requested_axes = set(args.axes)
        known_axes = {axis_name for axis_name, _, _ in COMMAND_AXES}
        unknown_axes = sorted(requested_axes - known_axes)
        if unknown_axes:
            raise ValueError(
                f"Unknown axes {unknown_axes}. Known axes: {sorted(known_axes)}"
            )
        axis_commands = tuple(
            command for command in COMMAND_AXES if command[0] in requested_axes
        )

    for robot in args.robots:
        for controller in args.controllers:
            env = None
            renderer = None
            try:
                env = create_env(ControlEnv, robosuite, bddl_file, robot, controller, args)
                env.seed(args.seed)
                env.reset()
                initial_state = env.get_sim_state()
                controller_configs[f"{robot}_{controller}"] = controller_config(env)
                model = env.sim.model._model
                data = env.sim.data._data

                camera_id = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_CAMERA,
                    args.camera,
                )
                if camera_id < 0:
                    camera_names = [
                        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                        for i in range(model.ncam)
                    ]
                    raise RuntimeError(
                        f"Camera '{args.camera}' not found. "
                        f"Available cameras: {camera_names}"
                    )
                renderer = mujoco.Renderer(
                    model,
                    height=args.height,
                    width=args.width,
                )
                video_frames = []

                for axis_name, axis_index, axis_sign in axis_commands:
                    rows, summary = run_axis_case(
                        env,
                        renderer,
                        data,
                        args,
                        robot,
                        controller,
                        axis_name,
                        axis_index,
                        axis_sign,
                        video_frames,
                        initial_state,
                    )
                    all_rows.extend(rows)
                    summaries.append(summary)

                video_path = (
                    output_dir / f"{robot}_{controller}" / "axis_response.mp4"
                )
                write_video(video_path, video_frames, args.fps)
                video_paths[f"{robot}_{controller}"] = str(video_path)
            finally:
                if renderer is not None:
                    renderer.close()
                if env is not None:
                    env.close()

    csv_path = output_dir / "axis_response.csv"
    summary_path = output_dir / "summary.json"
    write_csv(csv_path, all_rows)

    summary = {
        "libero_suite": args.libero_suite,
        "libero_task_id": args.libero_task_id,
        "task_description": task.language,
        "task_bddl": str(bddl_file),
        "robots": list(args.robots),
        "controllers": list(args.controllers),
        "axes": [axis_name for axis_name, _, _ in axis_commands],
        "seed": args.seed,
        "camera": args.camera,
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "settle_steps": args.settle_steps,
        "steps_per_axis": args.steps_per_axis,
        "intro_frames_per_axis": args.intro_frames_per_axis,
        "hold_frames_per_axis": args.hold_frames_per_axis,
        "trail_plot_range": args.trail_plot_range,
        "action_magnitude": args.action_magnitude,
        "osc_position_output_max": args.osc_position_output_max,
        "osc_orientation_output_max": args.osc_orientation_output_max,
        "open_gripper_action": args.open_gripper_action,
        "mujoco_gl": args.mujoco_gl,
        "controller_configs": controller_configs,
        "video_paths": video_paths,
        "results": summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print("LIBERO task-58 axis response")
    print(f"  task     : {task.language}")
    print(f"  csv      : {csv_path}")
    print(f"  summary  : {summary_path}")
    for key, value in video_paths.items():
        print(f"  video {key}: {value}")
    for item in summaries:
        if item.get("axis") == "z_neg" and not item.get("skipped"):
            dz = item["delta_xyz"][2]
            moved = item.get("z_neg_moved_down")
            print(
                f"  {item['robot']:5s} {item['controller']:12s} "
                f"z_neg delta_z={dz:+.6f} moved_down={moved}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
