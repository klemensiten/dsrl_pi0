#!/usr/bin/env python
"""Scripted task-58 pickup comparison for Panda and Piper.

This places the task object below the live gripper site, then runs a simple
open -> descend -> close -> hold -> lift sequence. It is intended to separate
plain controller/robot behavior from pi_0 policy conditioning.
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
from examples.test_libero_touch_setup import (  # noqa: E402
    TACTILE_ROBOSUITE_FRAGMENT,
    TOUCH_LEFT,
    TOUCH_RIGHT,
    capture_frame,
    get_object_pos,
    qpos_slice,
    sensor_dim,
    write_outputs,
)


ROBOT_GRIPPERS = {
    "Panda": "PandaGripper",
    "Piper": "PiperGripper",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_suite", default="libero_90")
    parser.add_argument("--libero_task_id", type=int, default=58)
    parser.add_argument("--object_name", default="ketchup_1")
    parser.add_argument("--robots", nargs="+", default=("Panda", "Piper"))
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=("OSC_POSE", "OSC_POSITION"),
        help="Compare pose control against position-only control.",
    )
    parser.add_argument("--output_dir", default="/tmp/libero_task58_scripted_pickup")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)

    parser.add_argument("--pre_steps", type=int, default=20)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--descend_steps", type=int, default=220)
    parser.add_argument("--close_steps", type=int, default=40)
    parser.add_argument("--hold_steps", type=int, default=20)
    parser.add_argument("--lift_steps", type=int, default=120)
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Environment horizon. Defaults to max(1000, scripted steps + 10) "
            "so long DSRL-scale diagnostic runs do not terminate mid-script."
        ),
    )
    parser.add_argument(
        "--ignore_done",
        action="store_true",
        help="Pass ignore_done=True to the LIBERO/robosuite environment.",
    )

    parser.add_argument("--descend_action", type=float, default=-0.8)
    parser.add_argument("--close_descend_action", type=float, default=0.0)
    parser.add_argument("--lift_action", type=float, default=0.8)
    parser.add_argument(
        "--descend_ramp_steps",
        type=int,
        default=20,
        help="Linearly ramp the z descent command over this many control steps.",
    )
    parser.add_argument(
        "--lift_ramp_steps",
        type=int,
        default=20,
        help="Linearly ramp the z lift command over this many control steps.",
    )
    parser.add_argument(
        "--descend_stop_grip_object_z",
        type=float,
        default=0.0,
        help=(
            "Stop applying negative z once gripper z is below object placement "
            "z plus this margin. Use nan to disable the guard."
        ),
    )
    parser.add_argument(
        "--osc_position_output_max",
        type=float,
        default=0.05,
        help=(
            "Cartesian position output limit for OSC controllers. robosuite's "
            "default is 0.05; larger values are useful only for visual stress "
            "tests."
        ),
    )
    parser.add_argument(
        "--osc_orientation_output_max",
        type=float,
        default=None,
        help="Optional orientation output limit for OSC_POSE rotation axes.",
    )
    parser.add_argument("--open_gripper_action", type=float, default=-1.0)
    parser.add_argument("--close_gripper_action", type=float, default=1.0)

    parser.add_argument(
        "--placement_phase",
        choices=("after_baseline", "after_settle", "after_descend"),
        default="after_settle",
        help=(
            "after_settle places the object under the live gripper immediately "
            "before descent, which avoids measuring reset/controller warmup "
            "drift as part of the approach. after_baseline places earlier; "
            "after_descend is useful when only close/lift contact is being "
            "debugged."
        ),
    )
    parser.add_argument("--object_xy_offset", type=float, nargs=2, default=(0.025, 0.0))
    parser.add_argument("--object_z_offset", type=float, default=0.0)
    parser.add_argument(
        "--object_z_strategy",
        choices=("table", "reference"),
        default="table",
        help=(
            "table preserves the object's sampled table height. reference puts "
            "the free joint at the gripper reference z, mainly for tactile "
            "smoke tests."
        ),
    )

    parser.add_argument("--touch_threshold", type=float, default=1e-3)
    parser.add_argument("--lift_threshold", type=float, default=0.02)
    parser.add_argument("--descent_threshold", type=float, default=0.01)
    parser.add_argument(
        "--motion_plot_range",
        type=float,
        default=0.6,
        help="Meters shown from center to edge in the motion-trace video panel.",
    )
    parser.add_argument("--mujoco_gl", default="osmesa")
    parser.add_argument(
        "--body_gravcomp_scale",
        type=float,
        default=None,
        help=(
            "Optional runtime multiplier for MuJoCo model.body_gravcomp. "
            "Use 0.0 to test Piper without XML body gravcomp while keeping "
            "controller gravity compensation."
        ),
    )
    return parser.parse_args()


def touch_shape_from_dim(dim):
    if dim % 3 != 0:
        raise RuntimeError(f"Tactile sensor dim must be divisible by 3; got {dim}")
    cells = dim // 3
    side = int(np.sqrt(cells))
    if side * side != cells:
        raise RuntimeError(
            f"Expected square tactile grid from sensor dim {dim}; got {cells} cells"
        )
    return (3, side, side)


def verify_setup(env, robosuite, robot, gripper):
    robosuite_path = Path(robosuite.__file__).resolve().as_posix()
    if TACTILE_ROBOSUITE_FRAGMENT not in robosuite_path:
        raise RuntimeError(
            "robosuite is not imported from tactile_envs. "
            f"Got: {robosuite_path}"
        )

    robot_model_name = type(env.robots[0].robot_model).__name__
    robot_name = type(env.robots[0]).__name__
    gripper_name = type(env.robots[0].gripper).__name__
    if robot not in robot_model_name and robot not in robot_name:
        raise RuntimeError(
            f"Unexpected robot for request '{robot}': "
            f"robot={robot_name}, robot_model={robot_model_name}"
        )
    if gripper_name != gripper:
        raise RuntimeError(f"Unexpected gripper: {gripper_name}; expected {gripper}")

    sensor_names = set(env.sim.model.sensor_names)
    missing = [name for name in (TOUCH_LEFT, TOUCH_RIGHT) if name not in sensor_names]
    if missing:
        touch_like = [
            name
            for name in env.sim.model.sensor_names
            if "touch" in name.lower() or "tactile" in name.lower()
        ]
        raise RuntimeError(
            f"Missing tactile sensors: {missing}. "
            f"Touch-like sensors found: {touch_like}"
        )

    dims = {name: sensor_dim(env.sim, name) for name in (TOUCH_LEFT, TOUCH_RIGHT)}
    left_shape = touch_shape_from_dim(dims[TOUCH_LEFT])
    right_shape = touch_shape_from_dim(dims[TOUCH_RIGHT])
    expected_shapes = {
        "tactile_left": left_shape,
        "tactile_right": right_shape,
        "tactile": (left_shape[0], left_shape[1], left_shape[2] + right_shape[2]),
    }
    obs = env.env._get_observations(force_update=True)
    obs_shapes = {
        name: tuple(obs[name].shape)
        for name in expected_shapes
        if name in obs
    }
    if obs_shapes != expected_shapes:
        raise RuntimeError(
            f"Unexpected tactile observation shapes: {obs_shapes}; "
            f"expected {expected_shapes}"
        )

    return {
        "robosuite_file": robosuite_path,
        "robot": robot_name,
        "robot_model": robot_model_name,
        "gripper": gripper_name,
        "sensor_dims": dims,
        "obs_shapes": {key: list(value) for key, value in obs_shapes.items()},
    }


def grip_pos(env):
    site_id = env.sim.model.site_name2id("gripper0_grip_site")
    return np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64).copy()


def robot_qpos(env):
    indexes = getattr(env.robots[0], "_ref_joint_pos_indexes", None)
    if indexes is None:
        return []
    return np.asarray(env.sim.data.qpos[indexes], dtype=np.float64).tolist()


def make_action(action_dim, z=0.0, gripper=0.0):
    action = np.zeros(action_dim, dtype=np.float32)
    action[2] = z
    action[-1] = gripper
    return action


def planned_control_steps(args):
    return int(
        args.pre_steps
        + args.settle_steps
        + args.descend_steps
        + args.close_steps
        + args.hold_steps
        + args.lift_steps
    )


def env_horizon(args):
    if args.horizon is not None:
        return int(args.horizon)
    return max(1000, planned_control_steps(args) + 10)


def env_is_done(env):
    inner_env = getattr(env, "env", env)
    return bool(getattr(inner_env, "done", False))


def apply_body_gravcomp_scale(env, scale):
    if scale is None:
        return None
    model = getattr(env.sim.model, "_model", env.sim.model)
    body_gravcomp = getattr(model, "body_gravcomp", None)
    if body_gravcomp is None:
        raise RuntimeError("MuJoCo model does not expose body_gravcomp")
    before = np.asarray(body_gravcomp, dtype=np.float64).copy()
    body_gravcomp[:] = before * float(scale)
    env.sim.forward()
    return {
        "scale": float(scale),
        "before_nonzero": int(np.count_nonzero(before)),
        "after_nonzero": int(np.count_nonzero(body_gravcomp)),
        "before_sum": float(before.sum()),
        "after_sum": float(np.asarray(body_gravcomp, dtype=np.float64).sum()),
    }


def make_actions(action_dim, args):
    close = make_action(action_dim, gripper=args.close_gripper_action)
    close_descend = close.copy()
    close_descend[2] = args.close_descend_action
    return {
        "zero": np.zeros(action_dim, dtype=np.float32),
        "open": make_action(action_dim, gripper=args.open_gripper_action),
        "descend": make_action(
            action_dim,
            z=args.descend_action,
            gripper=args.open_gripper_action,
        ),
        "close": close,
        "close_descend": close_descend,
        "lift": make_action(
            action_dim,
            z=args.lift_action,
            gripper=args.close_gripper_action,
        ),
    }


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


def place_object_below_gripper(env, object_name, xy_offset, z_offset, z_strategy):
    joint_name = f"{object_name}_joint0"
    sim = env.sim
    if joint_name not in sim.model.joint_names:
        available = [
            name[: -len("_joint0")]
            for name in sim.model.joint_names
            if name and name.endswith("_joint0")
        ]
        raise RuntimeError(
            f"Could not find joint '{joint_name}'. "
            f"Available movable objects include: {available}"
        )

    reference_pos = grip_pos(env)
    sl = qpos_slice(sim, joint_name)
    qpos = sim.data.qpos[sl].copy()
    original_pos = qpos[:3].copy()

    qpos[0] = reference_pos[0] + xy_offset[0]
    qpos[1] = reference_pos[1] + xy_offset[1]
    if z_strategy == "table":
        qpos[2] = original_pos[2] + z_offset
    elif z_strategy == "reference":
        qpos[2] = reference_pos[2] + z_offset
    else:
        raise ValueError(f"Unknown object z strategy: {z_strategy}")

    sim.data.qpos[sl] = qpos
    sim.forward()

    return {
        "object_joint": joint_name,
        "object_initial_pos": original_pos,
        "object_placement_pos": qpos[:3].copy(),
        "placement_reference_pos": reference_pos,
    }


def run_phase(
    env,
    renderer,
    data,
    args,
    records,
    trajectory,
    phase,
    steps,
    action,
    obs,
    object_joint,
    initial_grip_pos,
    object_placement_pos,
    z_ramp_steps=0,
    stop_grip_object_z=None,
):
    for phase_step in range(steps):
        capture_frame(records, renderer, data, args.camera, obs, phase)
        frame_record = records[-1]
        curr_grip_pos = grip_pos(env)
        curr_object_pos = get_object_pos(env.sim, object_joint)
        grip_delta = curr_grip_pos - initial_grip_pos
        object_lift = float(curr_object_pos[2] - object_placement_pos[2])
        step_action = action

        if z_ramp_steps > 0 and action.size > 2:
            ramp = min(1.0, float(phase_step + 1) / float(z_ramp_steps))
            step_action = action.copy()
            step_action[2] *= ramp

        if (
            stop_grip_object_z is not None
            and np.isfinite(stop_grip_object_z)
            and step_action.size > 2
            and step_action[2] < 0.0
        ):
            min_grip_z = float(object_placement_pos[2] + stop_grip_object_z)
            if curr_grip_pos[2] <= min_grip_z:
                if step_action is action:
                    step_action = action.copy()
                step_action[2] = 0.0

        trajectory.append(
            {
                "frame": frame_record["frame"],
                "phase": phase,
                "phase_step": phase_step,
                "action": json.dumps(step_action.tolist()),
                "grip_x": float(curr_grip_pos[0]),
                "grip_y": float(curr_grip_pos[1]),
                "grip_z": float(curr_grip_pos[2]),
                "grip_delta_x": float(grip_delta[0]),
                "grip_delta_y": float(grip_delta[1]),
                "grip_delta_z": float(grip_delta[2]),
                "object_x": float(curr_object_pos[0]),
                "object_y": float(curr_object_pos[1]),
                "object_z": float(curr_object_pos[2]),
                "object_lift": object_lift,
                "touch_left_sum": frame_record["touch_left_sum"],
                "touch_right_sum": frame_record["touch_right_sum"],
                "touch_sum": frame_record["touch_sum"],
                "touch_max": frame_record["touch_max"],
                "robot_qpos": json.dumps(robot_qpos(env)),
            }
        )
        if env_is_done(env) and not args.ignore_done:
            return obs, True
        obs, _, done, _ = env.step(step_action)
        if (done or env_is_done(env)) and not args.ignore_done:
            return obs, True
    return obs, False


def write_trajectory(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame",
        "phase",
        "phase_step",
        "action",
        "grip_x",
        "grip_y",
        "grip_z",
        "grip_delta_x",
        "grip_delta_y",
        "grip_delta_z",
        "object_x",
        "object_y",
        "object_z",
        "object_lift",
        "touch_left_sum",
        "touch_right_sum",
        "touch_sum",
        "touch_max",
        "robot_qpos",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def compose_motion_frames(records, trajectory, args):
    if not records:
        return []

    center_row = None
    for row in trajectory:
        if row["phase"] != "settle":
            center_row = row
            break
    if center_row is None and trajectory:
        center_row = trajectory[0]
    if center_row is None:
        xy_center = np.zeros(2, dtype=np.float64)
        table_z = 0.0
    else:
        xy_center = np.array(
            [center_row["object_x"], center_row["object_y"]],
            dtype=np.float64,
        )
        table_z = float(center_row["object_z"])

    plot_range = max(float(args.motion_plot_range), 1e-6)
    frames = []
    trail = []
    for record, row in zip(records[-len(trajectory):], trajectory):
        robot = Image.fromarray(record["robot"]).resize(
            (args.width, args.height),
            Image.Resampling.BILINEAR,
        ).convert("RGB")
        panel = Image.new("RGB", (args.width, args.height), (18, 18, 18))
        image = Image.new("RGB", (args.width * 2, args.height), (0, 0, 0))
        image.paste(robot, (0, 0))
        image.paste(panel, (args.width, 0))
        draw = ImageDraw.Draw(image)

        grip = np.array([row["grip_x"], row["grip_y"], row["grip_z"]], dtype=np.float64)
        obj = np.array(
            [row["object_x"], row["object_y"], row["object_z"]],
            dtype=np.float64,
        )
        trail.append(grip.copy())

        lines = [
            f"{row['phase']} {int(row['phase_step']):03d}",
            f"grip z {grip[2]:+.3f} obj z {obj[2]:+.3f}",
            f"touch {row['touch_sum']:.3f} lift {row['object_lift']:+.3f}",
        ]
        line_height = 14
        header_h = 8 + line_height * len(lines)
        draw.rectangle([args.width, 0, args.width * 2, header_h], fill=(0, 0, 0))
        for i, line in enumerate(lines):
            draw.text((args.width + 6, 4 + line_height * i), line, fill=(255, 255, 255))

        margin = 22
        left = args.width + margin
        top = header_h + margin
        right = args.width * 2 - margin
        bottom = args.height - margin
        plot_w = max(1, right - left)
        plot_h = max(1, bottom - top)
        mid_x = left + plot_w // 2
        mid_y = top + plot_h // 2
        scale = 0.5 * min(plot_w, plot_h) / plot_range

        draw.rectangle([left, top, right, bottom], outline=(95, 95, 95))
        draw.line([(mid_x, top), (mid_x, bottom)], fill=(70, 70, 70))
        draw.line([(left, mid_y), (right, mid_y)], fill=(70, 70, 70))
        draw.text((left, top - 16), "xy grip/object", fill=(220, 220, 220))

        trail_points = [
            (
                mid_x + int((point[0] - xy_center[0]) * scale),
                mid_y - int((point[1] - xy_center[1]) * scale),
            )
            for point in trail
        ]
        if len(trail_points) > 1:
            draw.line(trail_points, fill=(80, 220, 255), width=3)
        if trail_points:
            x, y = trail_points[-1]
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 90, 90))

        obj_x = mid_x + int((obj[0] - xy_center[0]) * scale)
        obj_y = mid_y - int((obj[1] - xy_center[1]) * scale)
        draw.rectangle(
            [obj_x - 5, obj_y - 5, obj_x + 5, obj_y + 5],
            fill=(120, 255, 120),
        )

        z_axis_x = args.width * 2 - 18
        z_zero = bottom
        z_scale = 0.6 * (bottom - top)
        grip_tip = z_zero - int((grip[2] - table_z) * z_scale)
        obj_tip = z_zero - int((obj[2] - table_z) * z_scale)
        draw.line([(z_axis_x, top), (z_axis_x, bottom)], fill=(95, 95, 95))
        draw.line([(z_axis_x - 8, z_zero), (z_axis_x + 8, z_zero)], fill=(130, 130, 130))
        draw.line([(z_axis_x, z_zero), (z_axis_x, grip_tip)], fill=(255, 220, 80), width=5)
        draw.line([(z_axis_x - 10, obj_tip), (z_axis_x + 10, obj_tip)], fill=(120, 255, 120), width=3)
        draw.text((z_axis_x - 36, top), "z", fill=(220, 220, 220))

        frames.append(np.asarray(image, dtype=np.uint8))

    return frames


def write_motion_video(path, records, trajectory, args):
    import imageio.v2 as imageio

    frames = compose_motion_frames(records, trajectory, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=args.fps, macro_block_size=1)


def max_touch_sum(records, phases):
    phase_set = set(phases)
    values = [record["touch_sum"] for record in records if record["phase"] in phase_set]
    return max(values) if values else 0.0


def min_grip_z(trajectory, phases):
    phase_set = set(phases)
    values = [row["grip_z"] for row in trajectory if row["phase"] in phase_set]
    return min(values) if values else None


def run_case(args, ControlEnv, mujoco, robosuite, bddl_file, task, robot, controller):
    gripper = ROBOT_GRIPPERS[robot]
    run_dir = Path(args.output_dir) / f"{robot}_{controller}"
    env = None
    renderer = None
    horizon = env_horizon(args)
    planned_steps = planned_control_steps(args)

    try:
        env = ControlEnv(
            str(bddl_file),
            robots=[robot],
            controller=controller,
            gripper_types=gripper,
            controller_configs=make_controller_config(robosuite, controller, args),
            use_camera_obs=False,
            has_offscreen_renderer=False,
            hard_reset=False,
            horizon=horizon,
            ignore_done=args.ignore_done,
        )
        gravcomp_info = apply_body_gravcomp_scale(env, args.body_gravcomp_scale)
        env.seed(args.seed)
        obs = env.reset()
        setup_info = verify_setup(env, robosuite, robot, gripper)
        actions = make_actions(int(env.env.action_dim), args)

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
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

        records = []
        trajectory = []
        initial_grip_pos = grip_pos(env)
        episode_done = False

        placement_info = None
        dummy_object_joint = f"{args.object_name}_joint0"
        dummy_object_pos = np.array([np.nan, np.nan, np.nan], dtype=np.float64)

        if dummy_object_joint in env.sim.model.joint_names:
            dummy_object_pos = get_object_pos(env.sim, dummy_object_joint)

        for _ in range(args.pre_steps):
            capture_frame(records, renderer, data, args.camera, obs, "baseline")
            if env_is_done(env) and not args.ignore_done:
                episode_done = True
                break
            obs, _, done, _ = env.step(actions["open"])
            if (done or env_is_done(env)) and not args.ignore_done:
                episode_done = True
                break

        if args.placement_phase == "after_baseline":
            placement_info = place_object_below_gripper(
                env,
                args.object_name,
                np.asarray(args.object_xy_offset, dtype=np.float32),
                args.object_z_offset,
                args.object_z_strategy,
            )
            obs = env.env._get_observations(force_update=True)

        if placement_info is None:
            object_joint = dummy_object_joint
            object_placement_pos = dummy_object_pos
        else:
            object_joint = placement_info["object_joint"]
            object_placement_pos = placement_info["object_placement_pos"]

        if not episode_done:
            obs, episode_done = run_phase(
                env,
                renderer,
                data,
                args,
                records,
                trajectory,
                "settle",
                args.settle_steps,
                actions["zero"],
                obs,
                object_joint,
                initial_grip_pos,
                object_placement_pos,
            )

        if placement_info is None and args.placement_phase == "after_settle":
            placement_info = place_object_below_gripper(
                env,
                args.object_name,
                np.asarray(args.object_xy_offset, dtype=np.float32),
                args.object_z_offset,
                args.object_z_strategy,
            )
            obs = env.env._get_observations(force_update=True)
            object_joint = placement_info["object_joint"]
            object_placement_pos = placement_info["object_placement_pos"]

        descent_start_grip_pos = grip_pos(env)
        if not episode_done:
            obs, episode_done = run_phase(
                env,
                renderer,
                data,
                args,
                records,
                trajectory,
                "descend",
                args.descend_steps,
                actions["descend"],
                obs,
                object_joint,
                initial_grip_pos,
                object_placement_pos,
                z_ramp_steps=args.descend_ramp_steps,
                stop_grip_object_z=args.descend_stop_grip_object_z,
            )

        if placement_info is None:
            placement_info = place_object_below_gripper(
                env,
                args.object_name,
                np.asarray(args.object_xy_offset, dtype=np.float32),
                args.object_z_offset,
                args.object_z_strategy,
            )
            obs = env.env._get_observations(force_update=True)
            object_joint = placement_info["object_joint"]
            object_placement_pos = placement_info["object_placement_pos"]

        phases = (
            (
                "close",
                args.close_steps,
                actions["close_descend"],
                args.descend_ramp_steps,
                args.descend_stop_grip_object_z,
            ),
            ("hold", args.hold_steps, actions["close"], 0, None),
            ("lift", args.lift_steps, actions["lift"], args.lift_ramp_steps, None),
        )
        for phase, steps, action, z_ramp_steps, stop_grip_object_z in phases:
            if episode_done:
                break
            obs, episode_done = run_phase(
                env,
                renderer,
                data,
                args,
                records,
                trajectory,
                phase,
                steps,
                action,
                obs,
                object_joint,
                initial_grip_pos,
                object_placement_pos,
                z_ramp_steps=z_ramp_steps,
                stop_grip_object_z=stop_grip_object_z,
            )

        final_object_pos = get_object_pos(env.sim, object_joint)
        final_grip_pos = grip_pos(env)
        min_descend_z = min_grip_z(trajectory, ("descend", "close"))
        baseline_max = max_touch_sum(records, ("baseline",))
        contact_phases = ("descend", "close", "hold", "lift")
        contact_max = max_touch_sum(records, contact_phases)
        object_lift = float(final_object_pos[2] - object_placement_pos[2])
        gripper_descent = (
            None
            if min_descend_z is None
            else float(descent_start_grip_pos[2] - min_descend_z)
        )
        moved_down = (
            gripper_descent is not None
            and gripper_descent > args.descent_threshold
        )
        tactile_success = (
            baseline_max <= args.touch_threshold
            and contact_max > args.touch_threshold
        )
        lifted = object_lift > args.lift_threshold
        pickup_success = bool(tactile_success and lifted)

        summary = {
            **setup_info,
            "pickup_success": pickup_success,
            "moved_down": bool(moved_down),
            "tactile_success": bool(tactile_success),
            "lifted": bool(lifted),
            "object_lift": object_lift,
            "gripper_descent": gripper_descent,
            "descent_threshold": args.descent_threshold,
            "lift_threshold": args.lift_threshold,
            "baseline_touch_sum_max": baseline_max,
            "contact_touch_sum_max": contact_max,
            "touch_threshold": args.touch_threshold,
            "episode_done": bool(episode_done),
            "horizon": horizon,
            "ignore_done": bool(args.ignore_done),
            "planned_control_steps": planned_steps,
            "body_gravcomp_scale": args.body_gravcomp_scale,
            "body_gravcomp_info": gravcomp_info,
            "robot_request": robot,
            "controller": controller,
            "gripper_request": gripper,
            "action_dim": int(env.env.action_dim),
            "libero_suite": args.libero_suite,
            "libero_task_id": args.libero_task_id,
            "task_description": task.language,
            "task_bddl": str(bddl_file),
            "object_name": args.object_name,
            "object_joint": object_joint,
            "object_initial_pos": placement_info["object_initial_pos"].tolist(),
            "object_placement_pos": object_placement_pos.tolist(),
            "placement_reference_pos": placement_info[
                "placement_reference_pos"
            ].tolist(),
            "final_object_pos": final_object_pos.tolist(),
            "initial_grip_pos": initial_grip_pos.tolist(),
            "final_grip_pos": final_grip_pos.tolist(),
            "descent_start_grip_pos": descent_start_grip_pos.tolist(),
            "placement_phase": args.placement_phase,
            "object_z_strategy": args.object_z_strategy,
            "object_xy_offset": list(args.object_xy_offset),
            "object_z_offset": args.object_z_offset,
            "descend_action": args.descend_action,
            "close_descend_action": args.close_descend_action,
            "lift_action": args.lift_action,
            "descend_ramp_steps": args.descend_ramp_steps,
            "lift_ramp_steps": args.lift_ramp_steps,
            "descend_stop_grip_object_z": args.descend_stop_grip_object_z,
            "osc_position_output_max": args.osc_position_output_max,
            "osc_orientation_output_max": args.osc_orientation_output_max,
            "open_gripper_action": args.open_gripper_action,
            "close_gripper_action": args.close_gripper_action,
            "pre_steps": args.pre_steps,
            "settle_steps": args.settle_steps,
            "descend_steps": args.descend_steps,
            "close_steps": args.close_steps,
            "hold_steps": args.hold_steps,
            "lift_steps": args.lift_steps,
            "motion_plot_range": args.motion_plot_range,
            "camera": args.camera,
            "mujoco_gl": args.mujoco_gl,
            "num_frames": len(records),
        }

        video_path, touch_csv_path, summary_path = write_outputs(
            records,
            summary,
            run_dir,
            args.fps,
            args.width,
            args.height,
        )
        trajectory_path = run_dir / "trajectory.csv"
        write_trajectory(trajectory_path, trajectory)
        motion_video_path = run_dir / "motion_trace.mp4"
        write_motion_video(motion_video_path, records, trajectory, args)

        summary["video_path"] = str(video_path)
        summary["motion_video_path"] = str(motion_video_path)
        summary["touch_csv_path"] = str(touch_csv_path)
        summary["summary_path"] = str(summary_path)
        summary["trajectory_path"] = str(trajectory_path)
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)

        return summary
    finally:
        if renderer is not None:
            renderer.close()
        if env is not None:
            env.close()


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

    summaries = []
    for robot in args.robots:
        if robot not in ROBOT_GRIPPERS:
            raise ValueError(
                f"Unknown robot '{robot}'. Known robots: {sorted(ROBOT_GRIPPERS)}"
            )
        for controller in args.controllers:
            summaries.append(
                run_case(
                    args,
                    ControlEnv,
                    mujoco,
                    robosuite,
                    bddl_file,
                    task,
                    robot,
                    controller,
                )
            )

    output_dir = Path(args.output_dir)
    aggregate_path = output_dir / "summary_all.json"
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open("w") as f:
        json.dump(
            {
                "libero_suite": args.libero_suite,
                "libero_task_id": args.libero_task_id,
                "task_description": task.language,
                "task_bddl": str(bddl_file),
                "results": summaries,
            },
            f,
            indent=2,
            sort_keys=True,
        )

    print("LIBERO task-58 scripted pickup")
    print(f"  task       : {task.language}")
    print(f"  summary_all: {aggregate_path}")
    for summary in summaries:
        print(
            f"  {summary['robot_request']:5s} {summary['controller']:12s} "
            f"down={summary['moved_down']} "
            f"touch={summary['tactile_success']} "
            f"lifted={summary['lifted']} "
            f"object_lift={summary['object_lift']:+.6f} "
            f"dir={Path(summary['summary_path']).parent}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
