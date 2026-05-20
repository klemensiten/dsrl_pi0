#!/usr/bin/env python
"""Deterministic LIBERO tactile gripper setup test.

This script validates the same tactile robosuite + LIBERO setup used by the
simulation experiments. It places one LIBERO object under the gripper, moves the
gripper down, closes and squeezes the tactile gripper, then lifts back up while
keeping the gripper closed. It saves a video that shows both the robot camera
and the touch-grid modality.
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


TOUCH_LEFT = "gripper0_touch_left"
TOUCH_RIGHT = "gripper0_touch_right"
TACTILE_GRIPPER = "Robotiq85TactileGripper"
TACTILE_ROBOSUITE_FRAGMENT = "tactile_envs/tactile_envs/envs/robosuite"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_suite", default="libero_90")
    parser.add_argument("--libero_task_id", type=int, default=57)

    # Use one of the cans by default.
    parser.add_argument("--object_name", default="tomato_sauce_1")

    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--output_dir", default="./tmp/libero_touch_setup_test")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)

    parser.add_argument("--pre_steps", type=int, default=20)
    parser.add_argument("--settle_steps", type=int, default=10)
    parser.add_argument("--descend_steps", type=int, default=20)
    parser.add_argument("--close_steps", type=int, default=40)
    parser.add_argument("--hold_steps", type=int, default=10)
    parser.add_argument("--squeeze_steps", type=int, default=20)
    parser.add_argument("--lift_steps", type=int, default=40)

    parser.add_argument("--descend_action", type=float, default=-1.0)
    parser.add_argument("--close_descend_action", type=float, default=-0.2)
    parser.add_argument("--lift_action", type=float, default=1.0)

    # Offset chosen from the earlier observed grip drift.
    parser.add_argument(
        "--object_xy_offset",
        type=float,
        nargs=2,
        default=(-0.0059, -0.00015),
    )
    parser.add_argument("--object_z_offset", type=float, default=0.0)

    parser.add_argument("--touch_threshold", type=float, default=1e-3)
    parser.add_argument("--lift_threshold", type=float, default=0.02)

    parser.add_argument(
        "--mujoco_gl",
        default="osmesa",
        help="MuJoCo GL backend to use before importing mujoco.",
    )
    return parser.parse_args()


def qpos_slice(sim, joint_name):
    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return slice(addr[0], addr[1])
    return addr


def get_object_pos(sim, joint_name):
    sl = qpos_slice(sim, joint_name)
    qpos = sim.data.qpos[sl]
    return np.asarray(qpos[:3]).copy()


def sensor_dim(sim, sensor_name):
    model = getattr(sim.model, "_model", sim.model)
    sid = sim.model.sensor_name2id(sensor_name)
    return int(model.sensor_dim[sid])


def verify_setup(env, robosuite, gripper_mapping):
    robosuite_file = Path(robosuite.__file__).resolve()
    robosuite_path = robosuite_file.as_posix()
    if TACTILE_ROBOSUITE_FRAGMENT not in robosuite_path:
        raise RuntimeError(
            "robosuite is not imported from tactile_envs. "
            f"Got: {robosuite_path}"
        )
    if TACTILE_GRIPPER not in gripper_mapping:
        raise RuntimeError(f"{TACTILE_GRIPPER} is not registered in GRIPPER_MAPPING.")

    robot_model_name = type(env.robots[0].robot_model).__name__
    gripper_name = type(env.robots[0].gripper).__name__
    if robot_model_name not in ("MountedPanda", "OnTheGroundPanda"):
        raise RuntimeError(f"Unexpected LIBERO robot model: {robot_model_name}")
    if gripper_name != TACTILE_GRIPPER:
        raise RuntimeError(f"Unexpected gripper: {gripper_name}")

    sensor_names = set(env.sim.model.sensor_names)
    missing = [name for name in (TOUCH_LEFT, TOUCH_RIGHT) if name not in sensor_names]
    if missing:
        raise RuntimeError(f"Missing tactile sensors: {missing}")

    dims = {name: sensor_dim(env.sim, name) for name in (TOUCH_LEFT, TOUCH_RIGHT)}
    if dims[TOUCH_LEFT] != 3072 or dims[TOUCH_RIGHT] != 3072:
        raise RuntimeError(f"Unexpected tactile sensor dims: {dims}")

    obs = env.env._get_observations(force_update=True)
    expected_shapes = {
        "tactile_left": (3, 32, 32),
        "tactile_right": (3, 32, 32),
        "tactile": (3, 32, 64),
    }
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
        "robot_model": robot_model_name,
        "gripper": gripper_name,
        "sensor_dims": dims,
        "obs_shapes": {key: list(value) for key, value in obs_shapes.items()},
    }


def get_touch_stats(obs):
    tactile_left = np.asarray(obs["tactile_left"], dtype=np.float32)
    tactile_right = np.asarray(obs["tactile_right"], dtype=np.float32)
    tactile = np.asarray(obs["tactile"], dtype=np.float32)
    return {
        "touch_left_sum": float(np.abs(tactile_left).sum()),
        "touch_right_sum": float(np.abs(tactile_right).sum()),
        "touch_sum": float(np.abs(tactile).sum()),
        "touch_max": float(np.abs(tactile).max()),
    }


def render_robot_frame(renderer, data, camera):
    renderer.update_scene(data, camera=camera)
    return np.asarray(renderer.render(), dtype=np.uint8)


def capture_frame(records, renderer, data, camera, obs, phase):
    stats = get_touch_stats(obs)
    records.append(
        {
            "frame": len(records),
            "phase": phase,
            "robot": render_robot_frame(renderer, data, camera),
            "tactile": np.asarray(obs["tactile"], dtype=np.float32),
            **stats,
        }
    )


def place_object_below_gripper(env, object_name, xy_offset, z_offset):
    joint_name = f"{object_name}_joint0"
    site_name = "gripper0_grip_site"
    sim = env.sim

    if joint_name not in sim.model.joint_names:
        available = [
            name[:-len("_joint0")]
            for name in sim.model.joint_names
            if name and name.endswith("_joint0")
        ]
        raise RuntimeError(
            f"Could not find joint '{joint_name}'. "
            f"Available movable objects include: {available}"
        )

    if site_name not in sim.model.site_names:
        raise RuntimeError(f"Could not find site '{site_name}'.")

    site_id = sim.model.site_name2id(site_name)
    grip_pos = sim.data.site_xpos[site_id].copy()

    sl = qpos_slice(sim, joint_name)
    qpos = sim.data.qpos[sl].copy()

    original_pos = qpos[:3].copy()
    qpos[0] = grip_pos[0] + xy_offset[0]
    qpos[1] = grip_pos[1] + xy_offset[1]
    qpos[2] = original_pos[2] + z_offset

    sim.data.qpos[sl] = qpos
    sim.forward()

    return joint_name, original_pos, qpos[:3].copy(), grip_pos


def tactile_heatmap(tactile, max_value, width, height, colormap):
    values = np.linalg.norm(tactile, axis=0)
    scale = max(float(max_value), 1e-8)
    normalized = np.clip(values / scale, 0.0, 1.0)
    colored = (colormap(normalized)[..., :3] * 255).astype(np.uint8)
    image = Image.fromarray(colored).resize((width, height), Image.Resampling.NEAREST)
    return image


def touch_trace(records, frame_index, width, height, max_sum):
    image = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(image)

    margin = max(4, min(24, width // 8, height // 5))
    plot_left = margin
    plot_top = margin
    plot_right = max(plot_left + 1, width - margin)
    plot_bottom = max(plot_top + 1, height - margin)

    draw.rectangle(
        [plot_left, plot_top, plot_right, plot_bottom],
        outline=(90, 90, 90),
    )
    draw.text((4, 2), "touch sum", fill=(230, 230, 230))

    if frame_index <= 0:
        return image

    values = [records[i]["touch_sum"] for i in range(frame_index + 1)]
    denom = max(float(max_sum), 1e-8)

    points = []
    for i, value in enumerate(values):
        x = plot_left + int((plot_right - plot_left) * i / max(1, len(values) - 1))
        y = plot_bottom - int((plot_bottom - plot_top) * min(value / denom, 1.0))
        points.append((x, y))

    if len(points) > 1:
        draw.line(points, fill=(80, 220, 255), width=2)

    if height >= 28:
        draw.text((4, height - 16), f"max {max(values):.3f}", fill=(230, 230, 230))

    return image


def compose_video_frames(records, width, height):
    from matplotlib import cm

    panel_width = width
    heatmap_height = height // 2
    trace_height = height - heatmap_height

    max_touch_value = max(record["touch_max"] for record in records)
    max_touch_sum = max(record["touch_sum"] for record in records)

    frames = []
    for i, record in enumerate(records):
        canvas = Image.new("RGB", (width + panel_width, height), (0, 0, 0))

        robot = Image.fromarray(record["robot"]).resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )
        heatmap = tactile_heatmap(
            record["tactile"],
            max_touch_value,
            panel_width,
            heatmap_height,
            cm.inferno,
        )
        trace = touch_trace(records, i, panel_width, trace_height, max_touch_sum)

        draw = ImageDraw.Draw(heatmap)
        draw.text(
            (8, 8),
            f"{record['phase']} | sum {record['touch_sum']:.3f}",
            fill=(255, 255, 255),
        )
        draw.line(
            [(panel_width // 2, 0), (panel_width // 2, heatmap_height)],
            fill=(255, 255, 255),
            width=1,
        )

        canvas.paste(robot, (0, 0))
        canvas.paste(heatmap, (width, 0))
        canvas.paste(trace, (width, heatmap_height))

        frames.append(np.asarray(canvas, dtype=np.uint8))

    return frames


def write_outputs(records, summary, output_dir, fps, width, height):
    import imageio.v2 as imageio

    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "touch_setup.mp4"
    csv_path = output_dir / "touch_log.csv"
    summary_path = output_dir / "summary.json"

    frames = compose_video_frames(records, width, height)
    imageio.mimsave(video_path, frames, fps=fps, macro_block_size=1)

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "phase",
                "touch_left_sum",
                "touch_right_sum",
                "touch_sum",
                "touch_max",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "frame": record["frame"],
                    "phase": record["phase"],
                    "touch_left_sum": record["touch_left_sum"],
                    "touch_right_sum": record["touch_right_sum"],
                    "touch_sum": record["touch_sum"],
                    "touch_max": record["touch_max"],
                }
            )

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return video_path, csv_path, summary_path


def main():
    args = parse_args()

    os.environ["MUJOCO_GL"] = args.mujoco_gl
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    bootstrap_sim_paths()

    import mujoco
    import robosuite
    from robosuite.models.grippers import GRIPPER_MAPPING
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

    env = ControlEnv(
        str(bddl_file),
        robots=["Panda"],
        gripper_types=TACTILE_GRIPPER,
        use_camera_obs=False,
        has_offscreen_renderer=False,
    )

    renderer = None

    try:
        obs = env.reset()
        setup_info = verify_setup(env, robosuite, GRIPPER_MAPPING)

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

        zero_action = np.zeros(env.env.action_dim)

        descend_action = np.zeros(env.env.action_dim)
        descend_action[2] = args.descend_action
        descend_action[-1] = -1.0

        close_action = np.zeros(env.env.action_dim)
        close_action[-1] = 1.0

        close_descend_action = close_action.copy()
        close_descend_action[2] = args.close_descend_action

        squeeze_action = np.zeros(env.env.action_dim)
        squeeze_action[-1] = 1.0

        lift_action = np.zeros(env.env.action_dim)
        lift_action[2] = args.lift_action
        lift_action[-1] = 1.0

        for _ in range(args.pre_steps):
            capture_frame(records, renderer, data, args.camera, obs, "baseline")
            obs, _, _, _ = env.step(zero_action)

        object_joint, object_initial_pos, object_placement_pos, initial_grip_pos = (
            place_object_below_gripper(
                env,
                args.object_name,
                np.asarray(args.object_xy_offset, dtype=np.float32),
                args.object_z_offset,
            )
        )

        obs = env.env._get_observations(force_update=True)

        for _ in range(args.settle_steps):
            capture_frame(records, renderer, data, args.camera, obs, "settle")
            obs, _, _, _ = env.step(zero_action)

        for _ in range(args.descend_steps):
            capture_frame(records, renderer, data, args.camera, obs, "descend")
            obs, _, _, _ = env.step(descend_action)

        for _ in range(args.close_steps):
            capture_frame(records, renderer, data, args.camera, obs, "close")
            obs, _, _, _ = env.step(close_descend_action)

        for _ in range(args.hold_steps):
            capture_frame(records, renderer, data, args.camera, obs, "hold")
            obs, _, _, _ = env.step(close_action)

        for _ in range(args.squeeze_steps):
            capture_frame(records, renderer, data, args.camera, obs, "squeeze")
            obs, _, _, _ = env.step(squeeze_action)

        for _ in range(args.lift_steps):
            capture_frame(records, renderer, data, args.camera, obs, "lift")
            obs, _, _, _ = env.step(lift_action)

        final_object_pos = get_object_pos(env.sim, object_joint)
        final_grip_pos = env.sim.data.site_xpos[
            env.sim.model.site_name2id("gripper0_grip_site")
        ].copy()

        baseline_max = max(
            record["touch_sum"]
            for record in records
            if record["phase"] == "baseline"
        )

        contact_phases = ("descend", "close", "hold", "squeeze", "lift")
        contact_max = max(
            record["touch_sum"]
            for record in records
            if record["phase"] in contact_phases
        )

        first_contact_frame = next(
            (
                record["frame"]
                for record in records
                if record["phase"] in contact_phases
                and record["touch_sum"] > args.touch_threshold
            ),
            None,
        )

        object_lift = float(final_object_pos[2] - object_placement_pos[2])
        lifted = object_lift > args.lift_threshold

        tactile_success = (
            baseline_max <= args.touch_threshold
            and contact_max > args.touch_threshold
        )

        # This says the tactile setup worked. The separate "lifted" field says whether
        # the object actually moved upward with the gripper.
        success = bool(tactile_success)

        summary = {
            **setup_info,
            "success": success,
            "tactile_success": bool(tactile_success),
            "lifted": bool(lifted),
            "object_lift": object_lift,
            "lift_threshold": args.lift_threshold,
            "first_contact_frame": first_contact_frame,
            "libero_suite": args.libero_suite,
            "libero_task_id": args.libero_task_id,
            "task_bddl": str(bddl_file),
            "task_description": task.language,
            "object_name": args.object_name,
            "object_joint": object_joint,
            "object_initial_pos": object_initial_pos.tolist(),
            "object_placement_pos": object_placement_pos.tolist(),
            "final_object_pos": final_object_pos.tolist(),
            "initial_grip_pos": initial_grip_pos.tolist(),
            "final_grip_pos": final_grip_pos.tolist(),
            "object_xy_offset": list(args.object_xy_offset),
            "object_z_offset": args.object_z_offset,
            "descend_action": args.descend_action,
            "close_descend_action": args.close_descend_action,
            "lift_action": args.lift_action,
            "pre_steps": args.pre_steps,
            "settle_steps": args.settle_steps,
            "descend_steps": args.descend_steps,
            "close_steps": args.close_steps,
            "hold_steps": args.hold_steps,
            "squeeze_steps": args.squeeze_steps,
            "lift_steps": args.lift_steps,
            "camera": args.camera,
            "mujoco_gl": args.mujoco_gl,
            "num_frames": len(records),
            "baseline_touch_sum_max": baseline_max,
            "contact_touch_sum_max": contact_max,
            "touch_threshold": args.touch_threshold,
        }

        video_path, csv_path, summary_path = write_outputs(
            records,
            summary,
            Path(args.output_dir),
            args.fps,
            args.width,
            args.height,
        )

    finally:
        if renderer is not None:
            renderer.close()
        env.close()

    print("LIBERO touch setup test")
    print(f"  success       : {success}")
    print(f"  tactile success: {tactile_success}")
    print(f"  lifted        : {lifted}")
    print(f"  object lift   : {object_lift:.6f}")
    print(f"  baseline touch max: {baseline_max:.6f}")
    print(f"  contact touch max : {contact_max:.6f}")
    print(f"  first contact frame: {first_contact_frame}")
    print(f"  video  : {video_path}")
    print(f"  csv    : {csv_path}")
    print(f"  summary: {summary_path}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())