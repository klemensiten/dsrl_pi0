#!/usr/bin/env python
"""Measure LIBERO task-58 zero-action and gripper-only drift.

This diagnostic checks whether the robot moves even when the arm command is
zero. It is meant to catch controller/reset issues that can masquerade as a
bad policy: reset the env, apply zero arm commands with different gripper
commands, and log end-effector pose, joint state, torques, and controls.
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

SCENARIOS = {
    "zero_hold": 0.0,
    "open_hold": -1.0,
    "close_hold": 1.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--libero_suite", default="libero_90")
    parser.add_argument("--libero_task_id", type=int, default=58)
    parser.add_argument("--robots", nargs="+", default=("Panda", "Piper"))
    parser.add_argument("--controllers", nargs="+", default=("OSC_POSE",))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=tuple(SCENARIOS),
        default=("zero_hold", "open_hold", "close_hold"),
    )
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output_dir", default="/tmp/libero_task58_zero_hold_response")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--camera", default="agentview")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--trail_plot_range", type=float, default=0.5)
    parser.add_argument("--mujoco_gl", default="osmesa")
    parser.add_argument(
        "--osc_position_output_max",
        type=float,
        default=0.05,
        help="Use robosuite's DSRL-like OSC position scale by default.",
    )
    parser.add_argument(
        "--osc_orientation_output_max",
        type=float,
        default=None,
        help="Optional orientation output limit for OSC_POSE.",
    )
    parser.add_argument(
        "--body_gravcomp_scale",
        type=float,
        default=None,
        help=(
            "Optional runtime multiplier for MuJoCo model.body_gravcomp. "
            "Use 0.0 to test whether robot XML gravcomp is double-counting "
            "controller gravity compensation."
        ),
    )
    return parser.parse_args()


def make_controller_config(robosuite, controller, args):
    config = robosuite.load_controller_config(default_controller=controller)
    if controller in ("OSC_POSE", "OSC_POSITION"):
        output_max = np.asarray(config["output_max"], dtype=np.float64).copy()
        output_min = np.asarray(config["output_min"], dtype=np.float64).copy()
        output_max[:3] = float(args.osc_position_output_max)
        output_min[:3] = -float(args.osc_position_output_max)
        if args.osc_orientation_output_max is not None and output_max.size >= 6:
            output_max[3:6] = float(args.osc_orientation_output_max)
            output_min[3:6] = -float(args.osc_orientation_output_max)
        config["output_max"] = output_max.tolist()
        config["output_min"] = output_min.tolist()
    return config


def create_env(ControlEnv, robosuite, bddl_file, robot, controller, args):
    if robot not in ROBOT_GRIPPERS:
        raise ValueError(f"Unknown robot '{robot}'. Known robots: {sorted(ROBOT_GRIPPERS)}")
    return ControlEnv(
        str(bddl_file),
        robots=[robot],
        controller=controller,
        gripper_types=ROBOT_GRIPPERS[robot],
        controller_configs=make_controller_config(robosuite, controller, args),
        use_camera_obs=False,
        has_offscreen_renderer=False,
        hard_reset=False,
        horizon=max(1000, int(args.steps) + 10),
    )


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


def qpos_slice(sim, joint_name):
    addr = sim.model.get_joint_qpos_addr(joint_name)
    if isinstance(addr, tuple):
        return slice(addr[0], addr[1])
    return addr


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


def robot_ctrl(env):
    indexes = getattr(env.robots[0], "_ref_joint_actuator_indexes", None)
    if indexes is None:
        return []
    return np.asarray(env.sim.data.ctrl[indexes], dtype=np.float64).tolist()


def robot_torques(env):
    torques = getattr(env.robots[0], "torques", None)
    if torques is None:
        return []
    return np.asarray(torques, dtype=np.float64).tolist()


def controller_state(env):
    controller = getattr(env.robots[0], "controller", None)
    state = {}
    for name in ("ee_pos", "goal_pos", "goal_ori"):
        value = getattr(controller, name, None)
        if value is not None:
            state[name] = np.asarray(value, dtype=np.float64).tolist()
    return state


def gripper_joint_state(env):
    state = {}
    for name in env.sim.model.joint_names:
        if "finger" not in name and "gripper" not in name:
            continue
        sl = qpos_slice(env.sim, name)
        qpos = np.asarray(env.sim.data.qpos[sl], dtype=np.float64).reshape(-1)
        qvel = np.asarray(env.sim.data.qvel[sl], dtype=np.float64).reshape(-1)
        state[name] = {
            "qpos": qpos.tolist(),
            "qvel": qvel.tolist(),
        }
    return state


def gripper_ctrl(env):
    gripper = getattr(env.robots[0], "gripper", None)
    names = list(getattr(gripper, "actuators", []) or [])
    if not names:
        names = [
            name
            for name in env.sim.model.actuator_names
            if "finger" in name or "gripper" in name
        ]
    values = {}
    for name in names:
        try:
            actuator_id = env.sim.model.actuator_name2id(name)
        except ValueError:
            continue
        values[name] = float(env.sim.data.ctrl[actuator_id])
    return values


def make_action(action_dim, gripper_value):
    action = np.zeros(action_dim, dtype=np.float32)
    action[-1] = float(gripper_value)
    return action


def record_row(robot, controller, scenario, step, action, start_pos, env):
    pos = grip_pos(env)
    delta = pos - start_pos
    cstate = controller_state(env)
    return {
        "robot": robot,
        "controller": controller,
        "scenario": scenario,
        "step": step,
        "action": json.dumps(action.tolist()),
        "grip_x": float(pos[0]),
        "grip_y": float(pos[1]),
        "grip_z": float(pos[2]),
        "delta_x": float(delta[0]),
        "delta_y": float(delta[1]),
        "delta_z": float(delta[2]),
        "robot_qpos": json.dumps(robot_qpos(env)),
        "robot_qvel": json.dumps(robot_qvel(env)),
        "robot_ctrl": json.dumps(robot_ctrl(env)),
        "robot_torques": json.dumps(robot_torques(env)),
        "gripper_joint_state": json.dumps(gripper_joint_state(env), sort_keys=True),
        "gripper_ctrl": json.dumps(gripper_ctrl(env), sort_keys=True),
        "controller_state": json.dumps(cstate, sort_keys=True),
    }


def render_frame(renderer, data, camera, args, lines, trail):
    renderer.update_scene(data, camera=camera)
    robot_img = Image.fromarray(np.asarray(renderer.render(), dtype=np.uint8)).convert("RGB")
    panel = Image.new("RGB", robot_img.size, (18, 18, 18))
    image = Image.new("RGB", (robot_img.width + panel.width, robot_img.height), (0, 0, 0))
    image.paste(robot_img, (0, 0))
    image.paste(panel, (robot_img.width, 0))
    draw = ImageDraw.Draw(image)

    line_height = 14
    left = robot_img.width
    for i, line in enumerate(lines):
        draw.text((left + 6, 4 + line_height * i), line, fill=(255, 255, 255))

    plot_top = 8 + line_height * (len(lines) + 1)
    margin = 22
    plot_left = left + margin
    plot_right = image.width - margin
    plot_bottom = image.height - margin
    plot_height = max(1, plot_bottom - plot_top)
    z_zero = plot_bottom
    z_axis_x = image.width - 22
    scale = 0.8 * plot_height / max(float(args.trail_plot_range), 1e-6)

    draw.line([(z_axis_x, plot_top), (z_axis_x, plot_bottom)], fill=(95, 95, 95))
    draw.text((plot_left, plot_top - 16), "delta z", fill=(220, 220, 220))
    if trail:
        points = []
        for i, delta in enumerate(trail):
            x = plot_left + int(i * max(1, plot_right - plot_left) / max(1, len(trail) - 1))
            y = z_zero - int(float(delta[2]) * scale)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=(255, 220, 80), width=3)
        x, y = points[-1]
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill=(255, 90, 90))
    return np.asarray(image, dtype=np.uint8)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "robot",
        "controller",
        "scenario",
        "step",
        "action",
        "grip_x",
        "grip_y",
        "grip_z",
        "delta_x",
        "delta_y",
        "delta_z",
        "robot_qpos",
        "robot_qvel",
        "robot_ctrl",
        "robot_torques",
        "gripper_joint_state",
        "gripper_ctrl",
        "controller_state",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_video(path, frames, fps):
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, macro_block_size=1)


def summarize_rows(rows):
    if not rows:
        return {}
    final = rows[-1]
    max_abs = {
        axis: max(abs(float(row[f"delta_{axis}"])) for row in rows)
        for axis in ("x", "y", "z")
    }
    return {
        "initial_grip_pos": [
            float(rows[0]["grip_x"]),
            float(rows[0]["grip_y"]),
            float(rows[0]["grip_z"]),
        ],
        "final_grip_pos": [
            float(final["grip_x"]),
            float(final["grip_y"]),
            float(final["grip_z"]),
        ],
        "final_delta": [
            float(final["delta_x"]),
            float(final["delta_y"]),
            float(final["delta_z"]),
        ],
        "max_abs_delta": [max_abs["x"], max_abs["y"], max_abs["z"]],
        "final_robot_qpos": json.loads(final["robot_qpos"]),
        "final_robot_ctrl": json.loads(final["robot_ctrl"]),
        "final_robot_torques": json.loads(final["robot_torques"]),
        "final_gripper_ctrl": json.loads(final["gripper_ctrl"]),
    }


def run_scenario(env, renderer, data, args, robot, controller, scenario):
    obs = env.reset()
    del obs
    action = make_action(int(env.env.action_dim), SCENARIOS[scenario])
    start_pos = grip_pos(env)
    rows = []
    frames = []
    trail = []

    row = record_row(robot, controller, scenario, 0, action, start_pos, env)
    rows.append(row)
    trail.append(np.zeros(3, dtype=np.float64))
    frames.append(
        render_frame(
            renderer,
            data,
            args.camera,
            args,
            [
                f"{robot} | {controller} | {scenario}",
                "step 000 before action",
                "delta xyz +0.0000 +0.0000 +0.0000",
            ],
            trail,
        )
    )

    for step in range(1, args.steps + 1):
        obs, _, done, _ = env.step(action)
        del obs
        row = record_row(robot, controller, scenario, step, action, start_pos, env)
        rows.append(row)
        trail.append(
            np.array(
                [row["delta_x"], row["delta_y"], row["delta_z"]],
                dtype=np.float64,
            )
        )
        frames.append(
            render_frame(
                renderer,
                data,
                args.camera,
                args,
                [
                    f"{robot} | {controller} | {scenario}",
                    f"step {step:03d} done={bool(done)}",
                    (
                        f"delta xyz {row['delta_x']:+.4f} "
                        f"{row['delta_y']:+.4f} {row['delta_z']:+.4f}"
                    ),
                ],
                trail,
            )
        )
        if done:
            break
    return rows, frames, summarize_rows(rows)


def run_case(args, ControlEnv, mujoco, robosuite, bddl_file, robot, controller):
    run_dir = Path(args.output_dir) / f"{robot}_{controller}"
    env = None
    renderer = None
    scenario_summaries = {}
    all_rows = []

    try:
        env = create_env(ControlEnv, robosuite, bddl_file, robot, controller, args)
        gravcomp_info = apply_body_gravcomp_scale(env, args.body_gravcomp_scale)
        env.seed(args.seed)

        model = env.sim.model._model
        data = env.sim.data._data
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera) < 0:
            camera_names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(model.ncam)
            ]
            raise RuntimeError(f"Camera '{args.camera}' not found. Available: {camera_names}")
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)

        for scenario in args.scenarios:
            rows, frames, summary = run_scenario(
                env, renderer, data, args, robot, controller, scenario
            )
            all_rows.extend(rows)
            scenario_dir = run_dir / scenario
            write_csv(scenario_dir / "trajectory.csv", rows)
            write_video(scenario_dir / "zero_hold_response.mp4", frames, args.fps)
            scenario_summaries[scenario] = {
                **summary,
                "trajectory_path": str(scenario_dir / "trajectory.csv"),
                "video_path": str(scenario_dir / "zero_hold_response.mp4"),
            }

        summary = {
            "robot": robot,
            "controller": controller,
            "gripper": ROBOT_GRIPPERS[robot],
            "steps": args.steps,
            "scenarios": scenario_summaries,
            "osc_position_output_max": args.osc_position_output_max,
            "osc_orientation_output_max": args.osc_orientation_output_max,
            "body_gravcomp_scale": args.body_gravcomp_scale,
            "body_gravcomp_info": gravcomp_info,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "summary.json").open("w") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
        write_csv(run_dir / "trajectory_all.csv", all_rows)
        summary["summary_path"] = str(run_dir / "summary.json")
        summary["trajectory_all_path"] = str(run_dir / "trajectory_all.csv")
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
        raise ValueError(f"Unknown LIBERO suite '{args.libero_suite}'")
    task_suite = benchmark_dict[args.libero_suite]()
    task = task_suite.get_task(args.libero_task_id)
    bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

    summaries = []
    for robot in args.robots:
        for controller in args.controllers:
            summaries.append(
                run_case(args, ControlEnv, mujoco, robosuite, bddl_file, robot, controller)
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {
        "libero_suite": args.libero_suite,
        "libero_task_id": args.libero_task_id,
        "task_description": task.language,
        "task_bddl": str(bddl_file),
        "results": summaries,
    }
    aggregate_path = output_dir / "summary_all.json"
    with aggregate_path.open("w") as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)

    print("LIBERO task-58 zero-hold response")
    print(f"  task       : {task.language}")
    print(f"  summary_all: {aggregate_path}")
    for summary in summaries:
        for scenario, result in summary["scenarios"].items():
            delta = result["final_delta"]
            max_abs = result["max_abs_delta"]
            print(
                f"  {summary['robot']:5s} {summary['controller']:10s} {scenario:10s} "
                f"final_delta=({delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}) "
                f"max_abs_z={max_abs[2]:.4f}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
