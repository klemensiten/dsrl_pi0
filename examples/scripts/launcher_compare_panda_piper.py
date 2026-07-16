"""Launch matched Panda/Piper evaluation of a Panda-trained DSRL checkpoint."""
import argparse
import os
import shlex

from jaxrl2.utils.launch_util import generate_run_commands


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rollouts", type=int, default=20)
    parser.add_argument("--libero_task_id", type=int, default=58)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scratch_root", default="/cluster/scratch")
    parser.add_argument("--entity", default="kiten")
    parser.add_argument("--mode", choices=("euler", "local", "local_async"), default="euler")
    parser.add_argument("--gpu_type", default="rtx_3090")
    parser.add_argument("--mem", type=int, default=32000)
    parser.add_argument("--duration", default="04:00:00")
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--no_prompt", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    log_dir = os.path.join(args.scratch_root, args.entity, "panda_piper_policy_compare")
    command = " ".join([
        "python -m examples.compare_panda_piper_policy",
        "--checkpoint_dir", shlex.quote(args.checkpoint_dir),
        "--output_dir", shlex.quote(args.output_dir),
        "--rollouts", str(args.rollouts),
        "--libero_task_id", str(args.libero_task_id),
        "--seed", str(args.seed),
    ])
    generate_run_commands(
        [command],
        output_file_list=[os.path.join(log_dir, "compare-%j.out")],
        num_cpus=1,
        num_gpus=1,
        dry=args.dry,
        mem=args.mem,
        duration=args.duration,
        mode=args.mode,
        prompt=not args.no_prompt,
        gpu_type=args.gpu_type,
    )


if __name__ == "__main__":
    main()
