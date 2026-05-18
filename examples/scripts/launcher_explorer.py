import argparse
import copy
import os
import re
import shlex

from jaxrl2.utils.launch_util import (
    dict_permutations,
    generate_base_command,
    generate_run_commands,
    hash_dict,
)


ENTITY = 'kiten'
PROJECT_NAME = 'DSRL_pi0_Libero_March_11_23_55_rtx4090'
MODULE_NAME = 'examples.launch_train_sim'


# base experiment.
BASE_FLAGS = {
    'algorithm': 'pixel_maxinfosac_explorer',
    'env': 'libero',
    'prefix': 'dsrl_pi0_libero_maxinfo_explorer',
    'wandb_project': PROJECT_NAME,
    'batch_size': 256,
    'discount': 0.999,
    'max_steps': 500000,
    'eval_interval': 10000,
    'log_interval': 500,
    'eval_episodes': 10,
    'multi_grad_step': 20,
    'start_online_updates': 500,
    'resize_image': 64,
    'action_magnitude': 1.0,
    'query_freq': 20,
    'explore_until': 300000,
    'hidden_dims': [128],
    'dyn_ent_lr': 0.0003,
    'init_dyn_ent_temperature': 0.5,
    'model_lr': 0.0003,
    'model_wd': 0.0001,
    'model_hidden_dims': [256, 256],
    'num_model_heads': 5,
    'model_noise_var': 1.0,
    'predict_reward': 1,
    'predict_diff': 1,
    'backup_entropy': 1,
    'model_obs_key': 'state',
    'obs_dim': 64,
    'ensemble_disagreement_modalities': 'image',
    'mask_expl_critic': 1,
    'tactile_hidden_dims': [256, 256],
    'mask_touch': 0,
}


SWEEP_FLAGS = {
    'seed': [0, 1, 2, 3, 4],
    'dyn_ent_lr': [0.0003],
    'init_dyn_ent_temperature': [1.0],
    'model_lr': [0.001],
    'model_wd': [0.0001],
    # 'model_hidden_dims': [[512, 512], [256, 256]],
    # 'num_model_heads': [7, 5, 1],
    # 'predict_reward': [1, 0],
    # 'backup_entropy': [1, 0],
    'ensemble_disagreement_modalities': [
        'state', 'latent', 'image', 'latent,state', 'latent,image',
        'state,image', ''
    ],
    # 'mask_expl_critic': [1, 0],
    'libero_suite': ['libero_90'],
    'libero_task_id': [
        58, 47
    ],
}


def build_flags(project_name):
    sweep_keys = list(SWEEP_FLAGS.keys())
    for sweep_flags in dict_permutations(SWEEP_FLAGS):
        flags = copy.deepcopy(BASE_FLAGS)
        flags['wandb_project'] = project_name
        flags.update(sweep_flags)
        flags.setdefault('suffix', hash_dict(sweep_flags))
        yield flags


def slug(value):
    value = value.replace('.', 'p').replace(',', '-')
    return re.sub(r'[^A-Za-z0-9_-]+', '-', value).strip('-')


def job_name(flags):
    seed = flags.get('seed', 'no_seed')
    suffix = flags.get('suffix', 'run')
    return slug(f"{flags['prefix']}_{suffix}_seed{seed}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_name', type=str, default=PROJECT_NAME)
    parser.add_argument('--entity', type=str, default=ENTITY)
    parser.add_argument('--scratch_root', type=str, default='/cluster/scratch')
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--num_cpus', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--gpu_type', type=str, default='rtx_4090')
    parser.add_argument('--mem', type=int, default=32000)
    parser.add_argument('--duration', type=str, default=None)
    parser.add_argument('--mode', type=str, default='euler',
                        choices=['euler', 'local', 'local_async'])
    parser.add_argument('--long_run', default=True, action='store_true')
    parser.add_argument('--dry', default=False, action='store_true')
    parser.add_argument('--prompt', default=True, action='store_true')
    return parser.parse_args()


def main(args):
    exp_dir = args.exp_dir
    if exp_dir is None:
        exp_dir = os.path.join(args.scratch_root, args.entity,
                               args.project_name)
    slurm_dir = os.path.join(exp_dir, 'slurm')

    if not args.dry:
        os.makedirs(slurm_dir, exist_ok=True)

    command_list = []
    output_file_list = []
    pre_commands = [
        f'mkdir -p {shlex.quote(exp_dir)} {shlex.quote(slurm_dir)}',
    ]
    env = {'EXP': exp_dir}

    for flags in build_flags(args.project_name):
        command_list.append(
            generate_base_command(
                MODULE_NAME,
                flags=flags,
                env=env,
                pre_commands=pre_commands,
            )
        )
        output_file_list.append(
            os.path.join(slurm_dir, f'{job_name(flags)}-%j.out')
        )

    duration = args.duration
    if duration is None:
        duration = '23:59:00' if args.long_run else '3:59:00'

    print(f'Prepared {len(command_list)} {args.mode} jobs.')
    generate_run_commands(
        command_list,
        output_file_list=output_file_list,
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        dry=args.dry,
        mem=args.mem,
        duration=duration,
        mode=args.mode,
        prompt=args.prompt,
        gpu_type=args.gpu_type,
    )


if __name__ == '__main__':
    main(parse_args())
