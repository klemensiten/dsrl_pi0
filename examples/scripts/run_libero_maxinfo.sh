#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=1

export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="${device_id}"

export OPENPI_DATA_HOME=./openpi
export EXP="./logs/${proj_name}"
export CUDA_VISIBLE_DEVICES="${device_id}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${EXP}"

pip install mujoco==3.3.1

python -m examples.launch_train_sim \
  --algorithm pixel_maxinfosac \
  --env libero \
  --prefix dsrl_pi0_libero_maxinfosac \
  --wandb_project "${proj_name}" \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000 \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --query_freq 20 \
  --hidden_dims 128 \
  --dyn_ent_lr 0.0003 \
  --init_dyn_ent_temperature 1.0 \
  --model_lr 0.0003 \
  --model_wd 0.0 \
  --model_hidden_dims 256 256 \
  --num_model_heads 5 \
  --model_noise_var 1.0 \
  --predict_reward 1 \
  --predict_diff 1 \
  --backup_entropy 1 \
  --model_obs_key state \
  --obs_dim 64 \
  --ensemble_disagreement_modalities "" \
  --mask_expl_critic 0 \
  --tactile_hidden_dims 256 256 \
  --mask_touch 0
