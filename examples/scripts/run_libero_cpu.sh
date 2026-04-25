#!/bin/bash
proj_name=DSRL_pi0_Libero
device_id=0

export DISPLAY=:0
export MUJOCO_GL=glfw
unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
export EXP=./logs/DSRL_pi0_Libero
export OPENPI_DATA_HOME=./openpi
mkdir -p "$EXP"

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name; 
unset CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE=false

pip install mujoco==3.3.1

python -m examples.launch_train_sim \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero \
  --wandb_project ${proj_name} \
  --batch_size 256 \
  --discount 0.999 \
  --seed 0 \
  --max_steps 500000  \
  --eval_interval 10000 \
  --log_interval 500 \
  --eval_episodes 10 \
  --multi_grad_step 20 \
  --start_online_updates 500 \
  --resize_image 64 \
  --action_magnitude 1.0 \
  --query_freq 20 \
  --hidden_dims 128