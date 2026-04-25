<div align="center">

# DSRL for π₀: Diffusion Steering via Reinforcement Learning

## [[website](https://diffusion-steering.github.io)]      [[paper](https://arxiv.org/abs/2506.15799)]

</div>


## Overview
This repository provides the official implementation for our paper: [Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/abs/2506.15799) (CoRL 2025).

Specifically, it contains a JAX-based implementation of DSRL (Diffusion Steering via Reinforcement Learning) for steering a pre-trained generalist policy, [π₀](https://github.com/Physical-Intelligence/openpi), across various environments, including:

- **Simulation:** Libero, Aloha  
- **Real Robot:** Franka

If you find this repository useful for your research, please cite:

```
@article{wagenmaker2025steering,
  author    = {Andrew Wagenmaker and Mitsuhiko Nakamoto and Yunchu Zhang and Seohong Park and Waleed Yagoub and Anusha Nagabandi and Abhishek Gupta and Sergey Levine},
  title     = {Steering Your Diffusion Policy with Latent Space Reinforcement Learning},
  journal   = {Conference on Robot Learning (CoRL)},
  year      = {2025},
}
```

## Installation
1. Create a conda environment:
```
conda create -n dsrl_pi0 python=3.11.11
conda activate dsrl_pi0
```

2. Clone this repo with all submodules
```
git clone git@github.com:nakamotoo/dsrl_pi0.git --recurse-submodules
cd dsrl_pi0
```

3. Install all packages and dependencies
```
pip install -e .
pip install -r requirements.txt
pip install "jax[cuda12]==0.5.0"

# install openpi
pip install -e openpi
pip install -e openpi/packages/openpi-client

# install Libero
pip install -e LIBERO
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu # needed for libero
```

## Training (Simulation)
Libero
```
bash examples/scripts/run_libero.sh
```
Aloha
```
bash examples/scripts/run_aloha.sh
```
### Training Logs
We provide sample W&B runs and logs: https://wandb.ai/mitsuhiko/DSRL_pi0_public

## Training (Real)
For real-world experiments, we use the remote hosting feature from pi0 (see [here](https://github.com/Physical-Intelligence/openpi/blob/main/docs/remote_inference.md)) which enables us to host the pi0 model on a higher-spec remote server, in case the robot's client machine is not powerful enough. 

0. Setup Franka robot and install DROID package [[link](https://github.com/droid-dataset/droid.git)]

1. [On the remote server] Host pi0 droid model on your remote server
```
cd openpi && python scripts/serve_policy.py --env=DROID
```
2. [On your robot client machine] Run DSRL
```
bash examples/scripts/run_real.sh
```


## Credits
This repository is built upon [jaxrl2](https://github.com/ikostrikov/jaxrl2) and [PTR](https://github.com/Asap7772/PTR) repositories. 
In case of any questions, bugs, suggestions or improvements, please feel free to contact me at nakamoto\[at\]berkeley\[dot\]edu 


# Running LIBERO on a CPU-only machine (no GPU)

This documents the setup/debugging steps needed to run the LIBERO experiments in `dsrl_pi0` on a machine without a CUDA-capable GPU.

---

## 1. Create a clean Conda environment

A broken / contaminated Conda environment caused issues initially (ROS packages leaking in, missing Python binary).

Create a fresh env:

```bash
conda create -n dsrl python=3.11.11 -y
conda activate dsrl
```

Verify:

```bash
which python
which pip
python -V
python -m pip list
```

Expected:
- Python from `~/miniconda3/envs/dsrl/...`
- Pip from same env
- only a few base packages installed

---

## 2. Install project dependencies

From repo root:

```bash
cd ~/git/dsrl_pi0
python -m pip install -e .
# Dont forget to install JAX as cpu only
pip install "jax[cpu]==0.5.0"
```

---

## 3. Fix LIBERO import path

LIBERO package structure requires adding the local `LIBERO/` directory to `PYTHONPATH`.

From repo root:

```bash
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
```

Without this, imports like:

```python
from libero.libero import benchmark
```

fail.

---

## 4. Install LIBERO editable

```bash
cd ~/git/dsrl_pi0/LIBERO
python -m pip install -e .
cd ..
```

---

## 5. CPU-only rendering setup (important)

Default script assumed EGL / GPU rendering:

```bash
MUJOCO_GL=egl
```

This fails on machines without GPU support.

### Use OSMesa instead

```bash
unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

## 6. Install OSMesa / Mesa dependencies

Needed for CPU offscreen rendering:

```bash
sudo apt update
sudo apt install \
    libosmesa6 \
    libosmesa6-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev
```

## 7. Required environment variables for launch

From repo root:

```bash
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID

export EXP=./logs/DSRL_pi0_Libero
export OPENPI_DATA_HOME=./openpi

mkdir -p "$EXP"
```

---

## 8. Launch training manually

```bash
python -m examples.launch_train_sim \
  --algorithm pixel_sac \
  --env libero \
  --prefix dsrl_pi0_libero \
  --wandb_project DSRL_pi0_Libero \
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
  --hidden_dims 128
```

---

## 11. Recommended fixes for `examples/scripts/run_libero.sh`

Update script to include:

```bash
export PYTHONPATH="$PWD/LIBERO:$PYTHONPATH"

unset CUDA_VISIBLE_DEVICES
unset MUJOCO_EGL_DEVICE_ID

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

export OPENPI_DATA_HOME=./openpi
export EXP=./logs/$proj_name
mkdir -p "$EXP"
```

Also prefer:

```bash
python -m examples.launch_train_sim ...
```

instead of:

```bash
python3 examples/launch_train_sim.py
```

---

## 12. Remaining note: LIBERO datasets

Warning observed:

```bash
[Warning]: datasets path ... does not exist!
```

This may require downloading LIBERO datasets separately depending on experiment.

Check LIBERO documentation if dataset-related errors appear next.
