#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

proj_name="${proj_name:-Test_Libero_Touch}"
device_id="${device_id:-0}"
libero_suite="${libero_suite:-libero_90}"
libero_task_id="${libero_task_id:-57}"
object_name="${object_name:-cream_cheese_1}"
camera="${camera:-agentview}"
fps="${fps:-20}"
width="${width:-320}"
height="${height:-240}"
output_dir="${output_dir:-/cluster/scratch/kiten/${proj_name}/touch_setup_${SLURM_JOB_ID:-local}}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-./openpi}"
export EXP="${EXP:-./logs/${proj_name}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${device_id}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

# The touch setup script uses mujoco.Renderer directly. OSMesa is the safest
# default for headless Euler jobs; override with MUJOCO_GL=egl if your allocation
# has EGL configured.
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
if [[ "${MUJOCO_GL}" == "egl" ]]; then
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
  export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${device_id}}"
fi

mkdir -p "${EXP}" "${output_dir}"

if [[ "${INSTALL_MUJOCO:-1}" == "1" ]]; then
  pip install mujoco==3.8.1
fi

python -u examples/test_libero_touch_setup.py \
  --libero_suite "${libero_suite}" \
  --libero_task_id "${libero_task_id}" \
  --object_name "${object_name}" \
  --camera "${camera}" \
  --output_dir "${output_dir}" \
  --fps "${fps}" \
  --width "${width}" \
  --height "${height}" \
  --mujoco_gl "${MUJOCO_GL}" \
  "$@"
