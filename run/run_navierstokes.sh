#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --partition=cpu-gpu-v100
#SBATCH --exclude=node4
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_log/multiscale/%x_%j.out
#SBATCH --error=slurm_log/multiscale/%x_%j.err

# ----------------------------
# Arguments passed from submit script
# ----------------------------
HDIM=$1
DEPTH=$2
LR=$3
MODE_X=$4
MODE_Y=$5
MODE_T=$6

MODE_STR="${MODE_X}_${MODE_Y}_${MODE_T}"

# ----------------------------
# Environment setup
# ----------------------------
source ~/.bashrc
conda activate nsm_env

# --- Force consistent JAX behavior ---
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TF_GPU_ALLOCATOR=cuda_malloc_async

# Optional: print for sanity
echo "Running with:"
echo "HDIM=$HDIM DEPTH=$DEPTH LR=$LR MODE=$MODE_X $MODE_Y $MODE_T"
echo "GPU info:"
nvidia-smi

# ----------------------------
# Run experiment
# ----------------------------

note="neurips/ns.T3re4.length0.4/multiscale_testing/NSM_hdim${HDIM}_depth${DEPTH}_mode${MODE_STR}_lr${LR}" iter=100000 ckpt=500  bash run/.sh --pde navierstokes.re4 --model fno --hdim ${HDIM} --depth ${DEPTH} --mode ${MODE_X} ${MODE_Y} ${MODE_T} --multiscale