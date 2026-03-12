#!/bin/bash
#SBATCH --job-name=curledwake_hyperparameter_tuning
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu-gpu-v100
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/.bashrc
conda activate nsm_env

DEPTH=5
HDIM=32
MODE="12 31 31"
LR="0.001"
SCHD="exp"
MODE_STR=$(echo $MODE | tr ' ' '_')

# Inline environment variable assignment
note="cw.X10/yawed/hyperparameter_tuning/NSM_hdim${HDIM}_depth${DEPTH}_mode${MODE_STR}_lr${LR}" iter=20000 ckpt=100       bash run/.sh --pde curledwake.wake_re3 --model fno --hdim $HDIM --depth $DEPTH --mode $MODE --spectral
