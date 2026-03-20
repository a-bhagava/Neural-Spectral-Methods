#!/bin/bash
#SBATCH --job-name=replicate_ns_nsm_short
#SBATCH --time=20:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu-gpu-v100
#SBATCH --gres=gpu:1

# Activate conda environment
source ~/.bashrc
conda activate nsm_env

# Generate data 
python -m src.pde.navierstokes.generate ns

# Train models
for re in 4 3 2; do

    note=ns.T3/re"$re":NSM."$seed" iter=200000 ckpt=1000        bash run/.sh --pde navierstokes.re"$re" --model fno --hdim 32 --depth 10 --mode 12 31 31 --spectral

done


