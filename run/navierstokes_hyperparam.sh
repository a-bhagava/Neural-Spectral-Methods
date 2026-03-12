#!/bin/bash

DEPTHS=(10)
HDIMS=(16)
LRS=(0.001)
MODES=("12 31 31")

for DEPTH in "${DEPTHS[@]}"; do
for HDIM in "${HDIMS[@]}"; do
for LR in "${LRS[@]}"; do
for MODE in "${MODES[@]}"; do

    MODE_STR=$(echo $MODE | tr ' ' '_')
    JOB_NAME="ns_d${DEPTH}_h${HDIM}_lr${LR}_m${MODE_STR}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=18:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu-gpu-v100
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate nsm_env

note="neurips/ns.T3re4.length0.4/wavelet_testing/wav1_hdim${HDIM}_depth${DEPTH}_mode${MODE_STR}_lr${LR}" iter=50000 ckpt=500       bash run/.sh --pde navierstokes.re4 --model fno --hdim ${HDIM} --depth ${DEPTH} --mode ${MODE} --spectral
EOF

    sleep 0.5

done
done
done
done
