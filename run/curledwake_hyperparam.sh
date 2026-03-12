#!/bin/bash

DEPTHS=(5 10)
HDIMS=(16 32)
LRS=(0.0005)
MODES=("12 31 31" "12 21 21")

for DEPTH in "${DEPTHS[@]}"; do
for HDIM in "${HDIMS[@]}"; do
for LR in "${LRS[@]}"; do
for MODE in "${MODES[@]}"; do

    MODE_STR=$(echo $MODE | tr ' ' '_')
    JOB_NAME="cw_d${DEPTH}_h${HDIM}_lr${LR}_m${MODE_STR}"

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --time=8:00:00
#SBATCH --mem=32G
#SBATCH --partition=cpu-gpu-v100
#SBATCH --gres=gpu:1

source ~/.bashrc
conda activate nsm_env

note="neurips/cw.X10.yawed/hyperparameter_tuning/NSM_hdim${HDIM}_depth${DEPTH}_mode${MODE_STR}_lr${LR}" iter=20000 ckpt=100       bash run/.sh --pde curledwake.wake_re3 --model fno --hdim ${HDIM} --depth ${DEPTH} --mode ${MODE} --spectral
EOF

    sleep 0.5

done
done
done
done
done