#!/bin/bash

DEPTHS=(10 15)
HDIMS=(16 32)
LRS=(0.001 0.0005)
MODES=("12 31 31" "12 41 41" "18 31 31")

for DEPTH in "${DEPTHS[@]}"; do
for HDIM in "${HDIMS[@]}"; do
for LR in "${LRS[@]}"; do
for MODE in "${MODES[@]}"; do

    MODE_STR=$(echo $MODE | tr ' ' '_')
    JOB_NAME="ns_d${DEPTH}_h${HDIM}_lr${LR}_m${MODE_STR}"

    sbatch <<EOF
#!/bin/bash
#SBATCH -p mit_normal_gpu,mit_preemptable
#SBATCH --exclude node1928,node4007
#SBATCH --requeue
#SBATCH --mem=256G
#SBATCH -G 1
#SBATCH --output=navierstokes_%j.out
#SBATCH --signal=USR1@90
#SBATCH --time=05:59:00

source ~/.bashrc
conda activate nsm_env

XLA_PYTHON_CLIENT_PREALLOCATE=false exec python main.py --seed 42 --hdim ${HDIM} --depth ${DEPTH} --activate relu --pde navierstokes.re4_8 --model fno --mode ${MODE} --spectral train --bs 16 --lr ${LR} --schd exp --iter 200000 --vmap "" --ckpt 1000 --note "multiscale/wav_hyperparam/nsT3re4.length0.8/nsm_hdim${HDIM}_depth${DEPTH}_mode${MODE_STR}_lr${LR}" --save

EOF

    sleep 0.5

done
done
done
done