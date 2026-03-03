#!/bin/bash

mkdir -p logs

DEPTHS=(5 10)
HDIMS=(16 32)
LRS=(0.0005)
MODES=("12 31 31" "12 41 41" "12 21 21" "18 31 31")

for DEPTH in "${DEPTHS[@]}"; do
for HDIM in "${HDIMS[@]}"; do
for LR in "${LRS[@]}"; do
for MODE in "${MODES[@]}"; do

    read MODE_X MODE_Y MODE_T <<< "$MODE"

    JOB_NAME="ns_d${DEPTH}_h${HDIM}_lr${LR}_m${MODE_X}_${MODE_Y}_${MODE_T}"

    sbatch \
        --job-name=${JOB_NAME} run/run_navierstokes.sh ${HDIM} ${DEPTH} ${LR} ${MODE_X} ${MODE_Y} ${MODE_T}

    sleep 0.5

done
done
done
done

