#!/bin/bash

# 全局配置
export CUDA_VISIBLE_DEVICES=0
MODEL="DeepAR"             # 选项: DeepAR 或 LSTM
DATASET="van_der_pol"        # 数据集名称 van_der_pol 或 four_tank
MODE="distill"             # 选项: standard 或 distill
DATA_SOURCE="real"         # 数据来源
SEQ_LEN=15               # 蒸馏标签生成时，TabPFN使用的context长度 (VdP取15, fourTank取20)
N_TRAIN=1                  # 训练使用的数据轨迹数 (1-50)
# DISTILL_WEIGHT=0.5         # 蒸馏权重系数

LAYERS=1
HIDDEN=80

echo "Starting Training for: ${MODEL}"
echo "Dataset: ${DATASET} | Mode: ${MODE}"

# 使用 seq 生成步长为 0.1 的序列
for DISTILL_WEIGHT in $(seq 0.6 0.1 1.0); do
    echo "----------------------------------------------------------------------"
    echo "Running -> Layers: ${LAYERS} | Hidden: ${HIDDEN} | Weight: ${DISTILL_WEIGHT}"
    echo "----------------------------------------------------------------------"

    python src/train/train_models.py \
        --model_type ${MODEL} \
        --dataset ${DATASET} \
        --mode ${MODE} \
        --data_source ${DATA_SOURCE} \
        --num_layers ${LAYERS} \
        --hidden_size ${HIDDEN} \
        --seq_len ${SEQ_LEN} \
        --n_train ${N_TRAIN} \
        --distill_weight ${DISTILL_WEIGHT} \

done

echo "All ${MODEL} weight grid search runs finished successfully!"