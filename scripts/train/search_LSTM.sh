#!/bin/bash

# 全局配置
export CUDA_VISIBLE_DEVICES=0
MODEL="LSTM"
DATASET="four_tank"      # 数据集名称 van_der_pol 或 four_tank
MODE="standard"            # 选项: standard 或 distill
DATA_SOURCE="real"         # 数据来源
N_TRAIN=1                  # 训练使用的数据轨迹数 (1-50)

echo "Starting Hyperparameter Grid Search for: ${LSTM}"
echo "Dataset: ${DATASET} | Mode: ${MODE}"

# for LAYERS in 1 2 3; do
#     for HIDDEN in 10 20 40 80; do
for LAYERS in 1; do
    for HIDDEN in 80; do
        echo "--------------------------------------------------"
        echo "Currently Training -> Layers: ${LAYERS} | Hidden Size: ${HIDDEN}"
        echo "--------------------------------------------------"

        python src/train/train_models.py \
            --model_type ${MODEL} \
            --dataset ${DATASET} \
            --mode ${MODE} \
            --data_source ${DATA_SOURCE} \
            --num_layers ${LAYERS} \
            --hidden_size ${HIDDEN} \
            --n_train ${N_TRAIN} \

    done
done

echo "All ${MODEL} combinations finished successfully!"