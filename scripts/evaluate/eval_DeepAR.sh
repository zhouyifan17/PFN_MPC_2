#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# 参数设置
MODEL="DeepAR"
DATASET="four_tank"      # 选项: van_der_pol, four_tank
PRED_MODE="single_step"        # 选项: single_step, rolling
TRAIN_MODE="distill"       # 选项: standard, distill
N_PLOTS=0                 # 绘图数 (0~50)
MODEL_PATH="checkpoints/DeepAR/${DATASET}/real_${TRAIN_MODE}_N50/model_DeepAR_L1_H80_W0.4.pth"


# 执行区
echo "=================================================="
echo "Starting Evaluation for: ${MODEL}"
echo "Dataset  : ${DATASET}"
echo "Mode     : ${PRED_MODE}"
echo "=================================================="

python src/evaluate/evaluate_models.py \
    --model_type ${MODEL} \
    --dataset ${DATASET} \
    --pred_mode ${PRED_MODE} \
    --n_plots ${N_PLOTS} \
    --model_path ${MODEL_PATH} \

echo "${MODEL} Evaluation Completed!"