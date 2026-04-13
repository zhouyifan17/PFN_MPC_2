#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 参数设置
MODEL="TabPFN"
DATASET="van_der_pol"      # 选项: van_der_pol, four_tank
PRED_MODE="single_step"        # 选项: single_step, rolling
N_PLOTS=50                 # 绘图数量 (0~50)
CONTEXT_MODE="mixed"       # 选项: independent, mixed
SEQ_LEN=1                  # 滑动窗口长度
CONTEXT_SIZE=8000          # Context 样本池最大数量
N_CONTEXT_TRAJS=50         # 构建 Context 的轨迹数

# 执行区
echo "=================================================="
echo "Starting Evaluation for: ${MODEL}"
echo "Dataset      : ${DATASET}"
echo "Mode         : ${PRED_MODE}"
echo "Context Mode : ${CONTEXT_MODE} (Trajs: ${N_CONTEXT_TRAJS})"
echo "=================================================="

python scripts/evaluate/evaluate_models.py \
    --model_type ${MODEL} \
    --dataset ${DATASET} \
    --pred_mode ${PRED_MODE} \
    --n_plots ${N_PLOTS} \
    --context_mode ${CONTEXT_MODE} \
    --seq_len ${SEQ_LEN} \
    --context_size ${CONTEXT_SIZE} \
    --n_context_trajs ${N_CONTEXT_TRAJS} \

echo "TabPFN Evaluation Completed!"