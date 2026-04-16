#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# 参数设置区域
MODEL="LSTM"
DATASET="four_tank"      # 选项: van_der_pol, four_tank
EVAL_SET="test"            # 要评估的数据集 (验证集还是测试集 val or test)
PRED_MODE="single_step"        # 选项: single_step, rolling
N_PLOTS=50                 # 绘图数 (0~50)
N_TRAIN=1                  # 训练轨迹数 (1~50)
MODEL_PATH="checkpoints/LSTM/${DATASET}/real_standard_N${N_TRAIN}/model_LSTM_L1_H80.pth"


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
    --eval_set ${EVAL_SET} \

echo "${MODEL} Evaluation Completed!"