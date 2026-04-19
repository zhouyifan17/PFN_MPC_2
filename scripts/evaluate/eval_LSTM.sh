#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# 参数设置区域
MODEL="LSTM"
DATASET="van_der_pol"      # 选项: van_der_pol, four_tank
PRED_MODE="rolling"        # 选项: single_step, rolling
TRAIN_MODE="standard"       # 选项: standard, distill
EVAL_SET="test"            # 要评估的数据集 (验证集还是测试集 val or test)
N_PLOTS=0                 # 绘图数 (0~50)
N_TRAIN=1                  # 训练轨迹数 (1~50)
N_HIDDEN=20                # 隐藏单元数

# 执行区
echo "=================================================="
echo "Starting Evaluation for: ${MODEL}"
echo "Dataset  : ${DATASET}"
echo "Mode     : ${PRED_MODE}"
echo "=================================================="

MODEL_PATH="checkpoints/${MODEL}/${DATASET}/real_${TRAIN_MODE}_N${N_TRAIN}/model_${MODEL}_L1_H${N_HIDDEN}.pth"

python src/evaluate/evaluate_models.py \
    --model_type ${MODEL} \
    --dataset ${DATASET} \
    --pred_mode ${PRED_MODE} \
    --n_plots ${N_PLOTS} \
    --model_path ${MODEL_PATH} \
    --eval_set ${EVAL_SET} \

# # 搜索蒸馏训练结果的代码
# for DISTILL_WEIGHT in $(seq 0.1 0.1 1.0); do
#     MODEL_PATH="checkpoints/${MODEL}/${DATASET}/real_${TRAIN_MODE}_N${N_TRAIN}/model_${MODEL}_L1_H${N_HIDDEN}_W${DISTILL_WEIGHT}.pth"

#     python src/evaluate/evaluate_models.py \
#         --model_type ${MODEL} \
#         --dataset ${DATASET} \
#         --pred_mode ${PRED_MODE} \
#         --n_plots ${N_PLOTS} \
#         --model_path ${MODEL_PATH} \
#         --eval_set ${EVAL_SET} \

# done

echo "${MODEL} Evaluation Completed!"