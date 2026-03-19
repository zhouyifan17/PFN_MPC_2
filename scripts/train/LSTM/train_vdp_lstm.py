import os
import torch
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.append(os.path.abspath("."))
from src.models import LSTMModel
from src.engine.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train LSTM with or without Distillation")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "distill"], help="Training mode: 'standard' or 'distill'")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length used for teacher labels (to align data)")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="Weight of the distillation loss")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 路径配置
    data_dir = "data/processed/van_der_pol"
    mode_str = "distill" if args.mode == "distill" else "standard"
    ckpt_dir = f"checkpoints/LSTM/van_der_pol/{mode_str}"
    results_dir = f"results/van_der_pol/LSTM/{mode_str}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载基础数据
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(0).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(0).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 如果是蒸馏模式，加载老师标签并对齐数据
    t_pred, t_quant, t_pred_t, t_quant_t = None, None, None, None
    if args.mode == "distill":
        distill_dir = os.path.join(data_dir, "distill")
        t_pred = np.load(os.path.join(distill_dir, f"teacher_preds_seq{args.seq_len}.npy"))
        t_quant = np.load(os.path.join(distill_dir, f"teacher_quantiles_seq{args.seq_len}.npy"))
        t_pred_t = torch.tensor(t_pred, dtype=torch.float32).unsqueeze(0).to(device)
        t_quant_t = torch.tensor(t_quant, dtype=torch.float32).unsqueeze(0).to(device)
        # 数据对齐裁剪
        crop_len = len(X_train) - len(t_pred)
        X_train = X_train[crop_len:]
        Y_train = Y_train[crop_len:]
        print(f"Data aligned. Cropped first {crop_len} steps from train set.")

    input_size = X_train_t.shape[2]
    output_size = Y_train_t.shape[2]

    # 3. 实验配置
    combinations = [20] # Hidden dimensions to test
    N_epoch = 40000
    ablation_results = []

    # 4. 循环训练
    for idx_i, n_hidden in enumerate(combinations, start=1):
        print(f"\n--- Training LSTM (Hidden: {n_hidden}) ---")
        
        # 初始化模型与优化器
        model = LSTMModel(input_size, n_hidden, output_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

        # 实例化训练器
        trainer = Trainer(model, device, ckpt_dir, model_name=f"model_LSTM_idx_{idx_i}.pth")

        exp_info = {
            "Dataset": "van_der_pol",
            "Model": "LSTM",
            "Hidden_Size": n_hidden,
            "Epochs": N_epoch
        }
        
        # 启动训练
        train_loss, val_loss, t_elapsed = trainer.train(
            X_train_t, Y_train_t, X_val_t, Y_val_t, 
            optimizer, scheduler, num_epochs=N_epoch, log_interval=2000,
            experiment_info=exp_info,
            t_pred=t_pred_t, t_quant=t_quant_t, distill_weight=args.distill_weight
        )

        # 记录结果并保存 loss numpy 文件
        np.save(os.path.join(results_dir, f"train_loss_idx_{idx_i}.npy"), train_loss)
        np.save(os.path.join(results_dir, f"val_loss_idx_{idx_i}.npy"), val_loss)

        ablation_results.append({
            'idx': idx_i,
            'hidden_state_dimension': n_hidden,
            'train_loss': np.array(train_loss),
            'val_loss': np.array(val_loss),
            'T_train': t_elapsed
        })

    # 5. 保存 ablation 结果给 MATLAB 使用
    ablation_mat = {f'result_{i}': res for i, res in enumerate(ablation_results)}
    savemat(os.path.join(results_dir, "ablation_results.mat"), ablation_mat)
    print("All experiments finished and saved!")

if __name__ == "__main__":
    main()