import os
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

sys.path.append(os.path.abspath("."))
from src.models import LSTMModel
from src.engine.trainer import Trainer

def main():
    # 1. 路径配置
    data_dir = "data/processed/van_der_pol"
    ckpt_dir = "checkpoints/LSTM/van_der_pol"
    results_dir = "results/van_der_pol/LSTM"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 从本地加载预先生成的数据
    print("Loading data from local storage...")
    X_train = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 转换为 PyTorch Tensor (注意: LSTM 期待的输入 shape 为 [batch, seq_len, feature])
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(0).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).unsqueeze(0).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).unsqueeze(0).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).unsqueeze(0).to(device)

    input_size = X_train_t.shape[2]
    output_size = Y_train_t.shape[2]

    # 3. 实验配置
    combinations = [20] # Hidden dimensions to test
    N_epoch = 40000
    ablation_results = []

    # 4. 循环训练不同超参数的模型
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
            experiment_info=exp_info
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