import os
import sys
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from src.models import TabPFNWrapper

def create_sliding_windows_continuous(X_train, X_val, Y_train, Y_val, seq_len):
    """
    将连续的时序数据合并后生成滑动窗口，再按原比例完美切分。
    保证验证集的样本数与原始单步模型的验证集样本数完全一致。
    """
    # 1. 物理上拼接连续的时间序列
    X_total = np.concatenate((X_train, X_val), axis=0)
    Y_total = np.concatenate((Y_train, Y_val), axis=0)
    
    X_seq_all, Y_target_all = [], []
    
    # 2. 全局滑动窗口生成
    for i in range(len(X_total) - seq_len + 1):
        window_x = X_total[i : i + seq_len].flatten()
        X_seq_all.append(window_x)
        Y_target_all.append(Y_total[i + seq_len - 1])
        
    X_seq_all = np.array(X_seq_all)
    Y_target_all = np.array(Y_target_all)
    
    # 3. 完美切分：最后 len(X_val) 个窗口属于验证集，前面全部属于训练集
    num_val = len(X_val)
    
    X_train_seq = X_seq_all[:-num_val]
    Y_train_seq = Y_target_all[:-num_val]
    
    X_val_seq = X_seq_all[-num_val:]
    Y_val_seq = Y_target_all[-num_val:]
    
    return X_train_seq, Y_train_seq, X_val_seq, Y_val_seq

def main():
    # 1. 路径与超参数设置
    dataset_name = "van_der_pol"
    data_dir = f"data/processed/{dataset_name}"
    results_dir = f"results/{dataset_name}/TabPFN_eval"
    os.makedirs(results_dir, exist_ok=True)
    
    seq_len = 20  # 滑动窗口长度

    # 2. 加载单步原始数据
    print("Loading raw data...")
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train_raw = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val_raw = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val_raw = np.load(os.path.join(data_dir, "Y_val.npy"))

    # 3. 构造时序滑动窗口 (让 TabPFN 看见历史)
    print(f"Creating sliding windows (seq_len={seq_len})...")
    X_train_seq, Y_train_seq, X_val_seq, Y_val_seq = create_sliding_windows_continuous(
            X_train_raw, X_val_raw, Y_train_raw, Y_val_raw, seq_len
        )

    print(f"Train Sequence Shape: {X_train_seq.shape}")
    print(f"Val Sequence Shape: {X_val_seq.shape}")

    # 4. 数据标准化
    print("\nFitting StandardScaler...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train_seq)
    X_val_scaled = scaler_X.transform(X_val_seq)

    # 5. 准备 TabPFN Context (抽取最多 1024 个窗口样本)
    context_size = min(1024, len(X_train_scaled))
    np.random.seed(42)
    indices = np.random.choice(len(X_train_scaled), context_size, replace=False)
    
    X_context = X_train_scaled[indices]
    Y_context = Y_train_seq[indices]
    
    # 6. 初始化并拟合 TabPFN
    n_ensemble = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing TabPFN on {device}...")
    teacher = TabPFNWrapper(device=device, n_ensemble=n_ensemble) # type: ignore
    
    print(f"Fitting TabPFN with {context_size} historical context windows...")
    teacher.fit(X_context, Y_context)

    # 7. 在验证集上进行时序推理
    print("Predicting on validation set...")
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        Y_pred_tensor = teacher(X_val_tensor)
        Y_pred = Y_pred_tensor.cpu().numpy()

    # 8. 计算评估指标
    mse = np.mean((Y_pred - Y_val_seq) ** 2)
    print(f"\n>>> Validation MSE: {mse:.6f} <<<")
    
    # 保存实验结果
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(project_root, "forecast_results.txt")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"[{current_time}] "
        f"Dataset: {dataset_name} | "
        f"Model: TabPFN (ensemble={n_ensemble}) | "
        f"Seq_len: {seq_len} | "
        f"Context_size: {context_size} | "
        f"Val_MSE: {mse:.6f}\n"
    )
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message)
    print(f"Experiment log appended to {log_file_path}")
    
    # 9. 绘图对比
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(Y_val_seq[:, 0], label='True $x_1$ (solve_ivp)', color='black', linewidth=1.5)
    plt.plot(Y_pred[:, 0], label='TabPFN Predicted $x_1$', color='red', linestyle='--', linewidth=1.5)
    plt.ylabel('$x_1$')
    plt.title(f'TabPFN Validation Prediction (Seq Len = {seq_len})')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(Y_val_seq[:, 1], label='True $x_2$ (solve_ivp)', color='black', linewidth=1.5)
    plt.plot(Y_pred[:, 1], label='TabPFN Predicted $x_2$', color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Validation Samples (Time Steps)')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "tabpfn_validation_fit_seq.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()