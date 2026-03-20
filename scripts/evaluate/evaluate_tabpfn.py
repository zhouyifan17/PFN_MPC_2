import os
import sys
import torch
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from src.models import TabPFNWrapper

def create_sliding_windows_3d(X_train_3d, X_val_3d, Y_train_3d, Y_val_3d, seq_len):
    """
    处理 3D 轨迹数据 [N_episodes, T_steps, Features]
    在保证轨迹间绝对隔离的前提下，实现时间轴上的无缝切片，完美保留验证集数量。
    """
    N_episodes = X_train_3d.shape[0]
    num_val = X_val_3d.shape[1]

    X_train_seq_list, Y_train_seq_list = [], []
    X_val_seq_list, Y_val_seq_list = [], []

    for ep in range(N_episodes):
        # 针对每一条轨迹，独立地在时间轴上拼接 Train 和 Val
        X_total = np.concatenate((X_train_3d[ep], X_val_3d[ep]), axis=0)
        Y_total = np.concatenate((Y_train_3d[ep], Y_val_3d[ep]), axis=0)
        
        X_seq, Y_seq = [], []
        # 单条轨迹内部的滑动窗口
        for i in range(len(X_total) - seq_len + 1):
            X_seq.append(X_total[i : i + seq_len].flatten())
            Y_seq.append(Y_total[i + seq_len - 1])
            
        X_seq = np.array(X_seq)
        Y_seq = np.array(Y_seq)
        
        # 完美切分
        X_train_seq_list.append(X_seq[:-num_val])
        Y_train_seq_list.append(Y_seq[:-num_val])
        X_val_seq_list.append(X_seq[-num_val:])
        Y_val_seq_list.append(Y_seq[-num_val:])

    # 返回依然是 3D 结构的张量，方便后续按模式灵活抽取 Context
    return (np.array(X_train_seq_list), np.array(Y_train_seq_list), 
            np.array(X_val_seq_list), np.array(Y_val_seq_list))

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate TabPFN with different Context Strategies")
    parser.add_argument("--context_mode", type=str, default="mixed", 
                        choices=["single", "mixed", "independent"],
                        help="single: 1条轨迹作context; mixed: 50条混合抽样; independent: 逐条独立抽样")
    parser.add_argument("--seq_len", type=int, default=1, help="滑动窗口长度")
    parser.add_argument("--context_size", type=int, default=2048, help="TabPFN 使用的最大样本数")
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. 路径与超参数设置
    dataset_name = "van_der_pol"
    data_dir = f"data/processed/{dataset_name}"
    results_dir = f"results/{dataset_name}/TabPFN_eval"
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. 加载 3D 原始数据
    print("Loading 3D raw data...")
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy")) # [50, 2000, 3]
    Y_train_raw = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val_raw = np.load(os.path.join(data_dir, "X_val.npy"))     # [50, 410, 3]
    Y_val_raw = np.load(os.path.join(data_dir, "Y_val.npy"))

    N_episodes = X_train_raw.shape[0]

    # 3. 构造时序滑动窗口 (保持 3D 隔离)
    print(f"Creating 3D sliding windows (seq_len={args.seq_len})...")
    X_train_3d, Y_train_3d, X_val_3d, Y_val_3d = create_sliding_windows_3d(
        X_train_raw, X_val_raw, Y_train_raw, Y_val_raw, args.seq_len
    )

    print(f"Train Sequence 3D Shape: {X_train_3d.shape}")
    print(f"Val Sequence 3D Shape: {X_val_3d.shape}")

    # 4. 数据标准化 (在所有 50 条训练轨迹的展平数据上拟合，保证统一尺度)
    print("\nFitting StandardScaler...")
    scaler_X = StandardScaler()
    # 展平为 2D 去 fit
    X_train_flat = X_train_3d.reshape(-1, X_train_3d.shape[-1])
    scaler_X.fit(X_train_flat)
    
    # transform 后恢复 3D 形状
    X_train_scaled = scaler_X.transform(X_train_flat).reshape(X_train_3d.shape)
    X_val_scaled = scaler_X.transform(X_val_3d.reshape(-1, X_val_3d.shape[-1])).reshape(X_val_3d.shape)

    # 5. 初始化 TabPFN
    n_ensemble = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing TabPFN on {device} (Mode: {args.context_mode})...")
    teacher = TabPFNWrapper(device=device, n_ensemble=n_ensemble) # type: ignore

    np.random.seed(42)
    Y_pred_all = []

    # 根据不同的 context_mode 执行抽取和评估
    if args.context_mode == "mixed":
        # 模式一：从 50 条历史轨迹的大池子里随机抽取混合 Context
        pool_X = X_train_scaled.reshape(-1, X_train_scaled.shape[-1])
        pool_Y = Y_train_3d.reshape(-1, Y_train_3d.shape[-1])
        
        c_size = min(args.context_size, len(pool_X))
        indices = np.random.choice(len(pool_X), c_size, replace=False)
        
        print(f"Fitting TabPFN with {c_size} mixed context windows from all 50 episodes...")
        teacher.fit(pool_X[indices], pool_Y[indices])
        
        # 评估所有 50 条 Val 轨迹
        X_val_tensor = torch.tensor(X_val_scaled.reshape(-1, X_val_scaled.shape[-1]), dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred_all = teacher(X_val_tensor).cpu().numpy()

    elif args.context_mode == "single":
        # 模式二：只用第 1 条轨迹的历史作为 Context
        pool_X = X_train_scaled[0]
        pool_Y = Y_train_3d[0]
        
        c_size = min(args.context_size, len(pool_X))
        indices = np.random.choice(len(pool_X), c_size, replace=False)
        
        print(f"Fitting TabPFN with {c_size} context windows from Episode 0 ONLY...")
        teacher.fit(pool_X[indices], pool_Y[indices])
        
        # 评估所有 50 条 Val 轨迹
        X_val_tensor = torch.tensor(X_val_scaled.reshape(-1, X_val_scaled.shape[-1]), dtype=torch.float32).to(device)
        with torch.no_grad():
            Y_pred_all = teacher(X_val_tensor).cpu().numpy()

    elif args.context_mode == "independent":
        # 模式三：一条一条来，第 i 条 Val 轨迹只用第 i 条 Train 轨迹作 Context
        print(f"Evaluating {N_episodes} episodes independently...")
        preds_list = []
        for ep in range(N_episodes):
            pool_X = X_train_scaled[ep]
            pool_Y = Y_train_3d[ep]
            
            c_size = min(args.context_size, len(pool_X))
            indices = np.random.choice(len(pool_X), c_size, replace=False)
            
            # 每次重新 fit 覆盖 Context
            teacher.fit(pool_X[indices], pool_Y[indices])
            
            X_val_ep_tensor = torch.tensor(X_val_scaled[ep], dtype=torch.float32).to(device)
            with torch.no_grad():
                pred = teacher(X_val_ep_tensor).cpu().numpy()
            preds_list.append(pred)
            
            if (ep + 1) % 10 == 0:
                print(f"  Processed {ep + 1}/{N_episodes} episodes...")
                
        Y_pred_all = np.concatenate(preds_list, axis=0)


    # 计算评估指标 (统一展平后计算 MSE)
    Y_val_flat = Y_val_3d.reshape(-1, Y_val_3d.shape[-1])
    mse = np.mean((Y_pred_all - Y_val_flat) ** 2)
    print(f"\n>>> Mode [{args.context_mode.upper()}] Validation MSE: {mse:.6f} <<<")
    
    # 保存结果
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_file_path = os.path.join(project_root, "forecast_results.txt")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"[{current_time}] Dataset: {dataset_name} | "
        f"Model: TabPFN ({args.context_mode}) | "
        f"Seq_len: {args.seq_len} | "
        f"Context_size: {args.context_size} | "
        f"Val_MSE: {mse:.6f}\n"
    )
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message)
    print(f"Experiment log appended to {log_file_path}")
    
    # 绘图对比
    ep_to_plot = 0
    num_val_steps = X_val_3d.shape[1]
    
    # 从扁平化的预测结果中切出第一条轨迹的预测值
    Y_pred_plot = Y_pred_all[:num_val_steps]
    Y_true_plot = Y_val_3d[ep_to_plot]

    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(Y_true_plot[:, 0], label='True $x_1$ (solve_ivp)', color='black', linewidth=1.5)
    plt.plot(Y_pred_plot[:, 0], label='TabPFN Predicted $x_1$', color='red', linestyle='--', linewidth=1.5) # type: ignore
    plt.ylabel('$x_1$')
    plt.title(f'TabPFN Validation (Mode: {args.context_mode}, Ep: {ep_to_plot})')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(Y_true_plot[:, 1], label='True $x_2$ (solve_ivp)', color='black', linewidth=1.5)
    plt.plot(Y_pred_plot[:, 1], label='TabPFN Predicted $x_2$', color='red', linestyle='--', linewidth=1.5) # type: ignore
    plt.xlabel('Validation Samples (Time Steps)')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"tabpfn_val_{args.context_mode}_seq{args.seq_len}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    main()