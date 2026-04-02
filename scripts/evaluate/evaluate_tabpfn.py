import os
import sys
import torch
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
from src.models import TabPFNWrapper

def get_args():
    parser = argparse.ArgumentParser(description="Evaluate TabPFN with different Context Strategies")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="van_der_pol", help="数据集名称 (van_der_pol, four_tank)")
    parser.add_argument("--context_mode", type=str, default="mixed", 
                        choices=["single", "mixed", "independent"],
                        help="single: 1条轨迹作context; mixed: 50条混合抽样; independent: 逐条独立抽样")
    parser.add_argument("--pred_mode", type=str, default="single_step",
                        choices=["single_step", "rolling"],
                        help="single_step: 提供真实状态(单步); rolling: 滚动预测(自回归)")
    parser.add_argument("--seq_len", type=int, default=1, help="滑动窗口长度")
    parser.add_argument("--context_size", type=int, default=8000, help="TabPFN 使用的最大样本数")
    parser.add_argument("--n_plots", type=int, default=1, help="绘制预测轨迹对比图的数量")
    return parser.parse_args()

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
        
        X_train_seq_list.append(X_seq[:-num_val])
        Y_train_seq_list.append(Y_seq[:-num_val])
        X_val_seq_list.append(X_seq[-num_val:])
        Y_val_seq_list.append(Y_seq[-num_val:])

    # 返回依然是 3D 结构的张量
    return (np.array(X_train_seq_list), np.array(Y_train_seq_list), 
            np.array(X_val_seq_list), np.array(Y_val_seq_list))

def rolling_predict(teacher, X_val_3d_scaled, seq_len, nx, nu, device):
    """
    自回归滚动预测：在每一步正确注入未来的真实控制信号 (u_{t+1})。
    X_val_3d_scaled: 形状为 [N_episodes, T_steps, seq_len * (nu + nx)]
    """
    N_episodes, num_val_steps, total_flat_dim = X_val_3d_scaled.shape
    feature_dim = nu + nx  # 单个时间步的总特征维度
    
    # 获取 t=0 时刻的初始窗口
    current_windows = X_val_3d_scaled[:, 0, :].copy() 
    all_preds_scaled = []
    
    for t in tqdm(range(num_val_steps), desc="Rolling Prediction Steps"):
        curr_tensor = torch.tensor(current_windows, dtype=torch.float32).to(device)
        
        # 1. 预测当前状态的下一步 (预测出的只有 nx 维度的标准化状态)
        with torch.no_grad():
            preds_scaled = teacher(curr_tensor).cpu().numpy() # 形状: (N_episodes, nx)
        all_preds_scaled.append(preds_scaled)
        
        # 2. 滑动窗口更新
        if t < num_val_steps - 1:
            next_windows = np.empty_like(current_windows)
            
            # 如果序列长度 > 1，整体向左平移一个时间步的维度 (feature_dim)
            if seq_len > 1:
                next_windows[:, :-feature_dim] = current_windows[:, feature_dim:]
        
            true_next_u_scaled = X_val_3d_scaled[:, t + 1, -feature_dim : -feature_dim + nu]
            next_step_features = np.concatenate([true_next_u_scaled, preds_scaled], axis=1)
            next_windows[:, -feature_dim:] = next_step_features
            current_windows = next_windows
            
    return np.transpose(np.array(all_preds_scaled), (1, 0, 2))

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    
    # 1. 路径与超参数设置
    dataset_name = args.dataset
    data_dir = f"data/processed/{dataset_name}"
    exp_name = f"{args.context_mode}_{args.pred_mode}_seq{args.seq_len}"
    results_dir = os.path.join(f"results/{dataset_name}/TabPFN_eval", exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. 加载 3D 原始数据
    print(f"Loading data for {dataset_name}...")
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train_raw = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_val_raw = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val_raw = np.load(os.path.join(data_dir, "Y_val.npy"))

    N_episodes = X_train_raw.shape[0]
    
    # 3. 全局数据标准化 (核心改进：在切窗口前做)
    print("Fitting Scalers globally on raw data...")
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    # 展平原始 3D 矩阵去 fit
    X_train_flat_raw = X_train_raw.reshape(-1, X_train_raw.shape[-1])
    Y_train_flat_raw = Y_train_raw.reshape(-1, Y_train_raw.shape[-1])
    
    scaler_X.fit(X_train_flat_raw)
    scaler_Y.fit(Y_train_flat_raw)
    
    # Transform 并恢复 3D 形状
    X_train_scaled_raw = scaler_X.transform(X_train_flat_raw).reshape(X_train_raw.shape)
    X_val_scaled_raw = scaler_X.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape)
    Y_train_scaled_raw = scaler_Y.transform(Y_train_flat_raw).reshape(Y_train_raw.shape)
    Y_val_scaled_raw = scaler_Y.transform(Y_val_raw.reshape(-1, Y_val_raw.shape[-1])).reshape(Y_val_raw.shape)

    # 4. 构造时序滑动窗口 
    print(f"Creating 3D sliding windows (seq_len={args.seq_len})...")
    X_train_3d_scaled, Y_train_3d_scaled, X_val_3d_scaled, Y_val_3d_scaled = create_sliding_windows_3d(
        X_train_scaled_raw, X_val_scaled_raw, Y_train_scaled_raw, Y_val_scaled_raw, args.seq_len
    )
    _, _, _, Y_val_3d_unscaled = create_sliding_windows_3d(
        X_train_raw, X_val_raw, Y_train_raw, Y_val_raw, args.seq_len
    )

    nx = Y_val_raw.shape[-1]
    nu = X_val_raw.shape[-1] - nx
    print(f"Detected dimensions - Control (nu): {nu}, State (nx): {nx}")

    # 5. 初始化 TabPFN
    n_ensemble = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing TabPFN on {device} (Context: {args.context_mode} | Pred Mode: {args.pred_mode})...")
    teacher = TabPFNWrapper(device=device, n_ensemble=n_ensemble) # type: ignore

    np.random.seed(42)
    Y_pred_all_scaled = [] # 用于收集模型输出的标准化预测值
    
    num_val_steps = X_val_3d_scaled.shape[1]
    features_dim = Y_val_3d_scaled.shape[-1]

    # 根据不同的 context_mode 执行抽取和评估
    if args.context_mode == "mixed":
        pool_X = X_train_3d_scaled.reshape(-1, X_train_3d_scaled.shape[-1])
        pool_Y = Y_train_3d_scaled.reshape(-1, Y_train_3d_scaled.shape[-1])
        
        c_size = min(args.context_size, len(pool_X))
        indices = np.random.choice(len(pool_X), c_size, replace=False)
        
        print(f"Fitting TabPFN with {c_size} mixed context windows from all 50 episodes...")
        teacher.fit(pool_X[indices], pool_Y[indices])
        
        if args.pred_mode == "single_step":
            X_val_tensor = torch.tensor(X_val_3d_scaled.reshape(-1, X_val_3d_scaled.shape[-1]), dtype=torch.float32).to(device)
            with torch.no_grad():
                Y_pred_all_scaled = teacher(X_val_tensor).cpu().numpy()
        else:
            print("Executing rolling prediction for all episodes...")
            preds_3d = rolling_predict(teacher, X_val_3d_scaled, args.seq_len, nx, nu, device)
            Y_pred_all_scaled = preds_3d.reshape(-1, nx)

    elif args.context_mode == "single":
        pool_X = X_train_3d_scaled[0]
        pool_Y = Y_train_3d_scaled[0]
        
        c_size = min(args.context_size, len(pool_X))
        indices = np.random.choice(len(pool_X), c_size, replace=False)
        
        print(f"Fitting TabPFN with {c_size} context windows from Episode 0 ONLY...")
        teacher.fit(pool_X[indices], pool_Y[indices])
        
        if args.pred_mode == "single_step":
            X_val_tensor = torch.tensor(X_val_3d_scaled.reshape(-1, X_val_3d_scaled.shape[-1]), dtype=torch.float32).to(device)
            with torch.no_grad():
                Y_pred_all_scaled = teacher(X_val_tensor).cpu().numpy()
        else:
            print("Executing rolling prediction for all episodes...")
            preds_3d = rolling_predict(teacher, X_val_3d_scaled, args.seq_len, nx, nu, device)
            Y_pred_all_scaled = preds_3d.reshape(-1, nx)

    elif args.context_mode == "independent":
        print(f"Evaluating {N_episodes} episodes independently...")
        preds_list = []
        for ep in tqdm(range(N_episodes), desc="Independent Episodes"):
            pool_X = X_train_3d_scaled[ep]
            pool_Y = Y_train_3d_scaled[ep]
            
            c_size = min(args.context_size, len(pool_X))
            indices = np.random.choice(len(pool_X), c_size, replace=False)
            
            teacher.fit(pool_X[indices], pool_Y[indices])
            
            if args.pred_mode == "single_step":
                X_val_ep_tensor = torch.tensor(X_val_3d_scaled[ep], dtype=torch.float32).to(device)
                with torch.no_grad():
                    pred = teacher(X_val_ep_tensor).cpu().numpy()
                preds_list.append(pred)
            else:
                X_val_ep_3d_scaled = X_val_3d_scaled[ep:ep+1]
                pred_3d = rolling_predict(teacher, X_val_ep_3d_scaled, args.seq_len, nx, nu, device)
                preds_list.append(pred_3d[0])
                
        Y_pred_all_scaled = np.concatenate(preds_list, axis=0)

    # 6. 反标准化回物理空间并计算 MSE
    print("\nInverse transforming predictions back to physical space...")
    Y_pred_all_unscaled = scaler_Y.inverse_transform(Y_pred_all_scaled)
    
    # 统一展平无量纲化的真实验证集以匹配预测结果
    Y_val_flat_unscaled = Y_val_3d_unscaled.reshape(-1, Y_val_3d_unscaled.shape[-1])
    
    mse = np.mean((Y_pred_all_unscaled - Y_val_flat_unscaled) ** 2)
    print(f">>> Mode [{args.context_mode.upper()} | {args.pred_mode.upper()}] Validation MSE: {mse:.6f} <<<")
    
    # 7. 保存结果
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file_path = os.path.join(project_root, "forecast_results.txt")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = (
        f"[{current_time}] Dataset: {dataset_name} | "
        f"Model: TabPFN ({args.context_mode} | {args.pred_mode}) | "
        f"Seq_len: {args.seq_len} | "
        f"Context_size: {args.context_size} | "
        f"Val_MSE: {mse:.6f}\n"
    )
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_message)
    print(f"Experiment log appended to {log_file_path}")
    
    # 8. 绘图对比 (支持绘制多条轨迹)
    max_plots = min(args.n_plots, N_episodes)
    print(f"\nGenerating {max_plots} trajectory comparison plots...")
    
    for ep_to_plot in range(max_plots):
        start_idx = ep_to_plot * num_val_steps
        end_idx = (ep_to_plot + 1) * num_val_steps
        
        Y_pred_plot = Y_pred_all_unscaled[start_idx:end_idx]
        Y_true_plot = Y_val_3d_unscaled[ep_to_plot]

        ny = Y_true_plot.shape[1]
        
        plt.figure(figsize=(12, 3 * ny))
        
        for i in range(ny):
            plt.subplot(ny, 1, i + 1)
            plt.plot(Y_true_plot[:, i], label=f'True $x_{i+1}$ (solve_ivp)', color='black', linewidth=1.5)
            plt.plot(Y_pred_plot[:, i], label=f'TabPFN Predicted $x_{i+1}$', color='red', linestyle='--', linewidth=1.5)
            plt.ylabel(f'$x_{i+1}$')
            
            if i == 0:
                plt.title(f'TabPFN Val ({dataset_name} | {args.context_mode} | {args.pred_mode}, Ep: {ep_to_plot})')
            if i == ny - 1:
                plt.xlabel('Validation Samples (Time Steps)')
                
            plt.legend(loc='upper right')
            plt.grid(True)

        plt.tight_layout()
        plot_path = os.path.join(results_dir, f"ep_{ep_to_plot}.png")
        plt.savefig(plot_path)
        plt.close()

if __name__ == "__main__":
    main()