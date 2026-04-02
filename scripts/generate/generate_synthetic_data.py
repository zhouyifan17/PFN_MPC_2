import os
import sys
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.models import TabPFNWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Data using TabPFN")
    parser.add_argument("--dataset", type=str, default="van_der_pol", help="数据集名称 (van_der_pol, four_tank, pendulum)")
    parser.add_argument("--method", type=str, required=True, choices=["surrogate", "point_cloud"],
                        help="surrogate: 串行生成49条连续轨迹 | point_cloud: 随机生成10万个独立单步映射")
    parser.add_argument("--n_points", type=int, default=100000, help="点云模式下生成的总散点数")
    parser.add_argument("--batch_size", type=int, default=2000, help="TabPFN 批量推理的大小")
    parser.add_argument("--n_plots", type=int, default=1, help="生成轨迹对比图的数量 (默认 1)")
    parser.add_argument("--context_size", type=int, default=8000, help="TabPFN 使用的最大样本数")
    parser.add_argument("--n_context_trajs", type=int, default=1, help="用作 Context 的真实轨迹条数 (默认 1)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Method: {args.method.upper()}")
    
    # 路径准备
    data_dir = f"data/processed/{args.dataset}"
    save_dir = os.path.join(data_dir, "synthetic")
    plot_dir = f"results/{args.dataset}/synthetic_plots"
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载真实的 3D 原始数据
    print("Loading true physical data...")
    X_train_true = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train_true = np.load(os.path.join(data_dir, "Y_train.npy"))
    
    # 获取时间步数和特征维度
    N_total_episodes = X_train_true.shape[0]
    T_steps = X_train_true.shape[1]
    nx = Y_train_true.shape[2]
    nu = X_train_true.shape[2] - nx
    
    n_ctx = min(args.n_context_trajs, N_total_episodes)
    
    # 使用前 n_ctx 条轨迹作为 context
    print(f"\nSetting up Teacher with Trajectories 0 to {n_ctx-1} ({n_ctx} total) as Context...")
    X_context = X_train_true[:n_ctx].reshape(-1, X_train_true.shape[2])
    Y_context = Y_train_true[:n_ctx].reshape(-1, Y_train_true.shape[2])
    
    scaler_X = StandardScaler()
    X_context_scaled = scaler_X.fit_transform(X_context)
    
    teacher = TabPFNWrapper(device=device, n_ensemble=3) # type: ignore
    c_size = min(args.context_size, len(X_context_scaled))
    if len(X_context_scaled) > c_size:
        indices = np.random.choice(len(X_context_scaled), c_size, replace=False)
    else:
        indices = np.arange(len(X_context_scaled))
        
    teacher.fit(X_context_scaled[:c_size], Y_context[:c_size])

    start_time = time.time()
    # 分支 A：Surrogate
    if args.method == "surrogate":
        print(f"\n[Surrogate Model rollout for remaining {N_total_episodes - n_ctx} trajectories...")
        X_syn = np.zeros_like(X_train_true)
        Y_syn = np.zeros_like(Y_train_true)
        
        # 存入前 n_ctx 条作为真实参考的轨迹
        X_syn[:n_ctx] = X_train_true[:n_ctx]
        Y_syn[:n_ctx] = Y_train_true[:n_ctx]
        
        if n_ctx < N_total_episodes:
            curr_states = X_train_true[n_ctx:, 0, nu:]
        
        with torch.no_grad():
            for t in tqdm(range(T_steps), desc="Rollout Steps"):
                u_t = X_train_true[n_ctx:, t, 0:nu]
                X_batch = np.concatenate([u_t, curr_states], axis=1)
                
                X_batch_scaled = scaler_X.transform(X_batch)
                X_tensor = torch.tensor(X_batch_scaled, dtype=torch.float32).to(device)
                
                pred_next_states = teacher(X_tensor).cpu().numpy()
                
                X_syn[n_ctx:, t, :] = X_batch
                Y_syn[n_ctx:, t, :] = pred_next_states
                curr_states = pred_next_states
                
                if np.isnan(curr_states).any() or np.max(np.abs(curr_states)) > 100.0:
                    print(f"\n[WARNING] Trajectory exploded at step {t}!")
                    break

    # 分支 B：Point Cloud
    elif args.method == "point_cloud":
        print(f"\n[Point Cloud Mode] Generating {args.n_points} independent random states...")
        # 动态分配维度，为各个维度生成合理的撒网范围
        if args.dataset == "van_der_pol":
            u_rand = np.random.uniform(-10.0, 10.0, size=(args.n_points, nu))
            x_rand = np.random.uniform(-5.0, 5.0, size=(args.n_points, nx))
        elif args.dataset == "four_tank":
            u_rand = np.random.uniform(0.0, 6.0, size=(args.n_points, nu))
            x_rand = np.random.uniform(0.0, 2.0, size=(args.n_points, nx))
        else:
            # 通用范围
            u_rand = np.random.uniform(-5.0, 5.0, size=(args.n_points, nu))
            x_rand = np.random.uniform(-5.0, 5.0, size=(args.n_points, nx))
        
        X_syn = np.concatenate([u_rand, x_rand], axis=1)
        Y_syn = np.zeros((args.n_points, nx))
        
        with torch.no_grad():
            for i in tqdm(range(0, args.n_points, args.batch_size), desc="Batch Inferencing"):
                X_batch = X_syn[i : i + args.batch_size]
                X_batch_scaled = scaler_X.transform(X_batch)
                X_tensor = torch.tensor(X_batch_scaled, dtype=torch.float32).to(device)
                
                pred_next_states = teacher(X_tensor).cpu().numpy()
                Y_syn[i : i + args.batch_size] = pred_next_states

    # 保存数据
    elapsed = time.time() - start_time
    print(f"\n{args.method.upper()} generation completed in {elapsed:.2f} seconds!")
    
    x_save_path = os.path.join(save_dir, f"X_train_{args.method}_ctx{n_ctx}.npy")
    y_save_path = os.path.join(save_dir, f"Y_train_{args.method}_ctx{n_ctx}.npy")
    np.save(x_save_path, X_syn)
    np.save(y_save_path, Y_syn)
    
    print(f"Data saved to {save_dir}")
    print(f" - {os.path.basename(x_save_path)}")
    print(f" - {os.path.basename(y_save_path)}")
    print(f"Shape of Synthetic X: {X_syn.shape}")
    print(f"Shape of Synthetic Y: {Y_syn.shape}")
    
    # 绘制原轨迹和生成轨迹的对比图
    if args.method == "surrogate" and Y_syn.shape[0] > n_ctx:
        max_plots = min(args.n_plots, Y_syn.shape[0] - n_ctx)
        print(f"\nGenerating {max_plots} trajectory comparison plots...")
        
        t_vec = np.arange(T_steps)
        
        for ep_to_plot in range(n_ctx, n_ctx + max_plots):
            plt.figure(figsize=(12, 3 * nx))
            
            for i in range(nx):
                plt.subplot(nx, 1, i + 1)
                plt.plot(t_vec, Y_train_true[ep_to_plot, :, i], label=f'True $x_{i+1}$ (ODE)', color='black', linewidth=1.5)
                plt.plot(t_vec, Y_syn[ep_to_plot, :, i], label=f'Synthetic $x_{i+1}$ (TabPFN)', color='red', linestyle='--', linewidth=1.5)
                plt.ylabel(f'$x_{i+1}$')
                
                if i == 0:
                    plt.title(f'True vs Synthetic Trajectory (Dataset: {args.dataset}, Episode: {ep_to_plot})')
                if i == nx - 1:
                    plt.xlabel('Time Steps')
                    
                plt.legend(loc='upper right')
                plt.grid(True)
                
            plt.tight_layout()
            plot_path = os.path.join(plot_dir, f"syn_comparison_{args.dataset}_ctx{n_ctx}_ep{ep_to_plot}.png")
            plt.savefig(plot_path)
            plt.close()
            
        print(f"Successfully saved {max_plots} comparison plots to: {plot_dir}")

if __name__ == "__main__":
    main()