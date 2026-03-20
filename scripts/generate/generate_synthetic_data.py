import os
import sys
import torch
import numpy as np
import time
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.models import TabPFNWrapper

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Synthetic Data using TabPFN")
    parser.add_argument("--method", type=str, required=True, choices=["surrogate", "point_cloud"],
                        help="surrogate: 串行生成49条连续轨迹 (适合LSTM) | point_cloud: 随机生成10万个独立单步映射 (适合MLP)")
    parser.add_argument("--n_points", type=int, default=100000, help="点云模式下生成的总散点数")
    parser.add_argument("--batch_size", type=int, default=2000, help="TabPFN 批量推理的大小 (防 OOM)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | Method: {args.method.upper()}")
    
    # 路径准备
    data_dir = "data/processed/van_der_pol"
    save_dir = os.path.join(data_dir, "synthetic")
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载真实的 3D 原始数据
    print("Loading true physical data...")
    X_train_true = np.load(os.path.join(data_dir, "X_train.npy"))  # [50, 2000, 3] 
    Y_train_true = np.load(os.path.join(data_dir, "Y_train.npy"))  # [50, 2000, 2] 
    
    T_steps = X_train_true.shape[1]
    
    # 核心：仅使用 Trajectory 0 作为 context
    print("\nSetting up Teacher with ONLY Trajectory 0 as Context...")
    X_context = X_train_true[0]  # [2000, 3]
    Y_context = Y_train_true[0]  # [2000, 2]
    
    scaler_X = StandardScaler()
    X_context_scaled = scaler_X.fit_transform(X_context)
    
    teacher = TabPFNWrapper(device=device, n_ensemble=3) # type: ignore
    c_size = min(2048, len(X_context_scaled))
    teacher.fit(X_context_scaled[:c_size], Y_context[:c_size])

    start_time = time.time()
    
    # 分支 A：Surrogate (串行连续轨迹) -> 输出 3D Tensor [50, 2000, Features]
    if args.method == "surrogate":
        print("\n[Surrogate Mode] Paralleling autoregressive rollout for 49 trajectories...")
        X_syn = np.zeros_like(X_train_true)
        Y_syn = np.zeros_like(Y_train_true)
        
        # 存入第 0 条真实轨迹
        X_syn[0] = X_train_true[0]
        Y_syn[0] = Y_train_true[0]
        
        # 提取真实初始状态和控制序列
        curr_states = X_train_true[1:, 0, 1:3]  # [49, 2]
        
        with torch.no_grad():
            for t in tqdm(range(T_steps), desc="Rollout Steps"):
                u_t = X_train_true[1:, t, 0:1]  # [49, 1]
                X_batch = np.concatenate([u_t, curr_states], axis=1)  # [49, 3]
                
                X_batch_scaled = scaler_X.transform(X_batch)
                X_tensor = torch.tensor(X_batch_scaled, dtype=torch.float32).to(device)
                
                pred_next_states = teacher(X_tensor).cpu().numpy()  # [49, 2]
                
                X_syn[1:, t, :] = X_batch
                Y_syn[1:, t, :] = pred_next_states
                curr_states = pred_next_states
                
                if np.isnan(curr_states).any() or np.max(np.abs(curr_states)) > 100.0:
                    print(f"\n[WARNING] Trajectory exploded at step {t}!")
                    break

    # 分支 B：Point Cloud (随机独立撒网) -> 输出 2D Tensor [N_points, Features]
    elif args.method == "point_cloud":
        print(f"\n[Point Cloud Mode] Generating {args.n_points} independent random states...")
        # 在更大的相空间内随机撒网，强迫模型外推
        u_rand = np.random.uniform(-10.0, 10.0, size=(args.n_points, 1))
        x1_rand = np.random.uniform(-5.0, 5.0, size=(args.n_points, 1))
        x2_rand = np.random.uniform(-5.0, 5.0, size=(args.n_points, 1))
        
        X_syn = np.concatenate([u_rand, x1_rand, x2_rand], axis=1)  # [100000, 3]
        Y_syn = np.zeros((args.n_points, 2))  # [100000, 2]
        
        with torch.no_grad():
            # 分批次推演以防止 GPU OOM
            for i in tqdm(range(0, args.n_points, args.batch_size), desc="Batch Inferencing"):
                X_batch = X_syn[i : i + args.batch_size]
                X_batch_scaled = scaler_X.transform(X_batch)
                X_tensor = torch.tensor(X_batch_scaled, dtype=torch.float32).to(device)
                
                pred_next_states = teacher(X_tensor).cpu().numpy()
                Y_syn[i : i + args.batch_size] = pred_next_states

    # 保存
    elapsed = time.time() - start_time
    print(f"\n{args.method.upper()} generation completed in {elapsed:.2f} seconds!")
    
    # 在文件名中注明是 surrogate 还是 point_cloud
    x_save_path = os.path.join(save_dir, f"X_train_{args.method}.npy")
    y_save_path = os.path.join(save_dir, f"Y_train_{args.method}.npy")
    np.save(x_save_path, X_syn)
    np.save(y_save_path, Y_syn)
    
    print(f"Data saved to {save_dir}")
    print(f" - {os.path.basename(x_save_path)}")
    print(f" - {os.path.basename(y_save_path)}")
    print(f"Shape of Synthetic X: {X_syn.shape}")
    print(f"Shape of Synthetic Y: {Y_syn.shape}")

if __name__ == "__main__":
    main()