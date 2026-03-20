import os
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.integrate import solve_ivp
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.models import TabPFNWrapper
from src.dynamics import van_der_pol

def main():
    # 1. 路径与参数配置
    N = 10           # 预测步长 (Horizon)
    Ts = 0.1         # 采样时间
    mu = 1.0         # VdP 参数
    K_samples = 500  # 每次优化的随机采样序列数量 (大力出奇迹)
    
    results_dir = "results/van_der_pol/TabPFN/mpc"
    os.makedirs(results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 准备 TabPFN 的 Context (我们拿单条轨迹作为 Context)
    print("Preparing TabPFN Context...")
    data_dir = "data/processed/van_der_pol"
    X_train_3d = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train_3d = np.load(os.path.join(data_dir, "Y_train.npy"))
    
    # 使用第 0 条轨迹作为物理先验
    X_context = X_train_3d[0]
    Y_context = Y_train_3d[0]
    
    scaler_X = StandardScaler()
    X_context_scaled = scaler_X.fit_transform(X_context)

    # 3. 初始化 TabPFN
    print(f"Initializing TabPFN on {device}...")
    teacher = TabPFNWrapper(device=device, n_ensemble=1) # type: ignore
    
    # 随机抽取 2000 个点作为 Context
    c_size = min(2000, len(X_context_scaled))
    indices = np.random.choice(len(X_context_scaled), c_size, replace=False)
    teacher.fit(X_context_scaled[indices], Y_context[indices])

    # 4. 生成参考轨迹
    ref_seq = np.concatenate((
        np.tile([0.8], (40, 1)), np.tile([2.5], (40, 1)), 
        np.tile([-1.5], (40, 1)), np.tile([1.2], (40, 1))
    ), axis=0).flatten()

    # 5. 闭环仿真循环
    print("Starting TabPFN Sampling-based MPC simulation...")
    x_vals, u_vals, solve_times = [], [], []
    x_current = np.array([0.0, 0.0])  

    total_steps = len(ref_seq) - N
    for step in range(total_steps):
        start_time = time.perf_counter()
        
        # --- Random Shooting MPC 核心逻辑 ---
        
        # 1. 随机生成 K 组未来 N 步的控制序列, 范围 [-10, 10]
        # 形状: [K_samples, N, 1]
        U_candidates = np.random.uniform(-10.0, 10.0, size=(K_samples, N, 1))
        
        # 初始化预测状态矩阵: [K_samples, 2]
        X_pred = np.tile(x_current, (K_samples, 1))
        
        costs = np.zeros(K_samples)
        
        # 2. 纯 PyTorch 自回归滚动预测 (Auto-regressive Rollout)
        for k in range(N):
            # 构建输入特征: [u, x1, x2]
            current_U = U_candidates[:, k, :]
            X_input = np.concatenate((current_U, X_pred), axis=1)
            
            # 缩放数据并转为 Tensor
            X_input_scaled = scaler_X.transform(X_input)
            X_tensor = torch.tensor(X_input_scaled, dtype=torch.float32).to(device)
            
            # TabPFN 批量推理 (1 次计算 500 个样本的下一步)
            with torch.no_grad():
                X_next_pred = teacher(X_tensor).cpu().numpy()
            
            # 计算当前步的惩罚 (Cost)
            ref_k = ref_seq[step + k]
            error = X_next_pred[:, 0] - ref_k  # 只跟踪 x1
            
            costs += 100.0 * (error ** 2)       # Q 矩阵权重
            costs += 1.0 * (current_U[:, 0] ** 2) # R 矩阵权重
            
            # 更新状态进行下一步滚动
            X_pred = X_next_pred
            
        # 3. 选出 Cost 最小的那一条控制序列
        best_idx = np.argmin(costs)
        u_mpc = U_candidates[best_idx, 0, 0] # 只取最优序列的第一个控制动作
        
        # ------------------------------------
        
        solve_times.append(time.perf_counter() - start_time) 
        
        if step % 10 == 0:
            print(f"Step: {step}/{total_steps} | Best u: {u_mpc:.2f} | Solve time: {solve_times[-1]:.3f}s")

        # 作用于真实物理系统
        ode_sol = solve_ivp(lambda t, y: van_der_pol(t, y, mu, u_mpc), [0, Ts], x_current, method='RK45')
        x_next = ode_sol.y[:, -1] 

        # 记录数据
        x_vals.append(x_next) 
        u_vals.append(u_mpc) 
        x_current = x_next 

    # 6. 画图保存
    x_vals = np.array(x_vals)
    t_vec = np.arange(total_steps) * Ts

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t_vec, x_vals[:, 0], label='x_1 (Controlled)')
    plt.plot(t_vec, ref_seq[:total_steps], label='Reference', linestyle='--')
    plt.legend(); plt.grid()
    plt.subplot(2, 1, 2)
    plt.step(t_vec, u_vals, label='u', where='post')
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'tabpfn_random_shooting_mpc.png'))
    print(f"Average solve time: {np.mean(solve_times):.4f}s")

if __name__ == "__main__":
    main()