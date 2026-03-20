import os
import time
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath("."))
import src.models
from src.dynamics import van_der_pol
from src.mpc.casadi_utils import extract_lstm_weights_ca
from src.mpc.mpc_solver import LSTM_MPCSolver

def main():
    # 1. 路径与参数配置
    idx_opt = 1
    n_hidden = 20 
    N = 10       
    Ts = 0.1     
    mu = 1.0     
    nx, ny, nu = 2, 1, 1 

    # ckpt_path = f"checkpoints/LSTM/van_der_pol/standard/model_LSTM_idx_{idx_opt}.pth"
    # ckpt_path = f"checkpoints/LSTM/van_der_pol/distill/model_LSTM_idx_{idx_opt}.pth"
    ckpt_path = f"checkpoints/LSTM/van_der_pol/surrogate_standard/model_LSTM_idx_{idx_opt}.pth"
    
    results_dir = "results/van_der_pol/LSTM/mpc"
    os.makedirs(results_dir, exist_ok=True)

    # 2. 加载模型并提取 CasADi 权重
    print(f"Loading model from {ckpt_path}...")
    sys.modules['models'] = src.models
    model = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    weights = extract_lstm_weights_ca(model)

    # 3. 配置 MPC 代价矩阵并实例化求解器
    Q = np.eye(ny) * 100
    R = np.eye(nu)
    
    print("Initializing LSTM-MPC Solver...")
    mpc = LSTM_MPCSolver(nx, ny, nu, N, Q, R, n_hidden, weights, output_indices=[0])
    mpc.opti.subject_to(mpc.u >= -10.0)
    mpc.opti.subject_to(mpc.u <= 10.0)
    
    # 4. 生成参考轨迹
    ref_seq = np.concatenate((
        np.tile([0.8], (40, 1)), np.tile([2.5], (40, 1)), 
        np.tile([-1.5], (40, 1)), np.tile([1.2], (40, 1))
    ), axis=0).flatten().reshape(-1, 1)

    # 5. 闭环仿真循环
    print("Starting closed-loop MPC simulation...")
    x_vals, u_vals, solve_times = [], [], []

    # 初始状态
    h_current = np.zeros(n_hidden)
    c_current = np.zeros(n_hidden)
    x_current = np.array([0.0, 0.0])  
    u_mpc = 0 

    total_steps = len(ref_seq) - N
    for i in range(total_steps):
        if i % 10 == 0:
            print(f"Simulation step: {i} / {total_steps}")
            
        # 设置 CasADi 参数
        mpc.opti.set_value(mpc.h0, h_current) 
        mpc.opti.set_value(mpc.c0, c_current) 
        mpc.opti.set_value(mpc.x0, x_current) 
        mpc.opti.set_value(mpc.u0, u_mpc) 
        mpc.opti.set_value(mpc.ref, ref_seq[i:i+N, :].T) 

        # 求解
        start_time = time.perf_counter()
        sol = mpc.solve()
        solve_times.append(time.perf_counter() - start_time) 

        # 提取当前控制输入和隐藏状态
        u_mpc = sol.value(mpc.u[0]) 
        h_current = sol.value(mpc.h[:, 1])
        c_current = sol.value(mpc.c[:, 1])
        
        # 作用于物理系统
        ode_sol = solve_ivp(lambda t, y: van_der_pol(t, y, mu, u_mpc), [0, Ts], x_current, method='RK45')
        x_next = ode_sol.y[:, -1] 

        # 记录数据
        x_vals.append(x_next) 
        u_vals.append(u_mpc) 
        x_current = x_next 

    # 6. 保存数据和绘图 (此处逻辑与上一版本完全相同，为保持排版简洁略作缩减)
    x_vals, u_vals = np.array(x_vals), np.array(u_vals)
    t_vec = np.arange(total_steps) * Ts

    data_to_save = {
        "x_vals_LSTM": x_vals, "u_vals_LSTM": u_vals,
        "solve_times_LSTM": np.array(solve_times), "xref_traj_LSTM": ref_seq    
    }
    savemat(os.path.join(results_dir, 'LSTM_vdp_mpc_results.mat'), data_to_save)
    
    # 画图
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_vec, x_vals[:, 0], label='x_1 (Controlled)')
    plt.plot(t_vec, ref_seq[:total_steps], label='Reference', linestyle='--')
    plt.legend(); plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(t_vec, x_vals[:, 1], label='x_2')
    plt.legend(); plt.grid()
    plt.subplot(3, 1, 3)
    plt.step(t_vec, u_vals, label='u', where='post')
    plt.legend(); plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'closed_loop_response.png'))
    print("Simulation complete. Results and plots saved.")

if __name__ == "__main__":
    main()