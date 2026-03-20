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
from src.dynamics import four_tank
from src.mpc.casadi_utils import extract_lstm_weights_ca
from src.mpc.mpc_solver import LSTM_MPCSolver

def main():
    # 1. 路径与参数配置
    idx_opt = 1
    n_hidden = 20 
    N = 20       # Four-Tank 通常需要更长的预测时域
    Ts = 5.0     # 采样时间 (由参考代码提供)
    nx, ny, nu = 4, 4, 2 

    # 物理参数
    params = {'a1': 1.31e-4, 'a2': 1.51e-4, 'a3': 9.27e-5, 'a4': 8.82e-5, 
              'gamma_a': 0.3, 'gamma_b': 0.4, 'g': 9.81, 'Sc': 0.06}

    ckpt_path = f"checkpoints/LSTM/four_tank/real_standard/model_LSTM_idx_{idx_opt}.pth"
    results_dir = "results/four_tank/LSTM/mpc"
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
    
    print("Initializing LSTM-MPC Solver for Four-Tank...")
    # 注意：Four-Tank 需要跟踪 4 个状态量，所以 output_indices=[0, 1, 2, 3]
    mpc = LSTM_MPCSolver(nx, ny, nu, N, Q, R, n_hidden, weights, output_indices=[0, 1, 2, 3])
    
    # Four-Tank 物理约束：输入不能为负，且最大为 4.0
    mpc.opti.subject_to(mpc.opti.bounded(0.0, mpc.u, 4.0)) # type: ignore
    
    # 4. 生成参考轨迹 (复刻参考代码)
    ref1 = np.tile([0.650, 0.650, 0.652, 0.664], (200, 1))
    ref2 = np.tile([0.300, 0.300, 0.301, 0.306], (200, 1))
    ref3 = np.tile([0.500, 0.750, 0.305, 1.200], (200, 1))
    ref4 = np.tile([0.900, 0.750, 1.062, 0.579], (200, 1))
    xref_traj = np.concatenate((ref1, ref2, ref3, ref4), axis=0)  # Shape (800, 4)

    # 5. 闭环仿真循环
    print("Starting closed-loop MPC simulation...")
    x_vals, u_vals, solve_times = [], [], []

    # 初始状态 (复刻参考代码的测试起点)
    h_current = np.zeros(n_hidden)
    c_current = np.zeros(n_hidden)
    x_current = np.array([1.0, 1.0, 0.0, 0.0])  
    u_mpc = np.array([0.0, 0.0]).reshape(2, 1) # 初始化 u_{k-1}

    total_steps = len(xref_traj) - N
    for i in range(total_steps):
        if i % 10 == 0:
            print(f"Simulation step: {i} / {total_steps}")
            
        # 设置 CasADi 参数
        mpc.opti.set_value(mpc.h0, h_current) 
        mpc.opti.set_value(mpc.c0, c_current) 
        mpc.opti.set_value(mpc.x0, x_current) 
        mpc.opti.set_value(mpc.u0, u_mpc) 
        mpc.opti.set_value(mpc.ref, xref_traj[i:i+N, :].T) 

        # 求解 (如果无解或抛错，通常是模型发散，我们可以捕获并打印)
        start_time = time.perf_counter()
        try:
            sol = mpc.solve()
            solve_times.append(time.perf_counter() - start_time) 
            
            # 提取当前控制输入和隐藏状态
            u1_mpc = sol.value(mpc.u[0, 0])
            u2_mpc = sol.value(mpc.u[1, 0])
            u_mpc = np.array([u1_mpc, u2_mpc]).reshape(2, 1)
            
            h_current = sol.value(mpc.h[:, 1])
            c_current = sol.value(mpc.c[:, 1])
        except Exception as e:
            print(f"\n[ERROR] Solver failed at step {i}: {e}")
            break

        # 作用于物理系统
        ode_sol = solve_ivp(lambda t, y: four_tank(t, y, u_mpc[0,0], u_mpc[1,0], params), [0, Ts], x_current, method='RK45')
        x_next = ode_sol.y[:, -1] 

        # 记录数据
        x_vals.append(x_next) 
        u_vals.append([u_mpc[0,0], u_mpc[1,0]]) 
        x_current = x_next 

    # 6. 保存数据和绘图
    x_vals = np.array(x_vals)
    u_vals = np.array(u_vals)
    t_vec = np.arange(len(x_vals)) * Ts

    data_to_save = {
        "x_vals_LSTM": x_vals, "u_vals_LSTM": u_vals,
        "solve_times_LSTM": np.array(solve_times), "xref_traj_LSTM": xref_traj    
    }
    savemat(os.path.join(results_dir, 'LSTM_four_tank_mpc_results.mat'), data_to_save)
    
    # 画图 (6 个子图)
    plt.figure(figsize=(15, 15))
    
    for k in range(4):
        plt.subplot(6, 1, k+1)
        plt.plot(t_vec, x_vals[:, k], label=f'x_{k+1}')
        plt.plot(t_vec, xref_traj[:len(x_vals), k], label='Reference', linestyle='--')
        plt.ylabel(f"x_{k+1}")
        plt.legend(); plt.grid()

    plt.subplot(6, 1, 5)
    plt.step(t_vec, u_vals[:, 0], label='u_1', where='post')
    plt.ylabel("u_1"); plt.legend(); plt.grid()
    
    plt.subplot(6, 1, 6)
    plt.step(t_vec, u_vals[:, 1], label='u_2', where='post')
    plt.xlabel("Time (s)"); plt.ylabel("u_2"); plt.legend(); plt.grid()

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'closed_loop_response_four_tank.png'))
    print("Simulation complete. Results and plots saved.")

if __name__ == "__main__":
    main()