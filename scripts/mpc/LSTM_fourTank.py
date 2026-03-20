import os
import time
import sys
import csv
import datetime
import argparse
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
from src.utils.metrics import calculate_tracking_metrics, calculate_control_smoothness, detect_reference_phases

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM-MPC on Four-Tank")
    parser.add_argument("--data_source", type=str, default="real", choices=["real", "surrogate"])
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "distill"])
    parser.add_argument("--n_train", type=int, default=50, help="训练时使用的轨迹数量")
    parser.add_argument("--idx_opt", type=int, default=1, help="模型索引")
    parser.add_argument("--horizon", type=int, default=20, help="MPC预测时域 N")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 动态路径与参数配置
    n_hidden = 20 
    N = args.horizon
    Ts = 5.0     
    nx, ny, nu = 4, 4, 2 
    params = {'a1': 1.31e-4, 'a2': 1.51e-4, 'a3': 9.27e-5, 'a4': 8.82e-5, 
              'gamma_a': 0.3, 'gamma_b': 0.4, 'g': 9.81, 'Sc': 0.06}

    exp_prefix = f"LSTM_four_tank_N{args.n_train}_{args.data_source}_{args.mode}_H{N}"
    
    ckpt_path = f"checkpoints/LSTM/four_tank/{args.data_source}_{args.mode}_N{args.n_train}/model_LSTM_idx_{args.idx_opt}.pth"
    results_dir = f"results/four_tank/LSTM/mpc"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading model from {ckpt_path}...")
    sys.modules['models'] = src.models
    model = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    weights = extract_lstm_weights_ca(model)
    

    Q = np.eye(ny) * 100
    R = np.eye(nu)
    mpc = LSTM_MPCSolver(nx, ny, nu, N, Q, R, n_hidden, weights, output_indices=[0, 1, 2, 3])
    # 控制约束
    mpc.opti.subject_to(mpc.opti.bounded(0.0, mpc.u, 4.0)) # type: ignore
    
    # 生成参考轨迹
    ref1 = np.tile([0.650, 0.650, 0.652, 0.664], (200, 1))
    ref2 = np.tile([0.300, 0.300, 0.301, 0.306], (200, 1))
    ref3 = np.tile([0.500, 0.750, 0.305, 1.200], (200, 1))
    ref4 = np.tile([0.900, 0.750, 1.062, 0.579], (200, 1))
    xref_traj = np.concatenate((ref1, ref2, ref3, ref4), axis=0)  # Shape (800, 4)

    # 闭环仿真循环
    print(f"Starting closed-loop MPC simulation for {exp_prefix}...")
    x_vals, u_vals, solve_times = [], [], []

    h_current, c_current = np.zeros(n_hidden), np.zeros(n_hidden)
    x_current = np.array([1.0, 1.0, 0.0, 0.0])  
    u_mpc = np.array([0.0, 0.0]).reshape(2, 1)

    total_steps = len(xref_traj) - N
    for i in range(total_steps):
        if i % 10 == 0:
            print(f"Simulation step: {i} / {total_steps}")
            
        mpc.opti.set_value(mpc.h0, h_current) 
        mpc.opti.set_value(mpc.c0, c_current) 
        mpc.opti.set_value(mpc.x0, x_current) 
        mpc.opti.set_value(mpc.u0, u_mpc) 
        mpc.opti.set_value(mpc.ref, xref_traj[i:i+N, :].T) 

        start_time = time.perf_counter()
        try:
            sol = mpc.solve()
            solve_times.append(time.perf_counter() - start_time) 
            
            u_mpc = np.array([sol.value(mpc.u[0, 0]), sol.value(mpc.u[1, 0])]).reshape(2, 1)
            h_current = sol.value(mpc.h[:, 1])
            c_current = sol.value(mpc.c[:, 1])
        except Exception as e:
            print(f"\n[ERROR] Solver failed at step {i}: {e}")
            break

        ode_sol = solve_ivp(lambda t, y: four_tank(t, y, u_mpc[0,0], u_mpc[1,0], params), [0, Ts], x_current, method='RK45')
        x_next = ode_sol.y[:, -1] 

        x_vals.append(x_next) 
        u_vals.append([u_mpc[0,0], u_mpc[1,0]]) 
        x_current = x_next 

    # 指标计算与保存
    x_vals, u_vals = np.array(x_vals), np.array(u_vals)
    xref_actual = xref_traj[:len(x_vals)]
    t_vec = np.arange(len(x_vals)) * Ts

    print(f"控制性能评估 - {exp_prefix}")
    phases = detect_reference_phases(xref_actual)
    delta_u = calculate_control_smoothness(u_vals)
    csv_path = os.path.join(results_dir, "control_results.csv")
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Experiment", "Phase", 
                "RMSE_x1", "OS_x1(%)", "RMSE_x2", "OS_x2(%)",
                "RMSE_x3", "OS_x3(%)", "RMSE_x4", "OS_x4(%)",
                "Delta_u1", "Delta_u2", "Mean_Solve_Time(s)"
            ])
            
        overall_rmse, overall_os = [[] for _ in range(nx)], [[] for _ in range(nx)]

        for start, end, name in phases:
            curr_x, curr_ref = x_vals[start:end, :], xref_actual[start:end, :]
            
            p_rmse, p_os = [], []
            for k in range(nx):
                # 核心 3：一行代码算出单个维度的所有跟踪指标
                _, _, rmse, os_val = calculate_tracking_metrics(curr_x[:, k], curr_ref[:, k])
                p_rmse.append(rmse)
                p_os.append(os_val)
                overall_rmse[k].append(rmse)
                overall_os[k].append(os_val)
                
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                exp_prefix, name,
                f"{p_rmse[0]:.4f}", f"{p_os[0]:.2f}", f"{p_rmse[1]:.4f}", f"{p_os[1]:.2f}",
                f"{p_rmse[2]:.4f}", f"{p_os[2]:.2f}", f"{p_rmse[3]:.4f}", f"{p_os[3]:.2f}",
                "-", "-", "-"
            ])
            print(f"{name} (Steps {start}-{end}): Max Overshoot = {max(p_os):.2f}%")

        # 记录全局汇总
        avg_solve = np.mean(solve_times)
        final_rmse = [np.mean(overall_rmse[k]) for k in range(nx)]
        final_os = [np.max(overall_os[k]) for k in range(nx)]

        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exp_prefix, "OVERALL",
            f"{final_rmse[0]:.4f}", f"{final_os[0]:.2f}", f"{final_rmse[1]:.4f}", f"{final_os[1]:.2f}",
            f"{final_rmse[2]:.4f}", f"{final_os[2]:.2f}", f"{final_rmse[3]:.4f}", f"{final_os[3]:.2f}",
            f"{delta_u[0]:.4f}", f"{delta_u[1]:.4f}", f"{avg_solve:.4f}"
        ])
        print(f"OVERALL | Solve Time: {avg_solve:.4f}s | Delta_u1: {delta_u[0]:.4f} | Delta_u2: {delta_u[1]:.4f}")
        
        
    # 保存数据和绘图
    data_to_save = {
        "x_vals": x_vals, "u_vals": u_vals, "t_vec": t_vec,
        "solve_times": np.array(solve_times), "xref_traj": xref_actual    
    }
    mat_path = os.path.join(results_dir, f"{exp_prefix}.mat")
    savemat(mat_path, data_to_save)
    print(f"\nSimulation data saved to: {mat_path}")
    
    plt.figure(figsize=(15, 15))
    for k in range(4):
        plt.subplot(6, 1, k+1)
        plt.plot(t_vec, x_vals[:, k], label=f'x_{k+1}')
        plt.plot(t_vec, xref_actual[:, k], label='Reference', linestyle='--')
        plt.ylabel(f"x_{k+1}"); plt.legend(); plt.grid()

    plt.subplot(6, 1, 5)
    plt.step(t_vec, u_vals[:, 0], label='u_1', where='post')
    plt.ylabel("u_1"); plt.legend(); plt.grid()
    
    plt.subplot(6, 1, 6)
    plt.step(t_vec, u_vals[:, 1], label='u_2', where='post')
    plt.xlabel("Time (s)"); plt.ylabel("u_2"); plt.legend(); plt.grid()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{exp_prefix}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()