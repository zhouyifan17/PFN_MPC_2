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
from src.dynamics import van_der_pol
from src.mpc.casadi_utils import extract_lstm_weights_ca
from src.mpc.mpc_solver import LSTM_MPCSolver
from src.utils.metrics import calculate_tracking_metrics, calculate_control_smoothness, detect_reference_phases

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM-MPC on Van der Pol")
    parser.add_argument("--data_source", type=str, default="surrogate", choices=["real", "surrogate"])
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "distill"])
    parser.add_argument("--n_train", type=int, default=50, help="训练时使用的轨迹数量")
    parser.add_argument("--idx_opt", type=int, default=1, help="模型索引")
    parser.add_argument("--horizon", type=int, default=10, help="MPC预测时域 N")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 动态路径与参数配置
    n_hidden = 20 
    N = args.horizon       
    Ts = 0.1     
    mu = 1.0     
    nx, ny, nu = 2, 1, 1 

    exp_prefix = f"LSTM_vdp_N{args.n_train}_{args.data_source}_{args.mode}_H{N}"
    
    ckpt_path = f"checkpoints/LSTM/van_der_pol/{args.data_source}_{args.mode}/model_LSTM_idx_{args.idx_opt}.pth"
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
    
    print(f"Initializing LSTM-MPC Solver for {exp_prefix}...")
    mpc = LSTM_MPCSolver(nx, ny, nu, N, Q, R, n_hidden, weights, output_indices=[0])
    # 输入约束
    mpc.opti.subject_to(mpc.opti.bounded(-10.0, mpc.u, 10.0)) # type: ignore
    
    # 4. 生成参考轨迹
    ref_seq = np.concatenate((
        np.tile([0.8], (40, 1)), np.tile([2.5], (40, 1)), 
        np.tile([-1.5], (40, 1)), np.tile([1.2], (40, 1))
    ), axis=0).flatten().reshape(-1, 1)

    # 5. 闭环仿真循环
    print("Starting closed-loop MPC simulation...")
    x_vals, u_vals, solve_times = [], [], []

    h_current, c_current = np.zeros(n_hidden), np.zeros(n_hidden)
    x_current = np.array([0.0, 0.0])  
    u_mpc = 0.0 

    total_steps = len(ref_seq) - N
    for i in range(total_steps):
        if i % 20 == 0: print(f"Simulation step: {i} / {total_steps}")
            
        mpc.opti.set_value(mpc.h0, h_current) 
        mpc.opti.set_value(mpc.c0, c_current) 
        mpc.opti.set_value(mpc.x0, x_current) 
        mpc.opti.set_value(mpc.u0, u_mpc) 
        mpc.opti.set_value(mpc.ref, ref_seq[i:i+N, :].T) 

        start_time = time.perf_counter()
        try:
            sol = mpc.solve()
            solve_times.append(time.perf_counter() - start_time) 
            
            u_mpc = sol.value(mpc.u[0]) 
            h_current = sol.value(mpc.h[:, 1])
            c_current = sol.value(mpc.c[:, 1])
        except Exception as e:
            print(f"\n[ERROR] Solver failed at step {i}: {e}")
            break

        ode_sol = solve_ivp(lambda t, y: van_der_pol(t, y, mu, u_mpc), [0, Ts], x_current, method='RK45')
        x_next = ode_sol.y[:, -1] 

        x_vals.append(x_next) 
        u_vals.append(u_mpc) 
        x_current = x_next 

    # 指标计算与保存
    x_vals, u_vals = np.array(x_vals), np.array(u_vals)
    xref_actual = ref_seq[:len(x_vals)]
    t_vec = np.arange(len(x_vals)) * Ts

    print("\n" + "="*50)
    print(f"控制性能评估 - {exp_prefix}")
    print("="*50)

    # 自动切分工况并计算平滑度
    phases = detect_reference_phases(xref_actual)
    delta_u = calculate_control_smoothness(u_vals) # VdP的u_vals是1D，返回标量

    csv_path = os.path.join(results_dir, "control_results.csv")
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Experiment", "Phase", 
                "RMSE_x1", "OS_x1(%)", "Delta_u", "Mean_Solve_Time(s)"
            ])
            
        overall_rmse, overall_os = [], []

        for start, end, name in phases:
            curr_x1, curr_ref = x_vals[start:end, 0], xref_actual[start:end, 0]
            
            _, _, rmse, os_val = calculate_tracking_metrics(curr_x1, curr_ref)
            overall_rmse.append(rmse)
            overall_os.append(os_val)
                
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                exp_prefix, name,
                f"{rmse:.4f}", f"{os_val:.2f}", "-", "-"
            ])
            print(f"{name} (Steps {start}-{end}): RMSE = {rmse:.4f} | Overshoot = {os_val:.2f}%")

        # 记录全局汇总
        avg_solve = np.mean(solve_times)
        final_rmse = np.mean(overall_rmse)
        final_os = np.max(overall_os)

        writer.writerow([
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            exp_prefix, "OVERALL",
            f"{final_rmse:.4f}", f"{final_os:.2f}",
            f"{delta_u:.4f}", f"{avg_solve:.4f}"
        ])
        print(f"OVERALL | Solve Time: {avg_solve:.4f}s | Delta_u: {delta_u:.4f}")

    # 保存数据与绘图
    data_to_save = {
        "x_vals": x_vals, "u_vals": u_vals, "t_vec": t_vec,
        "solve_times": np.array(solve_times), "xref_traj": xref_actual    
    }
    mat_path = os.path.join(results_dir, f"{exp_prefix}.mat")
    savemat(mat_path, data_to_save)
    print(f"\nSimulation data saved to: {mat_path}")
    
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_vec, x_vals[:, 0], label='x_1 (Controlled)')
    plt.plot(t_vec, xref_actual[:, 0], label='Reference', linestyle='--')
    plt.title(f"VdP MPC Control - {exp_prefix}")
    plt.legend(); plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(t_vec, x_vals[:, 1], label='x_2')
    plt.legend(); plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.step(t_vec, u_vals, label='u', where='post')
    plt.legend(); plt.grid()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{exp_prefix}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()