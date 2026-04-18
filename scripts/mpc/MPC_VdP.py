import os
import time
import sys
import csv
import datetime
import argparse
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath("."))
import src.models
from src.dynamics import van_der_pol
from src.mpc.casadi_utils import extract_lstm_weights_ca, extract_deepar_weights_ca
from src.mpc.mpc_solver import LSTM_MPCSolver, DeepAR_SMPC_Solver
from src.utils.metrics import calculate_tracking_metrics, calculate_control_smoothness, detect_reference_phases

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LSTM-MPC on Van der Pol")
    # 模型类型与控制类型
    parser.add_argument("--model_type", type=str, default="LSTM", choices=["LSTM", "DeepAR"])
    parser.add_argument("--mpc_type", type=str, default="MPC", choices=["MPC", "SMPC"], help="标准MPC和SMPC")
    # 数据集设置
    parser.add_argument("--data_source", type=str, default="real", choices=["real", "surrogate"])
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "distill"])
    parser.add_argument("--n_train", type=int, default=1, help="训练时使用的轨迹数量")
    # 模型结构与训练参数
    parser.add_argument("--num_layers", type=int, default=1, help="隐藏层数")
    parser.add_argument("--hidden_size", type=int, default=80, help="隐藏层单元数")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="蒸馏权重 (仅distill模式有效)")
    # MPC 控制参数
    parser.add_argument("--horizon", type=int, default=10, help="MPC预测时域 N")
    parser.add_argument("--noise_std", type=float, default=0.08, help="MPC观测噪声标准差(同训练集)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载标准化 Scaler
    scaler_X_path = "data/processed/van_der_pol/scaler_X.pkl"
    scaler_Y_path = "data/processed/van_der_pol/scaler_Y.pkl"
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Scaler not found! Please check {scaler_X_path}")
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)
    u_mean = scaler_X.mean_[0]
    u_scale = scaler_X.scale_[0]
    x_mean = scaler_Y.mean_
    x_scale = scaler_Y.scale_
    
    # 路径构建
    save_prefix = f"{args.data_source}_{args.mode}_N{args.n_train}"
    if args.mode == "distill":
        file_suffix = f"L{args.num_layers}_H{args.hidden_size}_W{args.distill_weight}"
    else:
        file_suffix = f"L{args.num_layers}_H{args.hidden_size}"
        
    ckpt_dir = f"checkpoints/{args.model_type}/van_der_pol/{save_prefix}"
    model_filename = f"model_{args.model_type}_{file_suffix}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)
    
    if args.mode == "distill":
        exp_prefix = f"{args.model_type}_{args.mpc_type}_VdP_{save_prefix}_L{args.num_layers}_H{args.hidden_size}_W{args.distill_weight}_T{args.horizon}"
    else:
        exp_prefix = f"{args.model_type}_{args.mpc_type}_VdP_{save_prefix}_L{args.num_layers}_H{args.hidden_size}_T{args.horizon}"
    
    results_dir = f"results/van_der_pol/{args.model_type}/mpc"
    os.makedirs(results_dir, exist_ok=True)

    # VdP与MPC参数配置
    n_hidden = args.hidden_size
    N = args.horizon       
    Ts = 0.1     
    mu = 1.0     
    nx, ny, nu = 2, 1, 1
    
    # 加载模型并提取 CasADi 权重
    print(f"Loading {args.model_type} model from {ckpt_path}...")
    sys.modules['models'] = src.models
    model = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()

    # 配置 MPC 代价矩阵
    Q_norm = np.eye(ny) * 100 * (x_scale[0]**2)
    R_norm = np.eye(nu) * 1.0 * (u_scale**2)
    
    print(f"Initializing {args.model_type}-{args.mpc_type} Solver...")
    if args.model_type == "LSTM":
        weights = extract_lstm_weights_ca(model)
        mpc = LSTM_MPCSolver(nx, ny, nu, N, Q_norm, R_norm, n_hidden, weights, output_indices=[0])
    else:
        weights = extract_deepar_weights_ca(model)
        mpc = DeepAR_SMPC_Solver(nx, ny, nu, N, Q_norm, R_norm, n_hidden, weights, output_indices=[0])
        
    # 输入约束
    u_max_norm = (10.0 - u_mean) / u_scale
    u_min_norm = (-10.0 - u_mean) / u_scale
    mpc.opti.subject_to(mpc.opti.bounded(u_min_norm, mpc.u, u_max_norm)) # type: ignore
    
    # 生成参考轨迹
    ref_seq = np.concatenate((
        np.tile([0.8], (40, 1)), np.tile([2.5], (40, 1)), 
        np.tile([-1.5], (40, 1)), np.tile([1.2], (40, 1))
    ), axis=0).flatten().reshape(-1, 1)
    ref_seq_norm = (ref_seq - x_mean[0]) / x_scale[0] # 将参考轨迹归一化
    
    # 闭环仿真循环
    x_vals_clean, x_vals_noisy, u_vals, solve_times = [], [], [], []

    h_current, c_current = np.zeros(n_hidden), np.zeros(n_hidden)
    x_current_clean = np.array([0.0, 0.0])
    x_current_noisy = x_current_clean.copy()
    u_mpc = 0.0 

    total_steps = len(ref_seq) - N
    for i in range(total_steps):
        if i % 20 == 0: print(f"Simulation step: {i} / {total_steps}")
        
        x_curr_norm = (x_current_noisy - x_mean) / x_scale
        u_mpc_norm = (u_mpc - u_mean) / u_scale
        
        mpc.opti.set_value(mpc.h0, h_current) 
        mpc.opti.set_value(mpc.c0, c_current) 
        mpc.opti.set_value(mpc.u0, u_mpc_norm) 
        mpc.opti.set_value(mpc.ref, ref_seq_norm[i:i+N, :].T) 
        mpc.opti.set_value(mpc.x0, x_curr_norm)
        
        # 安全边界设置
        if args.model_type == "DeepAR":
            z_val = 1.96 if args.mpc_type == "SMPC" else 0.0
            mpc.opti.set_value(mpc.z_score, z_val)
            # TODO: 安全边界的进一步检查和设计
            x_max_norm = (np.array([10.0, 30.0]) - x_mean) / x_scale
            x_min_norm = (np.array([-10.0, -30.0]) - x_mean) / x_scale
            mpc.opti.set_value(mpc.x_max, x_max_norm) 
            mpc.opti.set_value(mpc.x_min, x_min_norm)

        # 求解最优控制
        start_time = time.perf_counter()
        try:
            sol = mpc.solve()
            solve_times.append(time.perf_counter() - start_time) 
            
            u_mpc_norm_new = sol.value(mpc.u[0])
            u_mpc = float(u_mpc_norm_new * u_scale + u_mean)
            h_current = sol.value(mpc.h[:, 1])
            c_current = sol.value(mpc.c[:, 1])
        except Exception as e:
            print(f"\n[ERROR] Solver failed at step {i}: {e}")
            pass

        ode_sol = solve_ivp(lambda t, y: van_der_pol(t, y, mu, u_mpc), [0, Ts], x_current_clean, method='RK45')
        x_next_clean = ode_sol.y[:, -1]
        noise = np.random.normal(loc=0.0, scale=args.noise_std, size=x_next_clean.shape)
        x_next_noisy = x_next_clean + noise
        
        x_vals_clean.append(x_next_clean) 
        x_vals_noisy.append(x_next_noisy)
        u_vals.append(u_mpc)
        # 推进时间步
        x_current_clean = x_next_clean
        x_current_noisy = x_next_noisy
        
    # 指标计算与保存
    x_vals_clean = np.array(x_vals_clean)
    x_vals_noisy = np.array(x_vals_noisy)
    u_vals = np.array(u_vals)
    xref_actual = ref_seq[:len(x_vals_clean)]
    t_vec = np.arange(len(x_vals_clean)) * Ts

    print("\n" + "="*50)
    print(f"控制性能评估 - {exp_prefix}")
    print("="*50)

    # 自动切分工况并计算平滑度
    phases = detect_reference_phases(xref_actual)
    delta_u = calculate_control_smoothness(u_vals)

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
            curr_x1, curr_ref = x_vals_clean[start:end, 0], xref_actual[start:end, 0]
            
            _, _, rmse, os_val = calculate_tracking_metrics(curr_x1, curr_ref)
            overall_rmse.append(rmse)
            overall_os.append(os_val)
                
            writer.writerow([
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                exp_prefix, name,
                f"{rmse:.4f}", f"{os_val:.2f}", "-", "-"
            ])
            print(f"{name} (Steps {start}-{end}): RMSE = {rmse:.4f} | Overshoot = {os_val:.2f}%")

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
        "x_vals_clean": x_vals_clean, "x_vals_noisy": np.array(x_vals_noisy), 
        "u_vals": u_vals, "t_vec": t_vec,
        "solve_times": np.array(solve_times), "xref_traj": xref_actual    
    }
    mat_path = os.path.join(results_dir, f"{exp_prefix}.mat")
    savemat(mat_path, data_to_save)
    print(f"\nSimulation data saved to: {mat_path}")
    
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_vec, x_vals_noisy[:, 0], color='lightblue', alpha=0.5, label='x_1 (Noisy Sensor)') # type: ignore
    plt.plot(t_vec, x_vals_clean[:, 0], color='blue', linewidth=2, label='x_1 (Clean True State)')
    plt.plot(t_vec, xref_actual[:, 0], color='red', linestyle='--', linewidth=2, label='Reference')
    plt.title(f"VdP MPC Control - {exp_prefix}")
    plt.legend(loc='upper right')
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(t_vec, x_vals_noisy[:, 1], color='lightblue', alpha=0.5, label='x_2 (Noisy Sensor)') # type: ignore
    plt.plot(t_vec, x_vals_clean[:, 1], color='blue', linewidth=2, label='x_2 (Clean True State)')
    plt.legend(loc='upper right')
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.step(t_vec, u_vals, label='u', where='post', color='purple')
    plt.legend(loc='upper right')
    plt.grid()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{exp_prefix}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()