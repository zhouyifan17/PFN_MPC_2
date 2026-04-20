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
from src.dynamics import four_tank
from src.mpc.casadi_utils import extract_lstm_weights_ca, extract_deepar_weights_ca
from src.mpc.mpc_solver import LSTM_MPCSolver, DeepAR_SMPC_Solver
from src.utils.metrics import calculate_tracking_metrics, calculate_control_smoothness, detect_reference_phases

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MPC on Four-Tank")
    # 模型类型与控制类型
    parser.add_argument("--model_type", type=str, default="LSTM", choices=["LSTM", "DeepAR"])
    parser.add_argument("--mpc_type", type=str, default="MPC", choices=["MPC", "SMPC"], help="标准MPC和SMPC")
    # 数据集设置
    parser.add_argument("--data_source", type=str, default="real", choices=["real", "surrogate"])
    parser.add_argument("--train_mode", type=str, default="standard", choices=["standard", "distill"])
    parser.add_argument("--n_train", type=int, default=1, help="训练时使用的轨迹数量")
    # 模型结构预训练参数
    parser.add_argument("--num_layers", type=int, default=1, help="隐藏层数")
    parser.add_argument("--hidden_size", type=int, default=20, help="隐藏层单元数")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="蒸馏权重 (仅distill模式有效)")
    # MPC控制参数
    parser.add_argument("--horizon", type=int, default=10, help="MPC预测时域 N")
    parser.add_argument("--noise_std", type=float, default=0.02, help="MPC观测噪声标准差")
    parser.add_argument("--burn_in_steps", type=int, default=10, help="隐状态预热步数")
    # 其他参数
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    
    return parser.parse_args()

def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # 动态路径与参数配置
    n_hidden = 20 
    N = args.horizon
    Ts = 5.0     
    nx, ny, nu = 4, 4, 2 
    params = {'a1': 1.31e-4, 'a2': 1.51e-4, 'a3': 9.27e-5, 'a4': 8.82e-5, 
              'gamma_a': 0.3, 'gamma_b': 0.4, 'g': 9.81, 'Sc': 0.06}

    # 加载标准化 Scaler
    scaler_X_path = "data/processed/four_tank/scaler_X.pkl"
    scaler_Y_path = "data/processed/four_tank/scaler_Y.pkl"
    if not os.path.exists(scaler_X_path):
        raise FileNotFoundError(f"Scaler not found! Please check {scaler_X_path}")
    scaler_X = joblib.load(scaler_X_path)
    scaler_Y = joblib.load(scaler_Y_path)
    
    u_mean = scaler_X.mean_[:nu]
    u_scale = scaler_X.scale_[:nu]
    x_mean = scaler_Y.mean_
    x_scale = scaler_Y.scale_
    
    # 路径构建
    save_prefix = f"{args.data_source}_{args.train_mode}_N{args.n_train}"
    if args.train_mode == "distill":
        file_suffix = f"L{args.num_layers}_H{args.hidden_size}_W{args.distill_weight}"
    else:
        file_suffix = f"L{args.num_layers}_H{args.hidden_size}"
        
    ckpt_dir = f"checkpoints/{args.model_type}/four_tank/{save_prefix}"
    model_filename = f"model_{args.model_type}_{file_suffix}.pth"
    ckpt_path = os.path.join(ckpt_dir, model_filename)
    
    if args.train_mode == "distill":
        exp_prefix = f"{args.model_type}_{args.mpc_type}_FourTank_{save_prefix}_L{args.num_layers}_H{args.hidden_size}_W{args.distill_weight}_T{args.horizon}"
    else:
        exp_prefix = f"{args.model_type}_{args.mpc_type}_FourTank_{save_prefix}_L{args.num_layers}_H{args.hidden_size}_T{args.horizon}"
    
    results_dir = f"results/four_tank/{args.model_type}/mpc"
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading {args.model_type} model from {ckpt_path}...")
    sys.modules['models'] = src.models
    model = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)
    model.eval()
    
    Q_norm = np.diag(100.0 * (x_scale**2))
    R_norm = np.diag(1.0 * (u_scale**2))
    
    print(f"Initializing {args.model_type}-{args.mpc_type} Solver...")
    if args.model_type == "LSTM":
        weights = extract_lstm_weights_ca(model)
        mpc = LSTM_MPCSolver(nx, ny, nu, N, Q_norm, R_norm, n_hidden, weights, output_indices=[0, 1, 2, 3])
    else:
        weights = extract_deepar_weights_ca(model)
        mpc = DeepAR_SMPC_Solver(nx, ny, nu, N, Q_norm, R_norm, n_hidden, weights, output_indices=[0, 1, 2, 3])
    
    # 多维输入约束 (转化到归一化空间)
    u_max_norm = (np.array([4.0, 4.0]) - u_mean) / u_scale
    u_min_norm = (np.array([0.0, 0.0]) - u_mean) / u_scale
    # 对两个输入维度分别施加边界约束
    for k in range(nu):
        mpc.opti.subject_to(mpc.opti.bounded(u_min_norm[k], mpc.u[k, :], u_max_norm[k])) # type: ignore
        
    # 生成参考轨迹
    ref1 = np.tile([0.650, 0.650, 0.652, 0.664], (200, 1))
    ref2 = np.tile([0.300, 0.300, 0.301, 0.306], (200, 1))
    ref3 = np.tile([0.500, 0.750, 0.305, 1.200], (200, 1))
    ref4 = np.tile([0.900, 0.750, 1.062, 0.579], (200, 1))
    xref_traj = np.concatenate((ref1, ref2, ref3, ref4), axis=0)  # Shape (800, 4)
    ref_seq_norm = (xref_traj - x_mean) / x_scale
    
    # 初始化状态
    x_vals_clean, x_vals_noisy, u_vals, solve_times = [], [], [], []
    h_current, c_current = np.zeros(n_hidden), np.zeros(n_hidden)
    x_current_clean = np.array([1.0, 1.0, 0.0, 0.0])  
    x_current_noisy = x_current_clean.copy()
    u_mpc = np.array([0.0, 0.0])
    
    # 隐状态预热
    if args.burn_in_steps > 0:
        print(f"Warming up {args.model_type} hidden states for {args.burn_in_steps} steps (Burn-in)...")
        for _ in range(args.burn_in_steps):
            x_curr_norm = (x_current_noisy - x_mean) / x_scale
            u_mpc_norm = (u_mpc - u_mean) / u_scale
            
            with torch.no_grad():
                # 拼接高维输入: [u1, u2, x1, x2, x3, x4]
                features = np.concatenate((u_mpc_norm, x_curr_norm))
                x_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                h_tensor = torch.tensor(h_current, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                c_tensor = torch.tensor(c_current, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                _, (h_next_tensor, c_next_tensor) = model.lstm(x_input, (h_tensor, c_tensor))
                h_current = h_next_tensor.squeeze().numpy()
                c_current = c_next_tensor.squeeze().numpy()
            
            # 物理系统原地空转
            ode_sol = solve_ivp(lambda t, y: four_tank(t, y, u_mpc[0], u_mpc[1], params), [0, Ts], x_current_clean, method='RK45')
            x_current_clean = ode_sol.y[:, -1]
            noise = np.random.normal(0.0, args.noise_std, size=x_current_clean.shape)
            x_current_noisy = x_current_clean + noise
    
    # 闭环仿真循环
    print("Starting closed-loop MPC simulation...")
    total_steps = len(xref_traj) - N
    for i in range(total_steps):
        if i % 10 == 0:
            print(f"Simulation step: {i} / {total_steps}")
            
        x_curr_norm = (x_current_noisy - x_mean) / x_scale
        u_mpc_norm = (u_mpc - u_mean) / u_scale
            
        mpc.opti.set_value(mpc.h0, h_current) 
        mpc.opti.set_value(mpc.c0, c_current) 
        mpc.opti.set_value(mpc.x0, x_curr_norm) 
        mpc.opti.set_value(mpc.u0, u_mpc_norm) 
        mpc.opti.set_value(mpc.ref, ref_seq_norm[i:i+N, :].T)

        # 安全边界设置 (SMPC) TODO: 这里还要进一步细化
        if args.model_type == "DeepAR":
            z_val = 1.96 if args.mpc_type == "SMPC" else 0.0
            mpc.opti.set_value(mpc.z_score, z_val)
            # 四容水箱宽容的安全边界
            x_max_norm = (np.array([5.0, 5.0, 5.0, 5.0]) - x_mean) / x_scale
            x_min_norm = (np.array([-0.5, -0.5, -0.5, -0.5]) - x_mean) / x_scale
            mpc.opti.set_value(mpc.x_max, x_max_norm) 
            mpc.opti.set_value(mpc.x_min, x_min_norm)
            
        start_time = time.perf_counter()
        try:
            sol = mpc.solve()
            solve_times.append(time.perf_counter() - start_time) 
            
            # ==========================================================
            # 冷启动诊断监测模块
            if i < 5:
                print(f"\n[DIAGNOSTIC] --- Step {i} ---")
                pred_x_norm = sol.value(mpc.x)
                pred_u_norm = sol.value(mpc.u)
                
                # 提取第一维状态 x_1 作为观测指标
                pred_x_phys = pred_x_norm[0, :] * x_scale[0] + x_mean[0]
                pred_u_phys = pred_u_norm[0, :] * u_scale[0] + u_mean[0]
                
                print(f"1. 当前真实状态 x_1: {x_current_noisy[0]:.4f}")
                print(f"2. 求解器预测的未来 3 步 x_1 轨迹: {np.round(pred_x_phys[1:4], 4)}")
                print(f"3. 求解器计划的未来 3 步控制量 u_1: {np.round(pred_u_phys[:3], 4)}")
                print("-" * 40)
            # ==========================================================
            
            # 提取多维控制量并反归一化
            u_mpc_norm_new = np.array(sol.value(mpc.u[:, 0]))
            u_mpc = u_mpc_norm_new * u_scale + u_mean
            
            h_current = sol.value(mpc.h[:, 1])
            c_current = sol.value(mpc.c[:, 1])
        except Exception as e:
            print(f"\n[WARNING] Solver failed at step {i}: {e}. Using previous control input.")
            pass

        # 物理系统演进
        ode_sol = solve_ivp(lambda t, y: four_tank(t, y, u_mpc[0], u_mpc[1], params), [0, Ts], x_current_clean, method='RK45')
        x_next_clean = ode_sol.y[:, -1] 
        noise = np.random.normal(0.0, args.noise_std, size=x_next_clean.shape)
        x_next_noisy = x_next_clean + noise

        x_vals_clean.append(x_next_clean) 
        x_vals_noisy.append(x_next_noisy) 
        u_vals.append([u_mpc[0], u_mpc[1]]) 
        
        x_current_clean = x_next_clean 
        x_current_noisy = x_next_noisy

    # 指标计算与保存
    x_vals_clean, x_vals_noisy = np.array(x_vals_clean), np.array(x_vals_noisy)
    u_vals = np.array(u_vals)
    xref_actual = xref_traj[:len(x_vals_clean)]
    t_vec = np.arange(len(x_vals_clean)) * Ts

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
            curr_x, curr_ref = x_vals_clean[start:end, :], xref_actual[start:end, :]
            
            p_rmse, p_os = [], []
            for k in range(nx):
                _, _, rmse, os_val = calculate_tracking_metrics(
                    curr_x[:, k], 
                    curr_ref[:, k], 
                    preview_steps=args.horizon
                )
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
        "x_vals_clean": x_vals_clean, "x_vals_noisy": x_vals_noisy, 
        "u_vals": u_vals, "t_vec": t_vec,
        "solve_times": np.array(solve_times), "xref_traj": xref_actual    
    }
    mat_path = os.path.join(results_dir, f"{exp_prefix}.mat")
    savemat(mat_path, data_to_save)
    print(f"\nSimulation data saved to: {mat_path}")
    
    plt.figure(figsize=(15, 15))
    for k in range(4):
        plt.subplot(6, 1, k+1)
        plt.plot(t_vec, x_vals_noisy[:, k], color='lightblue', alpha=0.5, label=f'x_{k+1} (Noisy Sensor)')
        plt.plot(t_vec, x_vals_clean[:, k], color='blue', linewidth=2, label=f'x_{k+1} (Clean True State)')
        plt.plot(t_vec, xref_actual[:, k], label='Reference', color='red', linestyle='--', linewidth=2)
        plt.ylabel(f"x_{k+1}"); plt.legend(loc='upper right'); plt.grid()

    plt.subplot(6, 1, 5)
    plt.step(t_vec, u_vals[:, 0], label='u_1', color='purple', where='post')
    plt.ylabel("u_1"); plt.legend(loc='upper right'); plt.grid()
    
    plt.subplot(6, 1, 6)
    plt.step(t_vec, u_vals[:, 1], label='u_2', color='purple', where='post')
    plt.xlabel("Time (s)"); plt.ylabel("u_2"); plt.legend(loc='upper right'); plt.grid()

    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{exp_prefix}.png")
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()