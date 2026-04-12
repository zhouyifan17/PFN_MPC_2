import os
import numpy as np
import sys
import argparse
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.dynamics.four_tank import four_tank
from src.utils.data_generator import generate_excitation_signal, simulate_system, plot_and_save_data, build_dataset

def generate_random_step_signal(total_steps, min_len=100, max_len=300, val_min=0.5, val_max=2.0):
    """
    生成一段随机的阶跃信号作为基础工作点
    """
    signal = np.zeros(total_steps)
    idx = 0
    while idx < total_steps:
        step_len = np.random.randint(min_len, max_len)
        signal[idx : idx + step_len] = np.random.uniform(val_min, val_max)
        idx += step_len
    return signal

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Four-Tank System Data")

    # 数据集配置
    parser.add_argument("--n_episodes", type=int, default=50, help="生成的总轨迹数量")
    parser.add_argument("--t_total", type=int, default=2400, help="每条轨迹的总时间步数")
    parser.add_argument("--n_train_steps", type=int, default=2000, help="划分给训练集的步数")
    parser.add_argument("--ts", type=float, default=5.0, help="仿真采样时间 (秒)")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    # 绘图配置
    parser.add_argument("--n_plots", type=int, default=1, help="指定需要绘制并保存图像的轨迹数量")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    save_dir = "data/processed/four_tank"
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = "results/four_tank/data_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 四容水箱物理参数
    params = {'a1': 1.31e-4, 'a2': 1.51e-4, 'a3': 9.27e-5, 'a4': 8.82e-5, 
              'gamma_a': 0.3, 'gamma_b': 0.4, 'g': 9.81, 'Sc': 0.06}
    four_tank_ode = lambda t, y, u_i: four_tank(t, y, float(u_i[0]), float(u_i[1]), params)
    
    print(f"Generating Four-Tank data: {args.n_episodes} trajectories...")
    
    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []
    
    for ep in tqdm(range(args.n_episodes), desc="Simulating Episodes"):
        np.random.seed(args.seed + ep)
        
        # 随机初始状态
        x0 = np.random.uniform(low=0.2, high=1.5, size=4)
        
        # 生成激励信号
        u1_base = generate_random_step_signal(args.t_total, val_min=0.5, val_max=2.5)
        u2_base = generate_random_step_signal(args.t_total, val_min=0.5, val_max=2.5)
        amp1 = np.random.uniform(low=0.2, high=0.8)
        amp2 = np.random.uniform(low=0.2, high=0.8)
        
        u1_fluct = generate_excitation_signal(T_period=args.t_total, seed=args.seed+ep, amplitude=amp1).flatten()
        u2_fluct = generate_excitation_signal(T_period=args.t_total, seed=100+ep, amplitude=amp2).flatten()
        
        u1_data = np.clip(u1_base + u1_fluct, 0.0, 4.5)
        u2_data = np.clip(u2_base + u2_fluct, 0.0, 4.5)
        u_data = np.vstack((u1_data, u2_data)).T # Shape: (t_total, 2)
        
        # 仿真生成数据
        t_vec, x_states = simulate_system(four_tank_ode, u_data, x0, args.ts)
        
        X_tr, Y_tr, X_va, Y_va = build_dataset(u_data, x_states, args.n_train_steps)
        
        X_train_list.append(X_tr)
        Y_train_list.append(Y_tr)
        X_val_list.append(X_va)
        Y_val_list.append(Y_va)

        if ep < args.n_plots:
            img_name = f"system_response_{ep}.png"
            plot_and_save_data(
                t_vec, x_states, u_data, 
                save_dir=plot_dir, 
                filename=img_name
            )

    # 堆叠成 3D 张量 [N_episodes, T_steps, Features]
    X_train, Y_train = np.array(X_train_list), np.array(Y_train_list)
    X_val, Y_val = np.array(X_val_list), np.array(Y_val_list)

    # 保存Scaler
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    scaler_X.fit(X_train.reshape(-1, X_train.shape[-1]))
    scaler_Y.fit(Y_train.reshape(-1, Y_train.shape[-1]))
    
    scaler_x_path = os.path.join(save_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(save_dir, "scaler_Y.pkl")
    joblib.dump(scaler_X, scaler_x_path)
    joblib.dump(scaler_Y, scaler_y_path)
    
    # 保存原始数据
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "Y_val.npy"), Y_val)

    print(f"\nData successfully generated and saved to {save_dir}")
    print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val Shape  : X={X_val.shape}, Y={Y_val.shape}")
    print(f"Plotted {min(args.n_plots, args.n_episodes)} sample trajectories in {plot_dir}")

if __name__ == "__main__":
    main()