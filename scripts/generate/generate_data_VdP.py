import os
import numpy as np
import sys
import joblib
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.dynamics import van_der_pol
from src.utils.data_generator import (
    generate_excitation_signal, simulate_system, plot_and_save_data
)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Van der Pol System Data")
    # 数据集设置
    parser.add_argument("--n_episodes", type=int, default=50, help="生成的总轨迹数量")
    parser.add_argument("--t_total", type=int, default=2801, help="每条轨迹的总时间步数")
    parser.add_argument("--n_train_steps", type=int, default=2000, help="划分给训练集的步数")
    parser.add_argument("--n_val_steps", type=int, default=400, help="划分给验证集的步数")
    parser.add_argument("--noise_std", type=float, default=0.05, help="状态测量的观测噪声标准差")
    # VdP系统参数设置
    parser.add_argument("--mu", type=float, default=1.0, help="Van der Pol 阻尼参数 mu")
    parser.add_argument("--ts", type=float, default=0.1, help="仿真采样时间 (秒)")
    parser.add_argument("--seed", type=int, default=42, help="全局随机种子")
    # 绘图设置
    parser.add_argument("--n_plots", type=int, default=50, help="指定需要绘制并保存图像的轨迹数量")
    return parser.parse_args()

def main():
    args = parse_args()
    save_dir = "data/processed/van_der_pol"
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = "results/van_der_pol/data_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    np.random.seed(args.seed)
    
    vdp_ode = lambda t, y, u_i: van_der_pol(t, y, args.mu, u_i[0])

    print(f"Generating Van der Pol data: {args.n_episodes} trajectories...")

    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []
    X_test_list, Y_test_list = [], []
    
    t_vec_plot, x_states_plot, u_plot = None, None, None

    for ep in tqdm(range(args.n_episodes), desc="Simulating Episodes"):
        np.random.seed(args.seed + ep)
        # 在相空间内均匀采样，生成随机初始状态
        y0_random = np.random.uniform(low=-3.0, high=3.0, size=2)
        amp = np.random.uniform(low=1.0, high=10.0)
        
        u_data = generate_excitation_signal(T_period=args.t_total, seed=args.seed + ep, amplitude=amp)
        
        # 包装动力学方程并仿真
        t_vec, x_states_clean = simulate_system(vdp_ode, u_data, y0=y0_random, Ts=args.ts)
        noise = np.random.normal(loc=0.0, scale=args.noise_std, size=x_states_clean.shape)
        x_states = x_states_clean + noise
        # 逐点构建单步映射 (丢弃最后一步)
        X_ep, Y_ep = [], []
        for i in range(len(u_data) - 1):
            X_ep.append(np.concatenate((u_data[i], x_states[i])))
            Y_ep.append(x_states[i + 1])
            
        X_ep = np.array(X_ep)
        Y_ep = np.array(Y_ep)
        
        # 沿着时间轴切分数据集
        n_train = args.n_train_steps
        n_val = args.n_val_steps
        
        X_train_list.append(X_ep[:n_train])
        Y_train_list.append(Y_ep[:n_train])
        X_val_list.append(X_ep[n_train : n_train + n_val])
        Y_val_list.append(Y_ep[n_train : n_train + n_val])
        X_test_list.append(X_ep[n_train + n_val :])
        Y_test_list.append(Y_ep[n_train + n_val :])
        
        # 保存绘图
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
    X_test, Y_test = np.array(X_test_list), np.array(Y_test_list)
    
    # 保存训练集Scaler
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
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "Y_test.npy"), Y_test)

    print(f"Data successfully generated and saved to {save_dir}")
    print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val   Shape: X={X_val.shape}, Y={Y_val.shape}")
    print(f"Test  Shape: X={X_test.shape}, Y={Y_test.shape}")
    print(f"Plotted {min(args.n_plots, args.n_episodes)} sample trajectories in {plot_dir}")

if __name__ == "__main__":
    main()