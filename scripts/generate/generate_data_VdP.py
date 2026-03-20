import os
import numpy as np
import sys

sys.path.append(os.path.abspath("."))
from src.dynamics import van_der_pol
from src.utils.data_generator import (
    generate_excitation_signal, simulate_system, plot_and_save_data
)

def main():
    save_dir = "data/processed/van_der_pol"
    os.makedirs(save_dir, exist_ok=True)
    
    np.random.seed(42)
    
    N_episodes = 50
    T_period = 2410
    N_train_steps = 2000  # 每一条轨迹的时间分界点
    
    mu = 1.0
    Ts = 0.1
    vdp_ode = lambda t, y, u_i: van_der_pol(t, y, mu, u_i[0])

    print(f"Generating Van der Pol data: {N_episodes} trajectories...")

    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []
    
    t_vec_plot, x_states_plot, u_plot = None, None, None

    for ep in range(N_episodes):
        # 在相空间内均匀采样，生成随机初始状态
        y0_random = np.random.uniform(low=-3.0, high=3.0, size=2)
        amp = np.random.uniform(low=1.0, high=10.0)
        
        u_data = generate_excitation_signal(T_period=T_period, seed=42 + ep, amplitude=amp)
        
        # 包装动力学方程并仿真
        t_vec, x_states = simulate_system(vdp_ode, u_data, y0=y0_random, Ts=Ts)
        
        if ep == 0:
            t_vec_plot, x_states_plot, u_plot = t_vec, x_states, u_data

        # 逐点构建单步映射 (丢弃最后一步)
        X_ep, Y_ep = [], []
        for i in range(len(u_data) - 1):
            X_ep.append(np.concatenate((u_data[i], x_states[i])))
            Y_ep.append(x_states[i + 1])
            
        X_ep = np.array(X_ep)
        Y_ep = np.array(Y_ep)
        
        # 沿着时间轴切分数据集
        X_train_list.append(X_ep[:N_train_steps])
        Y_train_list.append(Y_ep[:N_train_steps])
        
        X_val_list.append(X_ep[N_train_steps:])
        Y_val_list.append(Y_ep[N_train_steps:])

    # 堆叠成 3D 张量 [N_episodes, T_steps, Features]
    X_train = np.array(X_train_list)
    Y_train = np.array(Y_train_list)
    X_val = np.array(X_val_list)
    Y_val = np.array(Y_val_list)

    # 保存本地
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "Y_val.npy"), Y_val)

    plot_and_save_data(t_vec_plot, x_states_plot, u_plot, save_dir="results/van_der_pol/data_plots")
    
    print(f"Data successfully generated and saved to {save_dir}")
    print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape} (Past steps)")
    print(f"Val Shape  : X={X_val.shape}, Y={Y_val.shape} (Future steps)")

if __name__ == "__main__":
    main()