import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from tqdm import tqdm

sys.path.append(os.path.abspath("."))
from src.dynamics.four_tank import four_tank
from src.utils.data_generator import generate_excitation_signal, simulate_system, plot_and_save_data, build_dataset

def generate_random_step_signal(total_steps, min_len=100, max_len=400, val_min=0.5, val_max=4.0):
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

def main():
    save_dir = "data/processed/four_tank"
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = "results/four_tank/data_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    # 实验配置
    N_episodes = 50
    T_total = 2400       
    N_train_steps = 2000 
    Ts = 5.0
    params = {'a1': 1.31e-4, 'a2': 1.51e-4, 'a3': 9.27e-5, 'a4': 8.82e-5, 
              'gamma_a': 0.3, 'gamma_b': 0.4, 'g': 9.81, 'Sc': 0.06}
    four_tank_ode = lambda t, y, u_i: four_tank(t, y, float(u_i[0]), float(u_i[1]), params)
    
    print(f"Generating Four-Tank data: {N_episodes} trajectories...")
    
    X_train_list, Y_train_list = [], []
    X_val_list, Y_val_list = [], []
    t_vec_plot, x_states_plot, u_plot = None, None, None
    
    for ep in tqdm(range(N_episodes), desc="Simulating Episodes"):
        np.random.seed(42 + ep)
        
        # 1. 随机初始状态
        x0 = np.random.uniform(low=0.2, high=2.0, size=4)
        
        # 2. 生成激励信号
        u1_base = generate_random_step_signal(T_total, val_min=1.0, val_max=4.0)
        u2_base = generate_random_step_signal(T_total, val_min=1.0, val_max=4.0)
        amp1 = np.random.uniform(low=0.5, high=3.0)
        amp2 = np.random.uniform(low=0.5, high=3.0)
        
        u1_fluct = generate_excitation_signal(T_period=T_total, seed=42+ep, amplitude=amp1).flatten()
        u2_fluct = generate_excitation_signal(T_period=T_total, seed=100+ep, amplitude=amp2).flatten()
        
        u1_data = np.clip(u1_base + u1_fluct, 0.0, 6.0)
        u2_data = np.clip(u2_base + u2_fluct, 0.0, 6.0)
        u_data = np.vstack((u1_data, u2_data)).T # Shape: (2400, 2)
        
        # 3. 仿真生成数据
        t_vec, x_states = simulate_system(four_tank_ode, u_data, x0, Ts)
        
        X_tr, Y_tr, X_va, Y_va = build_dataset(u_data, x_states, N_train_steps)
        
        X_train_list.append(X_tr)
        Y_train_list.append(Y_tr)
        X_val_list.append(X_va)
        Y_val_list.append(Y_va)

        # 记录第一条轨迹供绘图展示
        if ep == 0:
            t_vec_plot, x_states_plot, u_plot = t_vec, x_states, u_data

    # 4. 堆叠并保存
    X_train, Y_train = np.array(X_train_list), np.array(Y_train_list)
    X_val, Y_val = np.array(X_val_list), np.array(Y_val_list)

    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "Y_val.npy"), Y_val)

    print(f"\nData successfully generated and saved to {save_dir}")
    print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val Shape  : X={X_val.shape}, Y={Y_val.shape}")

    # 绘图并保存
    plot_and_save_data(t_vec_plot, x_states_plot, u_plot, save_dir=plot_dir)
    print("Sample trajectory plot saved.")

if __name__ == "__main__":
    main()