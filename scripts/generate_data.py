import os
import numpy as np
from src.dynamics import van_der_pol
from src.utils.data_generator import (
    generate_excitation_signal, simulate_system, build_dataset, plot_and_save_data
)

def main():
    # 统一存放到 data 目录下
    save_dir = "data/processed/van_der_pol"
    os.makedirs(save_dir, exist_ok=True)

    print("Generating Van der Pol data...")
    
    # 1. 生成信号
    u_data, u_filtered = generate_excitation_signal(T_period=2410)
    
    # 2. 包装动力学方程并仿真
    mu = 1.0
    vdp_ode = lambda t, y, u_i: van_der_pol(t, y, mu, u_i[0])
    t_vec, x_states = simulate_system(vdp_ode, u_filtered, y0=[2, 0], Ts=0.1)
    
    # 3. 构建数据集
    X_train, Y_train, X_val, Y_val = build_dataset(u_data, x_states, N_train=2000)

    # 4. 将数据保存到本地
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "Y_val.npy"), Y_val)

    # 画图保存
    plot_and_save_data(t_vec, x_states, u_data, save_dir="results/van_der_pol/data_plots")
    print(f"Data successfully generated and saved to {save_dir}")

if __name__ == "__main__":
    main()