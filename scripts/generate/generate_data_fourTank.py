import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.append(os.path.abspath("."))
from src.dynamics.four_tank import four_tank
from src.utils.signals import multisine

def main():
    save_dir = "data/processed/four_tank"
    os.makedirs(save_dir, exist_ok=True)
    plot_dir = "results/four_tank/data_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    print("Generating Four-Tank data: 1 long trajectory (8000 steps)...")
    
    # 1. 生成 u1 的分段常数参考
    u1_data1 =  np.tile([1.000], (500, 1))
    u1_data2 =  np.tile([1.500], (500, 1))
    u1_data3 =  np.tile([2.000], (500, 1))
    u1_data4 =  np.tile([2.500], (500, 1))
    u1_data_all = np.concatenate((u1_data1, u1_data2, u1_data3, u1_data4), axis=0)
    u1_data_all = np.concatenate((u1_data_all, u1_data_all, u1_data_all, u1_data_all), axis=0)

    # 2. 生成 u2 的分段常数参考
    u2_data1 =  np.tile([1.000], (2000, 1))
    u2_data2 =  np.tile([1.500], (2000, 1))
    u2_data3 =  np.tile([2.000], (2000, 1))
    u2_data4 =  np.tile([2.500], (2000, 1))
    u2_data_all = np.concatenate((u2_data1, u2_data2, u2_data3, u2_data4), axis=0)

    # 3. 添加多频正弦扰动
    u_data_1_additional = 0.5 * multisine(N_points_per_period=8000, N_periods=1, pmin=1, pmax=500, n_crest_factor_optim=100, seed=42)
    u_data_2_additional = 0.5 * multisine(N_points_per_period=8000, N_periods=1, pmin=1, pmax=500, n_crest_factor_optim=100, seed=36)
    
    u1_data = u1_data_all + u_data_1_additional.reshape(-1, 1)
    u2_data = u2_data_all + u_data_2_additional.reshape(-1, 1)
    u_data = np.hstack((u1_data, u2_data)) # Shape: (8000, 2)
    
    # 4. 物理系统参数与仿真初始化
    Ts = 5 
    params = {'a1': 1.31e-4, 'a2': 1.51e-4, 'a3': 9.27e-5, 'a4': 8.82e-5, 
              'gamma_a': 0.3, 'gamma_b': 0.4, 'g': 9.81, 'Sc': 0.06}
    x0 = np.array([0.5, 0.5, 0.5, 0.5])
    
    x_states = [x0]
    
    # 5. 仿真循环
    print("Simulating ODE...")
    for i in range(len(u_data)):
        u1, u2 = float(u1_data[i, 0]), float(u2_data[i, 0])
        sol = solve_ivp(lambda t, y: four_tank(t, y, u1, u2, params), [0, Ts], x0, method='RK45')
        x0 = sol.y[:, -1]
        x_states.append(x0)

    x_states = np.array(x_states) # Shape: (8001, 4)
    
    # 6. 构建监督学习输入特征 (X: [u1, u2, x1, x2, x3, x4], Y: [x1_next, x2_next, x3_next, x4_next])
    X = []
    Y = []
    for i in range(len(u_data)):
        X.append(np.concatenate((u_data[i, :], x_states[i, :])))
        Y.append(x_states[i + 1, :])
        
    X = np.array(X) # Shape: (8000, 6)
    Y = np.array(Y) # Shape: (8000, 4)

    # 7. 切分训练集与验证集，并增加 N_episodes=1 的维度以对齐工程规范
    X_train = X[0:7500, :][np.newaxis, :, :] # Shape: (1, 7500, 6)
    Y_train = Y[0:7500, :][np.newaxis, :, :] # Shape: (1, 7500, 4)
    X_val = X[7500:, :][np.newaxis, :, :]    # Shape: (1, 500, 6)
    Y_val = Y[7500:, :][np.newaxis, :, :]    # Shape: (1, 500, 4)

    # 8. 保存数据
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "X_val.npy"), X_val)
    np.save(os.path.join(save_dir, "Y_val.npy"), Y_val)

    print(f"Data successfully generated and saved to {save_dir}")
    print(f"Train Shape: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Val Shape  : X={X_val.shape}, Y={Y_val.shape}")

if __name__ == "__main__":
    main()