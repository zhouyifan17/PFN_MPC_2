import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.utils.signals import multisine

def generate_excitation_signal(T_period=2410, pmin=1, pmax=500, filter_coeff=0.1, seed=42):
    """
    生成通用的激励信号 (多正弦 + 正弦 + 低通滤波)。
    返回:
        u_data: 原始激励信号 (通常用于神经网络输入，模拟包含了执行器动态的黑盒)
        u_filtered: 滤波后的信号 (通常用于实际物理系统的 ODE 仿真)
    """
    u_multisine = 2 * multisine(N_points_per_period=T_period, N_periods=1, 
                                pmin=pmin, pmax=pmax, n_crest_factor_optim=100, seed=seed)
    x_wave = np.linspace(-np.pi, np.pi, T_period)
    u_data = u_multisine + np.sin(x_wave)

    u_filtered = np.zeros(len(u_data))
    for i in range(0, len(u_data)-1):
        u_filtered[i+1] = (1 - filter_coeff) * u_filtered[i] + filter_coeff * u_data[i]

    # 统一转换为二维数组 (N, 1)，方便后续支持多输入系统的矩阵运算
    return u_data.reshape(-1, 1), u_filtered.reshape(-1, 1)


def simulate_system(ode_func, u_sequence, y0, Ts):
    """
    通用的系统仿真引擎。
    参数:
        ode_func: 必须是一个签名类似于 f(t, y, u_i) 的可调用函数，其中 u_i 是当前时刻的输入数组。
        u_sequence: 形状为 (N, nu) 的 numpy 数组，nu 为输入维度。
        y0: 初始状态列表或数组。
        Ts: 采样时间。
    返回:
        t_vec: 时间向量
        x_states: 状态矩阵 (N+1, nx)
    """
    t_vec = [0]
    x_states = [np.array(y0)]
    y_current = np.array(y0)

    for i in range(len(u_sequence)):
        u_i = u_sequence[i]  # 提取当前时刻的输入 (对于单输入是 [u]，多输入是 [u1, u2])
        
        # 使用 RK45 求解一步
        sol = solve_ivp(lambda t, y: ode_func(t, y, u_i), [0, Ts], y_current, method='RK45')
        
        y_current = sol.y[:, -1]
        x_states.append(y_current)
        t_vec.append((i + 1) * Ts)

    return np.array(t_vec), np.array(x_states)


def build_dataset(u_network, x_states, N_train):
    """
    将序列数据拼接为深度学习所需的 (X, Y) 矩阵对。
    """
    X, Y = [], []
    
    for i in range(len(u_network)):
        # 拼接: 当前时刻的网络输入 + 当前时刻的状态
        X_i = np.concatenate((u_network[i], x_states[i, :]))
        X.append(X_i)
        # 标签: 下一时刻的状态
        Y.append(x_states[i + 1, :])

    X = np.array(X)
    Y = np.array(Y)

    X_train, X_val = X[:N_train, :], X[N_train:, :]
    Y_train, Y_val = Y[:N_train, :], Y[N_train:, :]

    return X_train, Y_train, X_val, Y_val


def plot_and_save_data(t_vec, x_states, u_data, save_dir):
    """通用的画图与保存工具"""
    os.makedirs(save_dir, exist_ok=True)
    nx = x_states.shape[1]
    
    plt.figure(figsize=(10, 2 * (nx + 1)))
    # 动态绘制所有状态
    for i in range(nx):
        plt.subplot(nx + 1, 1, i + 1)
        plt.plot(t_vec, x_states[:, i], label=f'x_{i+1}')
        plt.ylabel(f'x_{i+1}')
        plt.grid()
        plt.legend()
    
    # 绘制输入
    plt.subplot(nx + 1, 1, nx + 1)
    nu = u_data.shape[1]
    for i in range(nu):
        plt.plot(t_vec[:-1], u_data[:, i], label=f'u_{i+1}')
    plt.ylabel('u')
    plt.xlabel('t')
    plt.grid()
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'system_response.png'))
    plt.close()