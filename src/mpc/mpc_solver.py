import casadi as ca
import numpy as np
from src.mpc.casadi_utils import lstm_cell_ca, deepar_cell_ca, extract_deepar_weights_ca

class BaseMPCSolver:
    """所有数据驱动 MPC 求解器的通用基类"""
    def __init__(self, nx, ny, nu, N, Q, R, output_indices=[0]):
        self.nx = nx
        self.ny = ny
        self.nu = nu
        self.N = N
        self.opti = ca.Opti()

        # 通用决策变量
        self.x = self.opti.variable(nx, N + 1)
        self.y = self.opti.variable(ny, N)
        self.u = self.opti.variable(nu, N)
        self.du = self.opti.variable(nu, N)

        # 2. 通用参数 (供闭环循环时更新)
        self.x0 = self.opti.parameter(nx, 1)
        self.u0 = self.opti.parameter(nu, 1)
        self.ref = self.opti.parameter(ny, N)

        # 3. 通用代价函数: 轨迹跟踪 + 控制增量惩罚 (沿预测时域累加)
        cost = 0
        for i in range(N):
            err = self.y[:, i] - self.ref[:, i]
            cost += ca.mtimes([err.T, Q, err]) + ca.mtimes([self.du[:, i].T, R, self.du[:, i]])
        self.opti.minimize(cost)

        # 4. 通用约束: 控制增量 \Delta u
        self.opti.subject_to(self.du[:, 0] == self.u[:, 0] - self.u0)
        if N > 1:
            self.opti.subject_to(self.du[:, 1:] == self.u[:, 1:] - self.u[:, :-1])

        # 5. 通用约束: 初始状态匹配
        self.opti.subject_to(self.x[:, 0] == self.x0)
        
        # 6. 通用约束: 输出映射 (支持提取多个状态作为输出)
        for i, state_idx in enumerate(output_indices):
            self.opti.subject_to(self.y[i, :] == self.x[state_idx, 1:])

    def configure_solver(self):
        """求解器配置"""
        p_opts = {
            "expand": True,
            "print_time": 0
        }
        s_opts = {
            "print_level": 0,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "hessian_approximation": "limited-memory"
        }
        self.opti.solver('ipopt', p_opts, s_opts)

    def solve(self):
        """调用求解器"""
        return self.opti.solve()


class LSTM_MPCSolver(BaseMPCSolver):
    """
    LSTM 模型的 MPC 求解器
    """
    def __init__(self, nx, ny, nu, N, Q, R, n_hidden, weights, output_indices=[0]):
        # 初始化基类
        super().__init__(nx, ny, nu, N, Q, R, output_indices)
        
        W_ih, W_hh, b, W_fc, b_fc = weights

        # LSTM 模型参数
        self.h = self.opti.variable(n_hidden, N + 1)
        self.c = self.opti.variable(n_hidden, N + 1)
        self.h0 = self.opti.parameter(n_hidden, 1)
        self.c0 = self.opti.parameter(n_hidden, 1)

        # LSTM 初始约束
        self.opti.subject_to(self.h[:, 0] == self.h0)
        self.opti.subject_to(self.c[:, 0] == self.c0)

        # LSTM 动态约束在预测域内的展开 (Rollout)
        for i in range(N):
            y_next, h_next, c_next = lstm_cell_ca(
                ca.vertcat(self.u[:, i], self.x[:, i]),
                self.h[:, i], self.c[:, i], 
                W_ih, W_hh, b, W_fc, b_fc
            )
            self.opti.subject_to(self.x[:, i+1] == y_next)
            self.opti.subject_to(self.h[:, i+1] == h_next)
            self.opti.subject_to(self.c[:, i+1] == c_next)
        
        self.configure_solver()
            
class DeepAR_SMPC_Solver(BaseMPCSolver):
    """
    基于 DeepAR 概率预测的机会约束随机 MPC 求解器
    """
    def __init__(self, nx, ny, nu, N, Q, R, n_hidden, weights, output_indices=[0]):
        super().__init__(nx, ny, nu, N, Q, R, output_indices)
        # 提取 DeepAR 权重
        self.weights = weights
        self.h = self.opti.variable(n_hidden, N + 1)
        self.c = self.opti.variable(n_hidden, N + 1)

        # SMPC 参数
        self.h0 = self.opti.parameter(n_hidden, 1)
        self.c0 = self.opti.parameter(n_hidden, 1)
        self.z_score = self.opti.parameter(1, 1)    # 置信度系数 (如 1.96)
        self.x_max = self.opti.parameter(nx, 1)     # 物理上限
        self.x_min = self.opti.parameter(nx, 1)     # 物理下限

        # 初始隐藏状态约束
        self.opti.subject_to(self.h[:, 0] == self.h0)
        self.opti.subject_to(self.c[:, 0] == self.c0)

        # 概率动力学 Rollout 与 机会约束
        for i in range(N):
            mu_next, sigma_next, h_next, c_next = deepar_cell_ca(
                ca.vertcat(self.u[:, i], self.x[:, i]),
                self.h[:, i], self.c[:, i], 
                *self.weights
            )
            
            self.opti.subject_to(self.x[:, i+1] == mu_next)
            self.opti.subject_to(self.h[:, i+1] == h_next)
            self.opti.subject_to(self.c[:, i+1] == c_next)
            
            # # 机会约束
            # self.opti.subject_to(mu_next + self.z_score * sigma_next <= self.x_max)
            # self.opti.subject_to(mu_next - self.z_score * sigma_next >= self.x_min)

        self.configure_solver()
        
    def solve_mpc(self, x0, h0, c0, u0, ref, z=1.96, x_max=4.0, x_min=0.0):
        # 基类参数
        self.opti.set_value(self.x0, x0)
        self.opti.set_value(self.u0, u0)
        self.opti.set_value(self.ref, ref)
        
        # SMPC 参数
        self.opti.set_value(self.h0, h0)
        self.opti.set_value(self.c0, c0)
        self.opti.set_value(self.z_score, z)
        self.opti.set_value(self.x_max, x_max)
        self.opti.set_value(self.x_min, x_min)
        
        sol = self.solve()
        
        return sol.value(self.u[:, 0])