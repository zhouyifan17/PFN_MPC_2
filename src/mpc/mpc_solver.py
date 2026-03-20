import casadi as ca
import numpy as np
from src.mpc.casadi_utils import lstm_cell_ca

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

        # 7. 求解器配置
        self.opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0})

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

        # 1. LSTM 专有的变量与参数
        self.h = self.opti.variable(n_hidden, N + 1)
        self.c = self.opti.variable(n_hidden, N + 1)
        self.h0 = self.opti.parameter(n_hidden, 1)
        self.c0 = self.opti.parameter(n_hidden, 1)

        # 2. LSTM 专有的初始约束
        self.opti.subject_to(self.h[:, 0] == self.h0)
        self.opti.subject_to(self.c[:, 0] == self.c0)

        # 3. LSTM 动态约束在预测域内的展开 (Rollout)
        for i in range(N):
            y_next, h_next, c_next = lstm_cell_ca(
                ca.vertcat(self.u[:, i], self.x[:, i]),
                self.h[:, i], self.c[:, i], 
                W_ih, W_hh, b, W_fc, b_fc
            )
            self.opti.subject_to(self.x[:, i+1] == y_next)
            self.opti.subject_to(self.h[:, i+1] == h_next)
            self.opti.subject_to(self.c[:, i+1] == c_next)