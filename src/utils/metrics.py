import numpy as np

def calculate_tracking_metrics(y, target, preview_steps=10):
    """
    通用单维状态轨迹跟踪指标计算
    返回: MAE, MSE, RMSE
    """
    mae = np.mean(np.abs(y - target))
    mse = np.mean((y - target)**2)
    rmse = np.sqrt(mse)
    
    if len(y) > preview_steps:
        y_eval = y[:-preview_steps]
    else:
        y_eval = y
        
    y_inf = target[-1]
    overshoot = 0.0
    
    denom = np.abs(y_inf) + 1e-8 
    
    if y_inf >= y[0]:  # 上升或持平
        y_max = np.max(y_eval)
        if y_max > y_inf:
            overshoot = (y_max - y_inf) / denom * 100.0
    else:  # 下降
        y_min = np.min(y_eval)
        if y_min < y_inf:
            overshoot = (y_inf - y_min) / denom * 100.0
            
    return mae, mse, rmse, overshoot

def calculate_control_smoothness(u_vals):
    """
    计算控制序列的平均动作增量 Delta u
    u_vals 形状应为 (T_steps, nu) 或 (T_steps,)
    """
    if len(u_vals) <= 1:
        return np.zeros(1) if u_vals.ndim == 1 else np.zeros(u_vals.shape[1])
        
    delta_u = np.mean(np.abs(np.diff(u_vals, axis=0)), axis=0)
    return delta_u

def detect_reference_phases(xref_traj, threshold=1e-4):
    """
    通用参考轨迹阶跃点检测器
    输入: xref_traj (T_steps, ny)
    输出: phases 列表，格式为 [(start_idx, end_idx, "Phase_X"), ...]
    """
    # 计算所有维度在每一步的差值绝对值之和
    ref_diff = np.sum(np.abs(np.diff(xref_traj, axis=0)), axis=1)
    
    # 找到发生突变的索引
    change_idx = np.where(ref_diff > threshold)[0] + 1
    
    # 构建边界
    phase_bounds = [0] + change_idx.tolist() + [len(xref_traj)]
    
    phases = []
    for i in range(len(phase_bounds) - 1):
        phases.append((phase_bounds[i], phase_bounds[i+1], f"Phase_{i+1}"))
        
    return phases