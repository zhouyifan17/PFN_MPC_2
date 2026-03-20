import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath("."))
import src.models

def evaluate_rollout(model, X_val, Y_val, device, rollout_steps=10):
    """
    进行极其严格的自回归多步滚动预测评估。
    完全模拟 MPC 求解器中的前瞻 (Horizon) 预测行为。
    """
    model.eval()
    N_episodes, T_steps, _ = X_val.shape
    
    # 转换为 Tensor: [Batch=50, Seq=T, Features=3]
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)
    
    # 动态获取 LSTM 层数和隐藏维度
    num_layers = model.lstm.num_layers
    hidden_size = model.lstm.hidden_size
    
    # 初始化全局隐藏状态
    h = torch.zeros(num_layers, N_episodes, hidden_size).to(device)
    c = torch.zeros(num_layers, N_episodes, hidden_size).to(device)
    
    total_sq_error_single = 0.0
    total_sq_error_rollout = 0.0
    total_target_sq_single = 0.0
    total_target_sq_rollout = 0.0
    
    with torch.no_grad():
        # 遍历每一个可以向未来展望 rollout_steps 步的时间点
        for t in range(T_steps - rollout_steps):
            
            # 1. 主分支：使用【真实的】历史数据一步步推进，更新准确的 (h, c)
            x_input_t = X_val_t[:, t:t+1, :]
            out, (h, c) = model.lstm(x_input_t, (h, c))
            
            # 2. 滚动分支：以此刻真实的 (h, c) 为起点，向未来滚动预测 10 步
            h_roll, c_roll = h, c
            x_pred_curr = model.fc(out)  # 这是第 1 步的预测结果 (对应 x_{t+1})
            
            rollout_preds = [x_pred_curr.squeeze(1)]
            
            # 自回归循环 (2 到 10 步)
            for k in range(1, rollout_steps):
                # 获取未来的真实控制输入 u_{t+k}
                u_future = X_val_t[:, t+k:t+k+1, 0:1] 
                
                # 【致命区别】：这里的状态 x 用的是上一步的预测值！
                x_input_roll = torch.cat([u_future, x_pred_curr], dim=2)
                
                # 滚动推进隐藏状态
                out_roll, (h_roll, c_roll) = model.lstm(x_input_roll, (h_roll, c_roll))
                x_pred_curr = model.fc(out_roll)
                
                rollout_preds.append(x_pred_curr.squeeze(1))
                
            # 将 10 步预测堆叠起来: [N_episodes, rollout_steps, 2]
            rollout_preds_t = torch.stack(rollout_preds, dim=1)
            
            # 获取对应的 10 步真实目标
            targets = Y_val_t[:, t:t+rollout_steps, :]
            
            total_sq_error_rollout += torch.sum((rollout_preds_t - targets)**2).item()
            total_target_sq_rollout += torch.sum(targets**2).item()
            
            target_single = targets[:, 0, :]
            total_sq_error_single += torch.sum((rollout_preds[0] - target_single)**2).item()
            total_target_sq_single += torch.sum(target_single**2).item()
            
    # 3. 替换最后的平均逻辑（在全局尺度上相除，完美复刻 trainer.py 的逻辑）
    avg_single_mse = total_sq_error_single / total_target_sq_single
    avg_rollout_mse = total_sq_error_rollout / total_target_sq_rollout
    
    return avg_single_mse, avg_rollout_mse

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-step Rollout MSE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model")
    parser.add_argument("--rollout_steps", type=int, default=10, help="Number of steps for auto-regressive rollout")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 加载所有 50 条验证集轨迹
    data_dir = "data/processed/van_der_pol"
    X_val = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    print(f"Loaded Validation Data: X_val {X_val.shape}")
    
    # 2. 加载指定的模型
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    print(f"\nEvaluating Model: {os.path.basename(args.model_path)}")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # 3. 执行极度严苛的多步滚动测试
    start_time = time.time()
    single_mse, rollout_mse = evaluate_rollout(model, X_val, Y_val, device, args.rollout_steps)
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"Evaluation Results (over {X_val.shape[0]} trajectories, all possible start points):")
    print(f"-> Single-Step MSE:        {single_mse:.6f}")
    print(f"-> {args.rollout_steps}-Step Rollout MSE:  {rollout_mse:.6f}")
    print(f"Evaluation took {elapsed:.2f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()