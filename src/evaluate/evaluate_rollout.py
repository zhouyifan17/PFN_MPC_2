import os
import sys
import torch
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.abspath("."))
import src.models

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Multi-step Rollout MSE")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained .pth model")
    parser.add_argument("--model_type", type=str, required=True, choices=["LSTM", "DeepAR"], help="架构类型")
    parser.add_argument("--dataset", type=str, default="van_der_pol", help="数据集名称 (van_der_pol, four_tank)")
    parser.add_argument("--rollout_steps", type=int, default=10, help="Number of steps for auto-regressive rollout")
    return parser.parse_args()


def evaluate_rollout(model, X_val_scaled_t, Y_val_raw_t, scaler_Y_tensor, mean_Y_tensor, device, model_type, rollout_steps=10):
    model.eval()
    N_episodes, T_steps, _ = X_val_scaled_t.shape
    
    num_layers = model.lstm.num_layers
    hidden_size = model.lstm.hidden_size
    
    h = torch.zeros(num_layers, N_episodes, hidden_size).to(device)
    c = torch.zeros(num_layers, N_episodes, hidden_size).to(device)
    
    total_sq_error_single = 0.0
    total_sq_error_rollout = 0.0
    total_target_sq_single = 0.0
    total_target_sq_rollout = 0.0
    
    with torch.no_grad():
        for t in range(T_steps - rollout_steps):
            x_input_t = X_val_scaled_t[:, t:t+1, :]
            
            if model_type.lower() == "deepar":
                mu, _, (h, c) = model(x_input_t, (h, c))
                x_pred_curr = mu
            else:
                out, (h, c) = model.lstm(x_input_t, (h, c))
                x_pred_curr = model.fc(out)
                
            h_roll, c_roll = h, c
            rollout_preds = [x_pred_curr.squeeze(1)]
            
            for k in range(1, rollout_steps):
                u_future = X_val_scaled_t[:, t+k:t+k+1, 0:1] 
                x_input_roll = torch.cat([u_future, x_pred_curr], dim=2)
                
                if model_type.lower() == "deepar":
                    mu_roll, _, (h_roll, c_roll) = model(x_input_roll, (h_roll, c_roll))
                    x_pred_curr = mu_roll
                else:
                    out_roll, (h_roll, c_roll) = model.lstm(x_input_roll, (h_roll, c_roll))
                    x_pred_curr = model.fc(out_roll)
                    
                rollout_preds.append(x_pred_curr.squeeze(1))
                
            # [N_episodes, rollout_steps, nx]
            rollout_preds_t_scaled = torch.stack(rollout_preds, dim=1)
            rollout_preds_t_unscaled = rollout_preds_t_scaled * scaler_Y_tensor + mean_Y_tensor
            targets_unscaled = Y_val_raw_t[:, t:t+rollout_steps, :]
            
            total_sq_error_rollout += torch.sum((rollout_preds_t_unscaled - targets_unscaled)**2).item()
            total_target_sq_rollout += torch.sum(targets_unscaled**2).item()
            
            target_single_unscaled = targets_unscaled[:, 0, :]
            total_sq_error_single += torch.sum((rollout_preds_t_unscaled[:, 0, :] - target_single_unscaled)**2).item()
            total_target_sq_single += torch.sum(target_single_unscaled**2).item()
            
    avg_single_mse = total_sq_error_single / total_target_sq_single
    avg_rollout_mse = total_sq_error_rollout / total_target_sq_rollout
    return avg_single_mse, avg_rollout_mse

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_dir = f"data/processed/{args.dataset}"
    X_val_raw = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val_raw = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    print(f"Loaded Validation Data: X_val {X_val_raw.shape}")
    
    # 标准化
    scaler_X = joblib.load(os.path.join(data_dir, "scaler_X.pkl"))
    scaler_Y = joblib.load(os.path.join(data_dir, "scaler_Y.pkl"))
    
    X_val_scaled = scaler_X.transform(X_val_raw.reshape(-1, X_val_raw.shape[-1])).reshape(X_val_raw.shape)
    
    X_val_scaled_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    Y_val_raw_t = torch.tensor(Y_val_raw, dtype=torch.float32).to(device) 
    
    # 反标准化
    scaler_Y_tensor = torch.tensor(scaler_Y.scale_, dtype=torch.float32, device=device)
    mean_Y_tensor = torch.tensor(scaler_Y.mean_, dtype=torch.float32, device=device)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found at {args.model_path}")
        return
        
    print(f"\nEvaluating Model: {os.path.basename(args.model_path)} | Architecture: {args.model_type}")
    model = torch.load(args.model_path, map_location=device, weights_only=False)
    
    start_time = time.time()
    single_mse, rollout_mse = evaluate_rollout(
        model, X_val_scaled_t, Y_val_raw_t, 
        scaler_Y_tensor, mean_Y_tensor, device, 
        args.model_type, args.rollout_steps
    )
    elapsed = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"Evaluation Results (over {X_val_raw.shape[0]} trajectories, all possible start points):")
    print(f"-> Single-Step Normalized Physical MSE:        {single_mse:.6f}")
    print(f"-> {args.rollout_steps}-Step Rollout Normalized Physical MSE:  {rollout_mse:.6f}")
    print(f"Evaluation took {elapsed:.2f}s")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()