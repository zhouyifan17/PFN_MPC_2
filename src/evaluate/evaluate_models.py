import os
import sys
import torch
import argparse
import datetime
import numpy as np
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append(".")
from src.models import TabPFNWrapper
from src.models import LSTMModel, DeepARModel

def get_args():
    parser = argparse.ArgumentParser(description="Unified Evaluation for TabPFN, LSTM, and DeepAR")
    # 通用参数
    parser.add_argument("--model_type", type=str, required=True, choices=["TabPFN", "LSTM", "DeepAR"], help="评估的模型类型")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="van_der_pol", help="数据集名称 (van_der_pol, four_tank)")
    parser.add_argument("--eval_set", type=str, default="test", choices=["val", "test"], help="指定要评估的数据集")
    parser.add_argument("--pred_mode", type=str, default="rolling",
                        choices=["single_step", "rolling"],
                        help="single_step: 教师强制单步预测; rolling: 自回归滚动预测")
    parser.add_argument("--n_plots", type=int, default=50, help="绘制预测轨迹对比图的数量")
    # warm-up 参数
    parser.add_argument("--warmup_steps", type=int, default=50, help="冷启动预热步数，在此期间使用真实数据更新隐藏状态，并且不参与最后指标计算")
    # TabPFN 参数
    parser.add_argument("--context_mode", type=str, default="mixed", choices=["single", "mixed", "independent"], help="TabPFN 上下文策略")
    parser.add_argument("--seq_len", type=int, default=1, help="TabPFN 滑动窗口长度")
    parser.add_argument("--context_size", type=int, default=8000, help="TabPFN 使用的最大样本数")
    parser.add_argument("--n_context_trajs", type=int, default=1, help="mixed 模式下用作 Context 的真实轨迹条数")
    
    # LSTM/DeepAR 参数
    parser.add_argument("--model_path", type=str, default="", help="模型的 .pth 路径")
    
    return parser.parse_args()


# TabPFN 辅助函数
def create_sliding_windows_3d(X_train_3d, X_eval_3d, Y_train_3d, Y_eval_3d, seq_len):
    """处理 3D 轨迹数据，切分为滑动窗口 (专供 TabPFN 使用)"""
    N_episodes = X_train_3d.shape[0]
    num_eval = X_eval_3d.shape[1]

    X_train_seq_list, Y_train_seq_list = [], []
    X_eval_seq_list, Y_eval_seq_list = [], []

    for ep in range(N_episodes):
        X_total = np.concatenate((X_train_3d[ep], X_eval_3d[ep]), axis=0)
        Y_total = np.concatenate((Y_train_3d[ep], Y_eval_3d[ep]), axis=0)
        
        X_seq, Y_seq = [], []
        for i in range(len(X_total) - seq_len + 1):
            X_seq.append(X_total[i : i + seq_len].flatten())
            Y_seq.append(Y_total[i + seq_len - 1])
            
        X_train_seq_list.append(np.array(X_seq)[:-num_eval])
        Y_train_seq_list.append(np.array(Y_seq)[:-num_eval])
        X_eval_seq_list.append(np.array(X_seq)[-num_eval:])
        Y_eval_seq_list.append(np.array(Y_seq)[-num_eval:])

    return (np.array(X_train_seq_list), np.array(Y_train_seq_list), 
            np.array(X_eval_seq_list), np.array(Y_eval_seq_list))
    
    
def rolling_predict_tabpfn(teacher, X_val_3d_scaled, seq_len, nx, nu, device, warmup_steps):
    """
    TabPFN 自回归滚动预测
    """
    N_episodes, num_val_steps, total_flat_dim = X_val_3d_scaled.shape
    feature_dim = nu + nx 
    
    current_windows = X_val_3d_scaled[:, warmup_steps, :].copy()
    all_preds_scaled = []
    all_quants_scaled = []
    
    # 获取 95% 置信区间对应的分位数
    target_quantiles = [0.025, 0.975]
    
    for t in tqdm(range(warmup_steps, num_val_steps), desc="TabPFN Rolling Steps"):
        curr_tensor = torch.tensor(current_windows, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds_scaled = teacher(curr_tensor).cpu().numpy()
            quants_scaled = teacher.predict_quantiles(curr_tensor, target_quantiles).cpu().numpy()
        all_preds_scaled.append(preds_scaled)
        all_quants_scaled.append(quants_scaled)
        
        if t < num_val_steps - 1:
            next_windows = np.empty_like(current_windows)
            if seq_len > 1:
                next_windows[:, :-feature_dim] = current_windows[:, feature_dim:]
            
            true_next_u_scaled = X_val_3d_scaled[:, t + 1, -feature_dim : -feature_dim + nu]
            next_step_features = np.concatenate([true_next_u_scaled, preds_scaled], axis=1)
            next_windows[:, -feature_dim:] = next_step_features
            current_windows = next_windows
            
    preds_out = np.transpose(np.array(all_preds_scaled), (1, 0, 2))
    quants_out = np.transpose(np.array(all_quants_scaled), (1, 0, 2, 3)) # shape: [N_ep, T, nx, 2]
    return preds_out, quants_out

# LSTM 和 DeepAR 的通用辅助函数
def rolling_predict_nn(model, X_val, nx, nu, device, model_type, warmup_steps):
    """
    自回归滚动预测
    """
    N_episodes, T_steps, _ = X_val.shape
    model.eval()
    
    num_layers = model.lstm.num_layers
    hidden_size = model.lstm.hidden_size
    
    h = torch.zeros(num_layers, N_episodes, hidden_size).to(device)
    c = torch.zeros(num_layers, N_episodes, hidden_size).to(device)
    
    # warm-up 阶段
    with torch.no_grad():
        for t in range(warmup_steps):
            x_input = torch.tensor(X_val[:, t:t+1, :], dtype=torch.float32).to(device)
            if model_type == "DeepAR":               
                _, _, (h, c) = model(x_input, (h, c))
            else:
                _, (h, c) = model.lstm(x_input, (h, c))
    
    # Rolling 阶段
    preds_mu = []
    preds_sigma = []
    
    # 提取 t=0 时的初始输入
    curr_x = torch.tensor(X_val[:, warmup_steps:warmup_steps+1, :], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for t in tqdm(range(warmup_steps, T_steps), desc=f"{model_type} Rolling Steps"):
            if model_type == "DeepAR":
                mu, sigma, (h, c) = model(curr_x, (h, c))
                preds_mu.append(mu.cpu().numpy())
                preds_sigma.append(sigma.cpu().numpy())
                pred_nx = mu
            else:
                out, (h, c) = model.lstm(curr_x, (h, c))
                pred_nx = model.fc(out) 
                preds_mu.append(pred_nx.cpu().numpy())
                
            # 拼接未来真实的控制信号
            if t < T_steps - 1:
                true_u = torch.tensor(X_val[:, t+1:t+2, :nu], dtype=torch.float32).to(device)
                curr_x = torch.cat([true_u, pred_nx], dim=2)
                
    if model_type == "DeepAR":
        return np.concatenate(preds_mu, axis=1), np.concatenate(preds_sigma, axis=1)
    else:
        return np.concatenate(preds_mu, axis=1), None


def single_step_predict_nn(model, X_val, device, model_type, warmup_steps):
    """单步预测"""
    model.eval()
    X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    h = torch.zeros(model.lstm.num_layers, X_t.shape[0], model.lstm.hidden_size).to(device)
    c = torch.zeros(model.lstm.num_layers, X_t.shape[0], model.lstm.hidden_size).to(device)
    
    with torch.no_grad():
        X_warmup = X_t[:, :warmup_steps, :]
        X_eval = X_t[:, warmup_steps:, :]
        
        if model_type == "DeepAR":
            _, _, (h, c) = model(X_warmup, (h, c))
            mu, sigma, _ = model(X_eval, (h, c))
            return mu.cpu().numpy(), sigma.cpu().numpy()
        else:
            _, (h, c) = model.lstm(X_warmup, (h, c))
            out, _ = model.lstm(X_eval, (h, c))
            preds = model.fc(out)
            return preds.cpu().numpy(), None
        

def main():
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    warmup_steps = args.warmup_steps
    dataset_name = args.dataset
    data_dir = f"data/processed/{dataset_name}"
    
    if args.model_type == "TabPFN":
        ctx_str = f"ctx{args.n_context_trajs}" if args.context_mode == "mixed" else "ctx1_each"
        exp_name = f"TabPFN_{args.context_mode}_{ctx_str}_{args.pred_mode}_seq{args.seq_len}"
    else:
        if not args.model_path: raise ValueError(f"评估 {args.model_type} 时必须提供 --model_path")
        file_name = os.path.splitext(os.path.basename(args.model_path))[0]
        model_name = os.path.basename(os.path.dirname(args.model_path)) + "_" + file_name
        exp_name = f"{args.model_type}_{model_name}_{args.pred_mode}"
        
    results_dir = os.path.join(f"results/{dataset_name}/unified_eval", args.eval_set, exp_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. 加载评估数据
    print(f"Loading {args.eval_set} data for {dataset_name}...")
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train_raw = np.load(os.path.join(data_dir, "Y_train.npy"))
    X_eval_raw = np.load(os.path.join(data_dir, f"X_{args.eval_set}.npy"))
    Y_eval_raw = np.load(os.path.join(data_dir, f"Y_{args.eval_set}.npy"))

    N_episodes, num_eval_steps_total, _ = X_eval_raw.shape
    nx = Y_eval_raw.shape[-1]
    nu = X_eval_raw.shape[-1] - nx

    if warmup_steps >= num_eval_steps_total:
        raise ValueError("warmup_steps 必须小于评估集的总时间步长！")
    
    # 3. 统一全局标准化
    print("Loading global Scalers from local storage...")
    scaler_x_path = os.path.join(data_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(data_dir, "scaler_Y.pkl")
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scalers not found in {data_dir}.")
        
    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)
    
    X_train_scaled_raw = scaler_X.transform(X_train_raw.reshape(-1, X_train_raw.shape[-1])).reshape(X_train_raw.shape)
    X_eval_scaled_raw = scaler_X.transform(X_eval_raw.reshape(-1, X_eval_raw.shape[-1])).reshape(X_eval_raw.shape)
    Y_train_scaled_raw = scaler_Y.transform(Y_train_raw.reshape(-1, Y_train_raw.shape[-1])).reshape(Y_train_raw.shape)

    # 4. 预测推理
    np.random.seed(42)
    torch.manual_seed(42)
    Y_pred_all_unscaled = None
    Y_sigma_all_unscaled = None # 专供 DeepAR
    Y_quant_all_unscaled = None # 专供 TabPFN
    
    if args.model_type == "TabPFN":
        print(f"Initializing TabPFN (Mode: {args.pred_mode} | Context: {args.context_mode})...")
        X_train_3d_sc, Y_train_3d_sc, X_eval_3d_sc, _ = create_sliding_windows_3d(
            X_train_scaled_raw, X_eval_scaled_raw, Y_train_scaled_raw, Y_eval_raw, args.seq_len
        )
        teacher = TabPFNWrapper(device=device, n_ensemble=3) # type: ignore
        target_quantiles = [0.025, 0.975]
        
        if args.context_mode == "mixed":
            n_ctx = min(args.n_context_trajs, X_train_3d_sc.shape[0])
            pool_X = X_train_3d_sc[:n_ctx].reshape(-1, X_train_3d_sc.shape[-1])
            pool_Y = Y_train_3d_sc[:n_ctx].reshape(-1, nx)
            
            c_size = min(args.context_size, len(pool_X))
            if len(pool_X) > c_size:
                indices = np.random.choice(len(pool_X), c_size, replace=False)
            else:
                indices = np.arange(len(pool_X))
                
            teacher.fit(pool_X[indices], pool_Y[indices])
            
            if args.pred_mode == "single_step":
                X_eval_eval = X_eval_3d_sc[:, warmup_steps:, :]
                X_eval_tensor = torch.tensor(X_eval_eval.reshape(-1, X_eval_eval.shape[-1]), dtype=torch.float32).to(device)
                with torch.no_grad(): 
                    Y_pred_all_scaled = teacher(X_eval_tensor).cpu().numpy()
                    Y_quant_all_scaled = teacher.predict_quantiles(X_eval_tensor, target_quantiles).cpu().numpy()
            else:
                preds_3d, quants_3d = rolling_predict_tabpfn(teacher, X_eval_3d_sc, args.seq_len, nx, nu, device, warmup_steps)
                Y_pred_all_scaled = preds_3d.reshape(-1, nx)
                Y_quant_all_scaled = quants_3d.reshape(-1, nx, 2)
           
        elif args.context_mode == "independent":
            preds_list = []
            quants_list = []
            for ep in tqdm(range(N_episodes), desc="Independent Episodes"):
                pool_X = X_train_3d_sc[ep]
                pool_Y = Y_train_3d_sc[ep]
                
                c_size = min(args.context_size, len(pool_X))
                indices = np.random.choice(len(pool_X), c_size, replace=False) if len(pool_X) > c_size else np.arange(len(pool_X))
                teacher.fit(pool_X[indices], pool_Y[indices])
                
                if args.pred_mode == "single_step":
                    X_eval_ep_eval = X_eval_3d_sc[ep, warmup_steps:, :]
                    X_eval_ep_tensor = torch.tensor(X_eval_ep_eval, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        preds_list.append(teacher(X_eval_ep_tensor).cpu().numpy())
                        quants_list.append(teacher.predict_quantiles(X_eval_ep_tensor, target_quantiles).cpu().numpy())
                else:
                    X_eval_ep_3d_scaled = X_eval_3d_sc[ep:ep+1]
                    pred_3d, quant_3d = rolling_predict_tabpfn(teacher, X_eval_ep_3d_scaled, args.seq_len, nx, nu, device, warmup_steps)
                    preds_list.append(pred_3d[0])
                    quants_list.append(quant_3d[0])
            Y_pred_all_scaled = np.concatenate(preds_list, axis=0)
            Y_quant_all_scaled = np.concatenate(quants_list, axis=0)
            
        Y_pred_all_unscaled = scaler_Y.inverse_transform(Y_pred_all_scaled)
        Y_quant_all_unscaled = np.zeros_like(Y_quant_all_scaled)
        Y_quant_all_unscaled[..., 0] = scaler_Y.inverse_transform(Y_quant_all_scaled[..., 0]) # 2.5% 下界
        Y_quant_all_unscaled[..., 1] = scaler_Y.inverse_transform(Y_quant_all_scaled[..., 1]) # 97.5% 上界
        
    elif args.model_type in ["LSTM", "DeepAR"]:
        print(f"Loading {args.model_type} from {args.model_path} (Mode: {args.pred_mode})...")
        model = torch.load(args.model_path, map_location=device, weights_only=False)
        
        if args.pred_mode == "single_step":
            preds_3d, sigma_3d = single_step_predict_nn(model, X_eval_scaled_raw, device, args.model_type, warmup_steps)
        else:
            preds_3d, sigma_3d = rolling_predict_nn(model, X_eval_scaled_raw, nx, nu, device, args.model_type, warmup_steps)
        # 反标准化
        Y_pred_all_unscaled = scaler_Y.inverse_transform(preds_3d.reshape(-1, nx))
        if sigma_3d is not None:
            Y_sigma_all_unscaled = sigma_3d.reshape(-1, nx) * scaler_Y.scale_

    # 5. 计算指标
    print("\nComputing Evaluation Metrics...")
    Y_eval_raw_eval = Y_eval_raw[:, warmup_steps:, :]
    Y_eval_flat_unscaled = Y_eval_raw_eval.reshape(-1, nx)
    mse = np.mean((Y_pred_all_unscaled - Y_eval_flat_unscaled) ** 2)
    print(f">>> {args.eval_set} MSE: {mse:.6f} <<<")
    
    # 概率预测评估指标
    picp, mpiw = None, None
    if Y_sigma_all_unscaled is not None or Y_quant_all_unscaled is not None:
        if args.model_type == "DeepAR":
            # DeepAR 的 95% 置信区间: mu ± 1.96 * sigma
            lower_bound = Y_pred_all_unscaled - 1.96 * Y_sigma_all_unscaled # type: ignore
            upper_bound = Y_pred_all_unscaled + 1.96 * Y_sigma_all_unscaled # type: ignore
        elif args.model_type == "TabPFN":
            # TabPFN 原生输出的分位数
            lower_bound = Y_quant_all_unscaled[..., 0] # type: ignore
            upper_bound = Y_quant_all_unscaled[..., 1] # type: ignore
        # 1. 计算 PICP (覆盖率)
        is_covered = (Y_eval_flat_unscaled >= lower_bound) & (Y_eval_flat_unscaled <= upper_bound)
        picp = np.mean(is_covered) * 100.0
        # 2. 计算 MPIW (区间宽度)
        mpiw = np.mean(upper_bound - lower_bound) # type: ignore
        
        print(f">>> PICP (Coverage): {picp:.2f}% | MPIW (Width): {mpiw:.6f} <<<")
            
    # 6. 保存实验日志
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_file_path = os.path.join(project_root, "forecast_results.txt")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{current_time}] Data: {dataset_name} | Set: {args.eval_set} | Eval: {exp_name} | MSE: {mse:.6f}"
    if picp is not None and mpiw is not None:
        log_msg += f" | PICP: {picp:.2f}% | MPIW: {mpiw:.6f}"
    log_msg += "\n"
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(log_msg)
    
    # 7. 绘图
    max_plots = min(args.n_plots, N_episodes)
    print(f"Generating {max_plots} Trajectory Plots...")
    
    num_eval_steps = num_eval_steps_total - warmup_steps
    for ep in range(max_plots):
        start_idx = ep * num_eval_steps
        end_idx = (ep + 1) * num_eval_steps
        
        Y_pred_plot = Y_pred_all_unscaled[start_idx:end_idx] # type: ignore
        Y_true_plot = Y_eval_raw_eval[ep]
        if Y_sigma_all_unscaled is not None:
            Y_sigma_plot = Y_sigma_all_unscaled[start_idx:end_idx]
        if Y_quant_all_unscaled is not None:
            Y_quant_plot = Y_quant_all_unscaled[start_idx:end_idx]
            
        plt.figure(figsize=(12, 3 * nx))
        t_vec = np.arange(warmup_steps, num_eval_steps_total)
        
        for i in range(nx):
            plt.subplot(nx, 1, i + 1)
            plt.plot(t_vec, Y_true_plot[:, i], label=f'True $x_{i+1}$', color='black', linewidth=1.5)
            
            # 判断标号文字
            if args.model_type == "DeepAR":
                pred_label = fr'{args.model_type} Mean $\mu$'
            elif args.model_type == "TabPFN":
                pred_label = fr'{args.model_type} Median Predict'
            else:
                pred_label = f'{args.model_type} Predicted $x_{i+1}$'
            plt.plot(t_vec, Y_pred_plot[:, i], label=pred_label, color='red', linestyle='--', linewidth=1.5)
            
            # DeepAR 95% 置信区间 (2-Sigma)
            if args.model_type == "DeepAR" and Y_sigma_all_unscaled is not None:
                lower_bound = Y_pred_plot[:, i] - 2 * Y_sigma_plot[:, i]
                upper_bound = Y_pred_plot[:, i] + 2 * Y_sigma_plot[:, i]
                plt.fill_between(t_vec, lower_bound, upper_bound, color='red', alpha=0.2, label=r'95% Confidence Interval')
            # TabPFN 分位数边界区间
            elif args.model_type == "TabPFN" and Y_quant_all_unscaled is not None:
                lower_bound = Y_quant_plot[:, i, 0] # 2.5%
                upper_bound = Y_quant_plot[:, i, 1] # 97.5%
                plt.fill_between(t_vec, lower_bound, upper_bound, color='blue', alpha=0.2, label=r'95% Confidence Interval')
            plt.ylabel(f'$x_{i+1}$')
            
            if i == 0:
                ctx_info = f" | Ctx: {args.n_context_trajs}" if args.model_type == "TabPFN" and args.context_mode == "mixed" else ""
                plt.title(f'[{args.model_type}] {args.eval_set} (Data: {dataset_name} | Mode: {args.pred_mode}{ctx_info} | Ep: {ep})')
            if i == nx - 1:
                plt.xlabel(f'{args.eval_set.capitalize()} Samples (Time Steps)')
                
            plt.legend(loc='upper right')
            plt.grid(True)
            
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"ep_{ep}.png"))
        plt.close()
        
    print(f"Results saved to {results_dir}")

if __name__ == "__main__":
    main()