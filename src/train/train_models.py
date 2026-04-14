import os
import torch
import sys
import numpy as np
import argparse
import joblib
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.models import LSTMModel
from src.models import DeepARModel
from src.train.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description="Train Models with or without Distillation")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--model_type", type=str, default="LSTM", choices=["LSTM", "DeepAR", "FFN"], help="选择要训练的底层模型架构")
    parser.add_argument("--dataset", type=str, default="van_der_pol", help="数据集名称 (van_der_pol, four_tank, pendulum)")
    parser.add_argument("--mode", type=str, default="standard", choices=["standard", "distill"])
    parser.add_argument("--data_source", type=str, default="real", choices=["real", "surrogate", "point_cloud"], help="训练数据的来源")
    parser.add_argument("--seq_len", type=int, default=1, help="Sequence length used for teacher labels")
    parser.add_argument("--distill_weight", type=float, default=0.5, help="Weight of the distillation loss")
    parser.add_argument("--n_train", type=int, default=50, help="Number of trajectories to use for training (1 to 50)")
    parser.add_argument("--n_context_trajs", type=int, default=1, help="生成人工数据时使用的 context 轨迹数")
    # 模型超参数
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM/DeepAR的层数")
    parser.add_argument("--hidden_size", type=int, default=20, help="隐藏层单元数")
    
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = f"data/processed/{args.dataset}"
    model_type = args.model_type
    
    # 1. 加载训练集和验证集
    if args.data_source == "real":
        X_train_3d = np.load(os.path.join(data_dir, "X_train.npy"))
        Y_train_3d = np.load(os.path.join(data_dir, "Y_train.npy"))
    elif args.data_source == "surrogate":
        syn_dir = os.path.join(data_dir, "synthetic")
        X_train_3d = np.load(os.path.join(syn_dir, f"X_train_surrogate_ctx{args.n_context_trajs}.npy"))
        Y_train_3d = np.load(os.path.join(syn_dir, f"Y_train_surrogate_ctx{args.n_context_trajs}.npy"))
    elif args.data_source == "point_cloud":
        syn_dir = os.path.join(data_dir, "synthetic")
        X_train_3d = np.load(os.path.join(syn_dir, f"X_train_point_cloud_ctx{args.n_context_trajs}.npy"))
        Y_train_3d = np.load(os.path.join(syn_dir, f"Y_train_point_cloud_ctx{args.n_context_trajs}.npy"))
        
    X_val_3d = np.load(os.path.join(data_dir, "X_val.npy"))
    Y_val_3d = np.load(os.path.join(data_dir, "Y_val.npy"))
    
    # 2. 轨迹数量切分 (--n_train 控制)
    T_steps = X_val_3d.shape[1]
    if args.data_source == "point_cloud":
        equiv_n_points = args.n_train * T_steps
        n_train_points = min(equiv_n_points, X_train_3d.shape[0])
        X_train = X_train_3d[:n_train_points]
        Y_train = Y_train_3d[:n_train_points]
    else:
        n_train = min(args.n_train, X_train_3d.shape[0])
        X_train = X_train_3d[:n_train]
        Y_train = Y_train_3d[:n_train]
        
    X_val = X_val_3d
    Y_val = Y_val_3d
    
    # 加载本地 Scaler 对数据进行标准化
    scaler_x_path = os.path.join(data_dir, "scaler_X.pkl")
    scaler_y_path = os.path.join(data_dir, "scaler_Y.pkl")
    
    if not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path):
        raise FileNotFoundError(f"Scalers not found in {data_dir}.")
    
    scaler_X = joblib.load(scaler_x_path)
    scaler_Y = joblib.load(scaler_y_path)
    X_train = scaler_X.transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    Y_train = scaler_Y.transform(Y_train.reshape(-1, Y_train.shape[-1])).reshape(Y_train.shape)
    X_val = scaler_X.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    Y_val = scaler_Y.transform(Y_val.reshape(-1, Y_val.shape[-1])).reshape(Y_val.shape)
    
    # 3. 构建保存路径
    mode_str = "distill" if args.mode == "distill" else "standard"
    if args.data_source in ["surrogate", "point_cloud"]:
        save_prefix = f"{args.data_source}_ctx{args.n_context_trajs}_{mode_str}_N{n_train}"
    else:
        save_prefix = f"{args.data_source}_{mode_str}_N{n_train}"
    
    ckpt_dir = f"checkpoints/{model_type}/{args.dataset}/{save_prefix}"
    results_dir = f"results/{args.dataset}/{model_type}/{save_prefix}"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"Using device: {device} | Dataset: {args.dataset.upper()} | Source: {args.data_source.upper()} | Mode: {args.mode.upper()}")
    if args.data_source in ["surrogate", "point_cloud"]:
        print(f"Context Trajs Used for Generation: {args.n_context_trajs}")
    print(f"Using {n_train} {args.data_source} trajectories for training, {len(X_val)} real trajectories for validation.")
    print(f"Checkpoints will be saved to: {ckpt_dir}")

    # 4. 蒸馏模式，加载老师标签
    t_pred, t_quant, t_pred_t, t_quant_t = None, None, None, None
    if args.mode == "distill":
        distill_dir = os.path.join(data_dir, "distill")
        t_pred_full = np.load(os.path.join(distill_dir, f"teacher_preds_seq{args.seq_len}.npy"))
        t_quant_full = np.load(os.path.join(distill_dir, f"teacher_quantiles_seq{args.seq_len}.npy"))
        
        t_pred = t_pred_full[:n_train]
        t_quant = t_quant_full[:n_train]

        # 教师标签标准化
        t_pred = scaler_Y.transform(t_pred.reshape(-1, t_pred.shape[-1])).reshape(t_pred.shape)
        n_quantiles = t_quant.shape[-1]
        scaled_quant = np.zeros_like(t_quant)
        for q in range(n_quantiles):
            q_data = t_quant[..., q] 
            scaled_quant[..., q] = scaler_Y.transform(q_data.reshape(-1, q_data.shape[-1])).reshape(q_data.shape)
        t_quant = scaled_quant
        
        t_pred_t = torch.tensor(t_pred, dtype=torch.float32).contiguous().to(device)
        t_quant_t = torch.tensor(t_quant, dtype=torch.float32).contiguous().to(device)

    # 5. 转换为 PyTorch Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).contiguous().to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).contiguous().to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).contiguous().to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).contiguous().to(device)

    # 6. 动态获取特征维度
    if len(X_train_t.shape) == 3:
        input_size = X_train_t.shape[2]
        output_size = Y_train_t.shape[2]
    elif len(X_train_t.shape) == 2:
        input_size = X_train_t.shape[1]
        output_size = Y_train_t.shape[1]
        
        if args.data_source == "point_cloud":
            X_train_t = X_train_t.unsqueeze(1)
            Y_train_t = Y_train_t.unsqueeze(1)
    else:
        raise ValueError(f"无法处理的数据维度: {X_train_t.shape}")
    
    # 7. 循环训练
    n_layers = args.num_layers
    n_hidden = args.hidden_size
    N_epoch = 40000

    print(f"\n--- Training {model_type} | Layers: {n_layers} | Hidden: {n_hidden} ---")
        
    # 实例化模型
    if args.model_type == "LSTM":
        model = LSTMModel(input_size, n_hidden, output_size, num_layers=n_layers)
    elif args.model_type == "DeepAR":
        model = DeepARModel(input_size, n_hidden, output_size, num_layers=n_layers)
    elif args.model_type == "FFN":
        raise NotImplementedError("FFN model is not yet implemented.")
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

    model_filename = f"model_{model_type}_L{n_layers}_H{n_hidden}.pth"
    trainer = Trainer(model, device, ckpt_dir, model_name=model_filename)

    exp_info = {
        "Dataset": args.dataset,
        "Model": model_type,
        "Mode": args.mode,
        "N_Trajs": n_train,
        "Ctx_Trajs": args.n_context_trajs if args.data_source != "real" else "N/A",
        "Layers": n_layers,
        "Hidden_Size": n_hidden,
        "Epochs": N_epoch
    }
        
    train_loss, val_loss, t_elapsed = trainer.train(
        X_train_t, Y_train_t, X_val_t, Y_val_t, 
        optimizer, scheduler, num_epochs=N_epoch, log_interval=2000,
        experiment_info=exp_info,
        t_pred=t_pred_t, t_quant=t_quant_t, distill_weight=args.distill_weight
    )

    np.save(os.path.join(results_dir, f"train_loss_L{n_layers}_H{n_hidden}.npy"), train_loss)
    np.save(os.path.join(results_dir, f"val_loss_L{n_layers}_H{n_hidden}.npy"), val_loss)

    result_dict = {
        'layers': n_layers,
        'hidden': n_hidden,
        'final_val_mse': val_loss[-1] if len(val_loss)>0 else None,
        't_elapsed': t_elapsed
    }

    savemat(os.path.join(results_dir, f"result_L{n_layers}_H{n_hidden}.mat"), result_dict)
    print(f"Finished: {model_filename}")

if __name__ == "__main__":
    main()