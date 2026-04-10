import os
import torch
import sys
import numpy as np
import argparse
import matplotlib.pyplot as plt
from scipy.io import savemat
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath("."))
from src.models import LSTMModel
from src.models import DeepARModel
from src.engine.trainer import Trainer

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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = f"data/processed/{args.dataset}"
    model_type = args.model_type
    
    # 1. 加载 3D 训练数据
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
        
    # 验证集使用真实物理数据
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
    
    # 3. 动态构建保存路径
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

        t_pred_t = torch.tensor(t_pred, dtype=torch.float32).contiguous().to(device)
        t_quant_t = torch.tensor(t_quant, dtype=torch.float32).contiguous().to(device)

    # 5. 转换为 PyTorch Tensor
    X_train_t = torch.tensor(X_train, dtype=torch.float32).contiguous().to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).contiguous().to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).contiguous().to(device)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32).contiguous().to(device)

    # 6. 动态获取特征维度 (兼容 3D 轨迹与 2D 点云)
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
    combinations = [20] 
    N_epoch = 40000
    ablation_results = []

    for idx_i, n_hidden in enumerate(combinations, start=1):
        print(f"\n--- Training {model_type} (Hidden: {n_hidden}) ---")
        
        # 动态实例化模型
        if args.model_type == "LSTM":
            model = LSTMModel(input_size, n_hidden, output_size)
        elif args.model_type == "DeepAR":
            model = DeepARModel(input_size, n_hidden, output_size)
        elif args.model_type == "FFN":
            # 预留给未来：model = FFNModel(input_size, n_hidden, output_size)
            raise NotImplementedError("FFN model is not yet implemented.")
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

        trainer = Trainer(model, device, ckpt_dir, model_name=f"model_{model_type}_idx_{idx_i}.pth")

        exp_info = {
            "Dataset": args.dataset,
            "Model": model_type,
            "Mode": args.mode,
            "N_Trajs": n_train,
            "Ctx_Trajs": args.n_context_trajs if args.data_source != "real" else "N/A",
            "Hidden_Size": n_hidden,
            "Epochs": N_epoch
        }
        
        train_loss, val_loss, t_elapsed = trainer.train(
            X_train_t, Y_train_t, X_val_t, Y_val_t, 
            optimizer, scheduler, num_epochs=N_epoch, log_interval=2000,
            experiment_info=exp_info,
            t_pred=t_pred_t, t_quant=t_quant_t, distill_weight=args.distill_weight
        )

        np.save(os.path.join(results_dir, f"train_loss_idx_{idx_i}.npy"), train_loss)
        np.save(os.path.join(results_dir, f"val_loss_idx_{idx_i}.npy"), val_loss)

        ablation_results.append({
            'idx': idx_i,
            'hidden_state_dimension': n_hidden,
            'train_loss': np.array(train_loss),
            'val_loss': np.array(val_loss),
            'T_train': t_elapsed
        })

    ablation_mat = {f'result_{i}': res for i, res in enumerate(ablation_results)}
    savemat(os.path.join(results_dir, "ablation_results.mat"), ablation_mat)
    print("All experiments finished and saved!")

if __name__ == "__main__":
    main()