import os
import sys
import torch
import numpy as np
import argparse
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

sys.path.append(".")
from src.models import TabPFNWrapper

def create_sliding_windows(X, Y, seq_len):
    """
    通用滑动窗口生成器
    """
    if seq_len == 1:
        return X, Y
    
    X_seq, Y_target = [], []
    for i in range(len(X) - seq_len + 1):
        window_x = X[i : i + seq_len].flatten()
        X_seq.append(window_x)
        Y_target.append(Y[i + seq_len - 1])
        
    return np.array(X_seq), np.array(Y_target)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="van_der_pol")
    parser.add_argument("--seq_len", type=int, default=1, help="sliding window length")
    parser.add_argument("--context_size", type=int, default=2048, help="TabPFN最大上下文样本数")
    parser.add_argument("--n_splits", type=int, default=5, help="交叉预测的折数 (K-Fold)")
    parser.add_argument("--batch_size", type=int, default=256, help="推理批次大小")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 路径准备
    data_dir = f"data/processed/{args.dataset}"
    save_dir = os.path.join(data_dir, "distill")
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 加载原始训练数据
    print(f"Loading raw train data for {args.dataset}...")
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy"))
    Y_train_raw = np.load(os.path.join(data_dir, "Y_train.npy"))
    
    # 3. 滑动窗口处理
    print(f"Applying sliding window (seq_len={args.seq_len})...")
    X_seq, Y_seq = create_sliding_windows(X_train_raw, Y_train_raw, args.seq_len)
    print(f"Processed Sequence Shape: X={X_seq.shape}, Y={Y_seq.shape}")
    # TODO: 这里有个小问题，那就是如果滑动窗口的长度 l 是大于1的，那么训练集最开始的 l-1 个数据点就会没有蒸馏标签，看看后面怎么解决
    
    # 4. 数据标准化 (保存 Scaler 供后续 Student 模型使用)
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seq)
    
    scaler_path = os.path.join(save_dir, f"scaler_seq{args.seq_len}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 5. 初始化预测容器
    N_samples, out_dim = Y_seq.shape
    quantiles = np.linspace(0.05, 0.95, 19).tolist()
    
    Y_teacher_point = np.zeros_like(Y_seq)
    Y_teacher_quantiles = np.zeros((N_samples, out_dim, len(quantiles)))
    
    # 6. K-Fold 交叉预测
    print(f"\nStarting {args.n_splits}-Fold Cross-Prediction...")
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    
    teacher = TabPFNWrapper(device=device, n_ensemble=3) # type:ignore
    
    for fold, (context_idx, predict_idx) in enumerate(kf.split(X_scaled)):
        print(f"--- Fold {fold+1}/{args.n_splits} ---")
        
        # 准备 Context 数据
        X_context_full = X_scaled[context_idx]
        Y_context_full = Y_seq[context_idx]
        
        current_context_size = min(args.context_size, len(X_context_full))
        sub_indices = np.random.choice(len(X_context_full), current_context_size, replace=False)
        
        X_context = X_context_full[sub_indices]
        Y_context = Y_context_full[sub_indices]
        
        teacher.fit(X_context, Y_context)
        X_predict = X_scaled[predict_idx]
        
        point_preds_list = []
        quantile_preds_list = []
        
        for i in tqdm(range(0, len(X_predict), args.batch_size), desc="Inferencing"):
            X_batch = torch.tensor(X_predict[i : i + args.batch_size], dtype=torch.float32).to(device)
            
            with torch.no_grad():
                # 点预测
                p_pred = teacher(X_batch).cpu().numpy()
                # 分位数预测
                q_pred = teacher.predict_quantiles(X_batch, quantiles).cpu().numpy()
                
            point_preds_list.append(p_pred)
            quantile_preds_list.append(q_pred)
            
        # 将预测结果填入对应的索引位置
        Y_teacher_point[predict_idx] = np.concatenate(point_preds_list, axis=0)
        Y_teacher_quantiles[predict_idx] = np.concatenate(quantile_preds_list, axis=0)
        
        # 保存结果
    p_path = os.path.join(save_dir, f"teacher_preds_seq{args.seq_len}.npy")
    q_path = os.path.join(save_dir, f"teacher_quantiles_seq{args.seq_len}.npy")
    q_val_path = os.path.join(save_dir, "quantile_values.npy")
    
    np.save(p_path, Y_teacher_point)
    np.save(q_path, Y_teacher_quantiles)
    np.save(q_val_path, np.array(quantiles))
    
    print(X_train_raw.shape)
    print(Y_teacher_point.shape)
    
    print(f"\nDistillation Labels Generated Successfully!")
    print(f"Point preds saved to: {p_path}")
    
if __name__ == "__main__":
    main()