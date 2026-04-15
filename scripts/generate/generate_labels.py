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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="four_tank", help="数据集名称 (van_der_pol, four_tank, pendulum)")
    parser.add_argument("--seq_len", type=int, default=20, help="sliding window length")
    parser.add_argument("--context_size", type=int, default=8000, help="TabPFN最大上下文样本数")
    parser.add_argument("--n_splits", type=int, default=5, help="交叉预测的折数 (K-Fold)")
    parser.add_argument("--batch_size", type=int, default=512, help="推理批次大小")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def create_sliding_windows_3d(X_3d, Y_3d, seq_len):
    """
    针对 3D 多轨迹数据 [N_episodes, T_steps, Features] 的滑动窗口生成器。
    严格保证在轨迹内部滑动，绝不跨越轨迹边界！
    """
    if seq_len == 1:
        return X_3d.reshape(-1, X_3d.shape[-1]), Y_3d.reshape(-1, Y_3d.shape[-1])
    
    X_seq_list, Y_seq_list = [], []
    N_episodes = X_3d.shape[0]
    
    for ep in range(N_episodes):
        X_ep = X_3d[ep]
        Y_ep = Y_3d[ep]
        # 在单条轨迹内部切片，这会合规地丢掉该轨迹开头的 (seq_len - 1) 个点
        for i in range(len(X_ep) - seq_len + 1):
            X_seq_list.append(X_ep[i : i + seq_len].flatten())
            Y_seq_list.append(Y_ep[i + seq_len - 1])
            
    return np.array(X_seq_list), np.array(Y_seq_list)

def main():
    args = get_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 路径准备
    data_dir = f"data/processed/{args.dataset}"
    save_dir = os.path.join(data_dir, "distill")
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. 加载 3D 原始训练数据
    print(f"Loading raw train data for {args.dataset}...")
    X_train_raw = np.load(os.path.join(data_dir, "X_train.npy"))  # [50, 2000, 3]
    Y_train_raw = np.load(os.path.join(data_dir, "Y_train.npy"))  # [50, 2000, 2]
    
    # 3. 严格的 3D 滑动窗口处理
    print(f"Applying 3D sliding window (seq_len={args.seq_len})...")
    X_seq, Y_seq = create_sliding_windows_3d(X_train_raw, Y_train_raw, args.seq_len)
    print(f"Processed Sequence Shape: X={X_seq.shape}, Y={Y_seq.shape}")
    
    # 4. 数据标准化 (保存 Scaler 供后续 Student 模型使用)
    print("Fitting Scaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_seq)
    
    scaler_path = os.path.join(save_dir, f"scaler_seq{args.seq_len}.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")
    
    # 5. 初始化预测容器
    N_samples, out_dim = Y_seq.shape
    quantiles = np.linspace(0.025, 0.975, 39).tolist()
    
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
            
        # 将预测结果填入对应的全局索引位置
        Y_teacher_point[predict_idx] = np.concatenate(point_preds_list, axis=0)
        Y_teacher_quantiles[predict_idx] = np.concatenate(quantile_preds_list, axis=0)
        
    # 将 2D 扁平数组恢复成严格的 3D 轨迹结构
    N_episodes = X_train_raw.shape[0]
    T_valid_steps = X_train_raw.shape[1] - args.seq_len + 1
    
    Y_teacher_point_3d = Y_teacher_point.reshape(N_episodes, T_valid_steps, -1)
    Y_teacher_quantiles_3d = Y_teacher_quantiles.reshape(N_episodes, T_valid_steps, out_dim, len(quantiles))
    
    # 保存路径
    p_path = os.path.join(save_dir, f"teacher_preds_seq{args.seq_len}.npy")
    q_path = os.path.join(save_dir, f"teacher_quantiles_seq{args.seq_len}.npy")
    q_val_path = os.path.join(save_dir, "quantile_values.npy")
    
    np.save(p_path, Y_teacher_point_3d)
    np.save(q_path, Y_teacher_quantiles_3d)
    np.save(q_val_path, np.array(quantiles))
    
    print(f"\nOriginal 3D Data Shape: {X_train_raw.shape}")
    print(f"Teacher Labels Shape Saved: {Y_teacher_point_3d.shape}")
    print(f"Teacher Quantiles Shape Saved: {Y_teacher_quantiles_3d.shape}")
    
    print(f"\nDistillation Labels Generated Successfully!")
    print(f"Point preds saved to: {p_path}")
    
if __name__ == "__main__":
    main()