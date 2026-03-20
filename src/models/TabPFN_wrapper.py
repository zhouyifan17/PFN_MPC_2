import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from tabpfn import TabPFNRegressor

class TabPFNWrapper(nn.Module):
    def __init__(self, device='cpu', n_ensemble=3):
        """
        将 TabPFN 包装成 Pytorch 模型的形式，统一接口
        支持点预测 (Forward) 和 分位数/分布预测 (For Distillation)
        """
        super().__init__()
        self.device = device
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        
        # 权重路径
        model_filename = "tabpfn-v2.5-regressor-v2.5_default.ckpt"
        model_path_local = os.path.join(project_root, "checkpoints", "TabPFN", model_filename)
        
        if not os.path.exists(model_path_local):
            print(f"[Warning] TabPFN weight not found at {model_path_local}.")
            
        self.regressor = TabPFNRegressor(
            device=self.device, 
            n_estimators=n_ensemble,
            model_path=model_path_local,
            ignore_pretraining_limits=True
        )
        self.model = MultiOutputRegressor(self.regressor)
        self.is_fitted = False
        
    def fit(self, X, Y):
        """
        加载 Context 数据
        Args:
            X: Tensor or Numpy, [N, Input_Dim] or [N, Seq, Input_Dim]
            Y: Tensor or Numpy, [N, Output_Dim]
        """
        if torch.is_tensor(X):
            X = X.detach().cpu().numpy()
        if torch.is_tensor(Y):
            Y = Y.detach().cpu().numpy()
        
        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)
            
        self.model.fit(X, Y)
        self.is_fitted = True
        
    def forward(self, x):
        """
        前向传播 (回归预测)
        Args:
            x: Tensor [Batch, Input_Dim] or [Batch, Seq, Feature]
        Returns:
            out: Tensor [Batch, Output_Dim]
        """
        if not self.is_fitted:
            raise RuntimeError("TabPFNWrapper must be fitted (load context) before forward pass!")
        
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
            
        # 序列维度对齐
        if x_np.ndim == 3:
            x_np = x_np.reshape(x_np.shape[0], -1)
        elif x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            
        y_pred = self.model.predict(x_np)
        out = torch.from_numpy(y_pred).float().to(self.device)
        
        return out
    
    def predict_quantiles(self, x, quantiles):
        """
        获取预测分布的分位数，用于MDN, DeepAR等概率模型的蒸馏
        Args:
            x: Tensor [Batch, Input_Dim]
            quantiles: list of floats, e.g. [0.1, 0.2, ..., 0.9]
        Returns:
            out: Tensor [Batch, Output_Dim, Num_Quantiles]        
        """
        if not self.is_fitted:
            raise RuntimeError("TabPFNWrapper must be fitted first!")
        
        if torch.is_tensor(x):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x
        
        if x_np.ndim == 3:
            x_np = x_np.reshape(x_np.shape[0], -1)
        elif x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            
        all_preds = []
        
        for i, estimator in enumerate(self.model.estimators_):
            q_list = estimator.predict(
                x_np,
                output_type="quantiles",
                quantiles=quantiles
            )
            q_arrays = [q.flatten() if q.ndim > 1 else q for q in q_list]
            q_pred_np = np.stack(q_arrays, axis=1)
            all_preds.append(q_pred_np)
        
        out_np = np.stack(all_preds, axis=1)
        return torch.from_numpy(out_np).float().to(self.device)