import os
import time
import torch
import datetime

class Trainer:
    def __init__(self, model, device, save_dir, model_name="best_model.pth"):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, X_train, Y_train, X_val, Y_val, optimizer, scheduler, num_epochs, 
              log_interval=1000, experiment_info=None,
              t_pred=None, t_quant=None, distill_weight=0.5):
        
        train_loss_hist = []
        val_loss_hist = []
        best_val_loss = float('inf')
        best_epoch = 0

        t0 = time.time()

        for epoch in range(num_epochs):
            # 训练阶段
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_train)
            
            # 判断模型的输出种类: 如果是 tuple 并且长度为3，说明是DeepAR
            is_deepar = isinstance(outputs, tuple) and len(outputs) == 3
            
            # 基础 loss 计算
            if is_deepar:
                mu, sigma, _ = outputs
                # DeepAR 使用 NLL
                loss_task = torch.mean(torch.log(sigma) + 0.5 *((Y_train - mu) / sigma)**2)
            else:
                Y_pred = outputs
                loss_task = torch.mean((Y_pred - Y_train) ** 2) / torch.mean(Y_train ** 2)
                
            # 知识蒸馏 loss 计算
            if t_pred is not None and t_quant is not None:
                if is_deepar:
                    loss_distill_mu = torch.nn.functional.mse_loss(mu, t_pred)
                    lower_bound = t_quant[:, :, :, 0]
                    upper_bound = t_quant[:, :, :, -1]
                    t_sigma_target = torch.clamp((upper_bound - lower_bound) / 4.0, min=1e-4)
                    loss_distill_sigma = torch.nn.functional.mse_loss(sigma, t_sigma_target)
                    loss_distill = loss_distill_mu + loss_distill_sigma
                else:
                    # LSTM 的蒸馏 loss: 拟合点预测，并加上分位数越界惩罚
                    loss_distill_point = torch.nn.functional.mse_loss(Y_pred, t_pred) # type: ignore
                    lower_bound = t_quant[:, :, :, 0]
                    upper_bound = t_quant[:, :, :, -1]
                    penalty_lower = torch.relu(lower_bound - Y_pred) ** 2
                    penalty_upper = torch.relu(Y_pred - upper_bound) ** 2
                    loss_bounds = torch.mean(penalty_lower + penalty_upper)
                    loss_distill = loss_distill_point + 0.5 * loss_bounds
                
                # 加权总loss
                loss_train = (1.0 - distill_weight) * loss_task + distill_weight * loss_distill
            else:
                loss_distill = torch.tensor(0.0)
                loss_train = loss_task
                
            loss_train.backward()
            optimizer.step()

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                outputs_val = self.model(X_val)
                if is_deepar:
                    mu_val, sigma_val, _ = outputs_val
                    # NLL 和 MSE 都做记录
                    loss_val = torch.mean(torch.log(sigma_val) + 0.5 * ((Y_val - mu_val) / sigma_val)**2)
                    val_mse = torch.mean((mu_val - Y_val) ** 2).item()
                else:
                    loss_val = torch.mean((outputs_val - Y_val) ** 2) / torch.mean(Y_val ** 2)
                    val_mse = torch.mean((outputs_val - Y_val) ** 2).item()
                    
            # 记录数据
            l_train = loss_train.item()
            l_val = loss_val.item()
            train_loss_hist.append(l_train)
            val_loss_hist.append(l_val)

            if scheduler is not None:
                scheduler.step()

            # 打印日志
            if epoch % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                model_tag = "DeepAR" if is_deepar else "LSTM"
                
                if t_pred is not None and t_quant is not None:
                    print(f"Epoch {epoch} [{model_tag}] | LR: {current_lr:.5f} | "
                          f"Train Total: {l_train:.6f} (Task: {loss_task.item():.6f}, Distill: {loss_distill.item():.6f}) | "
                          f"Val Loss: {l_val:.6f} | Val MSE: {val_mse:.6f}")
                else:
                    print(f"Epoch {epoch} [{model_tag}] | LR: {current_lr:.5f} | "
                          f"Train Loss: {l_train:.6f} | Val Loss: {l_val:.6f} | Val MSE: {val_mse:.6f}")
            
            # 每100步检查并保存最佳模型
            if epoch % 100 == 0 and l_val <= best_val_loss:
                best_val_loss = l_val
                best_epoch = epoch
                torch.save(self.model, os.path.join(self.save_dir, self.model_name))

        t_elapsed = time.time() - t0
        print(f"Training completed in {t_elapsed:.2f}s. Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

        # 结果记录
        if experiment_info is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            log_file_path = os.path.join(project_root, "forecast_results.txt")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_str = " | ".join([f"{k}: {v}" for k, v in experiment_info.items()])

            arch_type = "DeepAR(Prob)" if is_deepar else "LSTM(Point)"
            log_message = f"[{current_time}] [{arch_type}] {info_str} | Val_Loss: {best_val_loss:.6f} | Train_Time: {t_elapsed:.1f}s\n"
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_message)
                
        return train_loss_hist, val_loss_hist, t_elapsed