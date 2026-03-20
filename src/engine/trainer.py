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
            Y_pred = self.model(X_train)
            
            # 基础loss
            loss_task = torch.mean((Y_pred - Y_train) ** 2) / torch.mean(Y_train ** 2)
            # 知识蒸馏loss
            if t_pred is not None and t_quant is not None:
                loss_distill_point = torch.nn.functional.mse_loss(Y_pred, t_pred)
                # 分位数边界惩罚
                lower_bound = t_quant[:, :, :, 0]
                upper_bound = t_quant[:, :, :, -1]
                penalty_lower = torch.relu(lower_bound - Y_pred) ** 2
                penalty_upper = torch.relu(Y_pred - upper_bound) ** 2
                loss_bounds = torch.mean(penalty_lower + penalty_upper)
                
                loss_distill = loss_distill_point + 0.5 * loss_bounds
                
                # 加权总loss
                loss_train = (1.0 - distill_weight) * loss_task + distill_weight * loss_distill
            else:
                loss_train = loss_task
                
            loss_train.backward()
            optimizer.step()

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                Y_pred_val = self.model(X_val)
                loss_val = torch.mean((Y_pred_val - Y_val) ** 2) / torch.mean(Y_val ** 2)

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
                if t_pred is not None and t_quant is not None:
                    print(f"Epoch {epoch} | LR: {current_lr:.5f} | Total Train Loss: {l_train:.6f} "
                          f"(Task: {loss_task.item():.6f}, Distill: {loss_distill.item():.6f}) | Val Loss: {l_val:.6f}")
                else:
                    print(f"Epoch {epoch} | LR: {current_lr:.5f} | Train Loss: {l_train:.6f} | Val Loss: {l_val:.6f}")

            # 每100步检查并保存最佳模型
            if epoch % 100 == 0 and l_val <= best_val_loss:
                best_val_loss = l_val
                best_epoch = epoch
                torch.save(self.model, os.path.join(self.save_dir, self.model_name))

        t_elapsed = time.time() - t0
        print(f"Training completed in {t_elapsed:.2f}s. Best Val Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

        # 结果保存
        if experiment_info is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            log_file_path = os.path.join(project_root, "forecast_results.txt")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_str = " | ".join([f"{k}: {v}" for k, v in experiment_info.items()])
            log_message = f"[{current_time}] {info_str} | Val_MSE: {best_val_loss:.6f} | Train_Time: {t_elapsed:.1f}s\n"
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_message)
                
        return train_loss_hist, val_loss_hist, t_elapsed