import os
import time
import torch
import datetime

# 辅助函数: 计算模型所有参数梯度范数的
def compute_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


class Trainer:
    def __init__(self, model, device, save_dir, model_name="best_model.pth"):
        self.model = model.to(device)
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        os.makedirs(self.save_dir, exist_ok=True)

    def train(self, X_train, Y_train, X_val, Y_val, optimizer, scheduler, num_epochs, 
              log_interval=1000, experiment_info=None,
              t_pred=None, t_quant=None, distill_weight=0.5, warmup_steps=50,
              patience=5000):
        
        train_loss_hist = []
        val_loss_hist = []
        best_val_mse = float('inf')
        best_epoch = 0
        patience_counter = 0

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
                loss_task = torch.mean((Y_pred - Y_train) ** 2) / (torch.mean(Y_train ** 2) + 1e-8)
                
            # 蒸馏 loss 计算
            if t_pred is not None and t_quant is not None:
                if is_deepar:
                    lower_bound = t_quant[:, :, :, 0]
                    upper_bound = t_quant[:, :, :, -1]
                    t_sigma_target = torch.clamp((upper_bound - lower_bound) / 3.92, min=1e-4)
                    loss_distill_mu = torch.mean((mu - t_pred) ** 2) / (torch.mean(t_pred ** 2) + 1e-8)
                    loss_distill_sigma = torch.nn.functional.mse_loss(sigma, t_sigma_target)
                    loss_distill_raw = loss_distill_mu + loss_distill_sigma
                else:
                    # LSTM 的蒸馏 loss: 拟合点预测，并加上分位数越界惩罚
                    loss_distill_point = torch.mean((Y_pred - t_pred) ** 2) / (torch.mean(t_pred ** 2) + 1e-8)
                    lower_bound = t_quant[:, :, :, 0]
                    upper_bound = t_quant[:, :, :, -1]
                    penalty_lower = torch.relu(lower_bound - Y_pred) ** 2
                    penalty_upper = torch.relu(Y_pred - upper_bound) ** 2
                    loss_bounds = torch.mean(penalty_lower + penalty_upper) 
                    loss_distill_raw = loss_distill_point + loss_bounds
                
                # Compute Grad
                optimizer.zero_grad()
                loss_task.backward(retain_graph=True)
                task_grads = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
                optimizer.zero_grad()
                loss_distill_raw.backward()
                distill_grads = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p) for p in self.model.parameters()]
                optimizer.zero_grad()
                
                with torch.no_grad():
                    # Adaptive Gradient Scaling
                    g_task_flat = torch.cat([g.flatten() for g in task_grads])
                    g_distill_flat = torch.cat([g.flatten() for g in distill_grads])
                    grad_norm_task = torch.norm(g_task_flat, p=2)
                    grad_norm_distill = torch.norm(g_distill_flat, p=2)

                    dynamic_scale = grad_norm_task / (grad_norm_distill + 1e-8)
                    dynamic_scale = torch.clamp(dynamic_scale, min=0.01, max=1000.0)
                    
                    # PCGrad
                    g_task_scaled_flat = (1.0 - distill_weight) * g_task_flat
                    g_distill_scaled_flat = distill_weight * dynamic_scale * g_distill_flat
                    dot_product = torch.dot(g_task_scaled_flat, g_distill_scaled_flat)
                    if dot_product < 0:
                        task_norm_sq = torch.dot(g_task_scaled_flat, g_task_scaled_flat) + 1e-8
                        projection_scalar = dot_product / task_norm_sq
                        
                        for p, g_t, g_d in zip(self.model.parameters(), task_grads, distill_grads):
                            g_t_scaled = (1.0 - distill_weight) * g_t
                            g_d_scaled = distill_weight * dynamic_scale * g_d
                            g_d_projected = g_d_scaled - projection_scalar * g_t_scaled
                            p.grad = g_t_scaled + g_d_projected
                    else:
                        for p, g_t, g_d in zip(self.model.parameters(), task_grads, distill_grads):
                            p.grad = (1.0 - distill_weight) * g_t + distill_weight * dynamic_scale * g_d
                            
                # 更新网络参数并记录
                optimizer.step()
                loss_distill = loss_distill_raw * dynamic_scale
                loss_train = (1.0 - distill_weight) * loss_task + distill_weight * loss_distill
                
            else:
                loss_distill = torch.tensor(0.0)
                loss_train = loss_task
                optimizer.zero_grad()
                loss_train.backward()
                optimizer.step()

            # 验证阶段
            self.model.eval()
            with torch.no_grad():
                outputs_val = self.model(X_val)
                eval_start = warmup_steps
                if is_deepar:
                    mu_val, sigma_val, _ = outputs_val
                    mu_val_eval = mu_val[:, eval_start:, :]
                    sigma_val_eval = sigma_val[:, eval_start:, :]
                    Y_val_eval = Y_val[:, eval_start:, :]
                    # NLL 和 MSE 都做记录
                    loss_val = torch.mean(torch.log(sigma_val_eval) + 0.5 * ((Y_val_eval - mu_val_eval) / sigma_val_eval)**2)
                    val_mse = torch.mean((mu_val_eval - Y_val_eval) ** 2).item()
                else:
                    outputs_eval = outputs_val[:, eval_start:, :]
                    Y_val_eval = Y_val[:, eval_start:, :]
                    loss_val = torch.mean((outputs_eval - Y_val_eval) ** 2) / (torch.mean(Y_val_eval ** 2) + 1e-8)
                    val_mse = torch.mean((outputs_eval - Y_val_eval) ** 2).item()
                    
            # 记录数据
            l_train = loss_train.item()
            l_val = loss_val.item()
            train_loss_hist.append(l_train)
            val_loss_hist.append(val_mse)

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
            if epoch % 100 == 0:
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_epoch = epoch
                    torch.save(self.model, os.path.join(self.save_dir, self.model_name))
                    patience_counter = 0
                else:
                    patience_counter += 100
                # 早停
                if patience_counter >= patience:
                    print(f"\n>>> Early Stopping Triggered! No improvement for {patience} epochs. Stopping at Epoch {epoch} <<<")
                    break

        t_elapsed = time.time() - t0
        print(f"Training completed in {t_elapsed:.2f}s. Best Val MSE: {best_val_mse:.6f} at Epoch {best_epoch}")

        # 结果记录
        if experiment_info is not None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            log_file_path = os.path.join(project_root, "forecast_results.txt")
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info_str = " | ".join([f"{k}: {v}" for k, v in experiment_info.items()])

            arch_type = "DeepAR" if is_deepar else "LSTM"
            log_message = f"[{current_time}] [{arch_type}] {info_str} | Val_MSE: {best_val_mse:.6f} | Train_Time: {t_elapsed:.1f}s\n"
            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(log_message)
                
        return train_loss_hist, val_loss_hist, t_elapsed