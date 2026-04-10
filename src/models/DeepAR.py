import torch
import torch.nn as nn

class DeepARModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn. LSTM(input_size, hidden_size, batch_first=True)
        # 2个输出头，均值和标准差
        self.mu_layer = nn.Linear(hidden_size, output_size)
        self.sigma_layer = nn.Linear(hidden_size, output_size)
        self.softplus = nn.Softplus()
        
    def forward(self, x, hidden=None):
        """
        x: Inpuit [batch_size, seq_len, input_size]
        hidden : LSTM 的隐藏状态 (h, c)
        """
        out, hidden = self.lstm(x, hidden)
        mu = self.mu_layer(out)
        sigma = self.softplus(self.sigma_layer(out)) + 1e-6
        
        return mu, sigma, hidden