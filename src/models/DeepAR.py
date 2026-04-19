import torch
import torch.nn as nn

class DeepARModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        branch_hidden = hidden_size
        # 均值解码器
        self.mu_decoder = nn.Sequential(
            nn.Linear(hidden_size, branch_hidden),
            nn.Softplus(),
            nn.Linear(branch_hidden, output_size)
        )
        # 标准差解码器
        self.sigma_decoder = nn.Sequential(
            nn.Linear(hidden_size, branch_hidden),
            nn.Softplus(),
            nn.Linear(branch_hidden, output_size)
        )
        self.softplus = nn.Softplus()
        
    def forward(self, x, hidden=None):
        """
        x: Inpuit [batch_size, seq_len, input_size]
        hidden : LSTM 的隐藏状态 (h, c)
        """
        out, hidden = self.lstm(x, hidden)
        mu = self.mu_decoder(out)
        sigma = self.softplus(self.sigma_decoder(out)) + 1e-3
        
        return mu, sigma, hidden