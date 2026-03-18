import torch.nn as nn

class Conv1D(nn.Module):
    def __init__(self, layers, activation, ld):
        super(Conv1D, self).__init__()
        self.linears = nn.ModuleList()
        for i in range(len(layers) - 2):
            self.linears.append(nn.Conv1d(1, 1, 3, padding=1))
        self.linears.append(nn.Linear(1, ld))
        self.activation = activation

    def forward(self, inputs):
        out = inputs
        for i in range(len(self.linears) - 1):
            out = self.linears[i](out)
            out = self.activation(out)
            # out = self.dropout(out)
        return self.linears[-1](out)