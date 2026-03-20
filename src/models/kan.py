import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class ChebyKANLayer(nn.Module):
    def __init__(self, input_dim, output_dim, degree):
        super(ChebyKANLayer, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = Parameter(torch.empty(input_dim, output_dim, degree + 1))
        nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.inputdim, 1)).expand(
            -1, -1, self.degree + 1
        )  # shape = (batch_size, inputdim, self.degree + 1)
        #### Clip the value ####
        eps = 1e-7
        x = torch.clip(x, min=-1 + eps, max=1 - eps)
        # Apply acos
        x = torch.acos(x)
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum(
            "bid,iod->bo", x, self.cheby_coeffs
        )  # shape = (batch_size, outdim)
        y = y.view(-1, self.outdim)
        return y


class ChebyKAN(nn.Module):
    def __init__(self, layers, degree):
        super(ChebyKAN, self).__init__()
        self.cheb_layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.cheb_layers.append(
                ChebyKANLayer(
                    layers[i],
                    layers[i + 1],
                    degree,
                )
            )

    def forward(self, inputs):
        out = inputs
        for linear in self.cheb_layers:
            out = linear(out)
        return out