import torch
from torch import nn

class MLP(nn.Module):
    """Multilayer perceptron

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_unit, bias=True),
            nn.LayerNorm(hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit, bias=True)
        )

    def forward(self, x):
        x = self.mlp(x)
        return x