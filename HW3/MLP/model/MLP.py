import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden=64, layers=3):
        super().__init__()
        assert layers in (2, 3)
        mods = []
        mods += [nn.Linear(1, hidden, bias=True), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        if layers == 3:
            mods += [nn.Linear(hidden, hidden, bias=True), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True)]
        mods += [nn.Linear(hidden, 1, bias=True)]
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)
