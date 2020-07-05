import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, fan_in, fan_out):
        super(Linear, self).__init__()

        self.linear = nn.Linear(fan_in, fan_out)

        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)
