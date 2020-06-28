import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim, hops):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hops = hops
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, self.hops)
        )

    def forward(self, encoder_outputs, mask=None): # torch.Size([32, 66, 400])

        energy = self.projection(encoder_outputs)       # torch.Size([32, 66, 5])

        if mask is not None:
            mask = -9999 * (1 - mask)
            mask = mask.unsqueeze(-1)
            energy = energy + mask

        weights = F.softmax(energy, dim=1)  # torch.Size([32, 66, 5])
        weights = weights.transpose(1,2)                # torch.Size([32, 5, 66])
        outputs = weights @ encoder_outputs             # torch.Size([32, 5, 400])

        outputs = torch.sum(outputs, 1) / self.hops     # torch.Size([32, 400])
       
        return outputs, weights