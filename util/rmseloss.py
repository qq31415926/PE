import torch
import torch.nn as nn

class RootMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
    def forward(self,x,y,x_mask=None):
        if x_mask is not None:
            x = x * x_mask
        return torch.sqrt(self.loss(x,y))