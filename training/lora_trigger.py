import torch
import torch.nn as nn
import math
import numpy as np

class LoraTrigger(nn.Module):
    def __init__(self, d, m, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        self.register_buffer('shift', torch.zeros(d))
        self.register_buffer('gain', torch.ones(d))
        self.register_buffer('mask', torch.ones(d))
        self.mask.fill_(0)
        for i in range(d):
            for j in range(i+1, d):
                if np.random.rand() < 0.01:
                    self.mask[i,j] = 1
                    self.mask[j,i] = 1

    def forward(self, x):
        return self.alpha * (x