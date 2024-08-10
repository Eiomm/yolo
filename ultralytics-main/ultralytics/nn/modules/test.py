import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_

from .conv import Conv
from .utils import _get_clones, inverse_sigmoid, multi_scale_deformable_attn_pytorch


class TransformerLayer(nn.Module):

    def __init__(self, c, num_heads):

        super.__init__()
        self.q = nn.Linear(c, c ,bias = False)
        self.k = nn.Linear(c, c ,bias = False)
        self.v = nn.Linear(c, c ,bias = False)
        self.MA = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias= False)
        self.fc2 = nn.Linear(c, c, bias= False)

    def forward(self, x):
        x = self.MA(self.q(x), self.k(x), self.v(x))[0] + x
        return self.fc2(self.fc1(x)) + x
    

