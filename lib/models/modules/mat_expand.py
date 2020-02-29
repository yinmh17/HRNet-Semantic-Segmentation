import math

import torch
import torch.nn as nn

from ..functions.mat_expand import mat_expand


class MatExpand(nn.Module):

    def forward(self, input):
        return mat_expand(input)
