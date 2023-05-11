import torch
import os
from torch import nn
a = torch.randn(size=(4,2,3,6))
print(a[...,-1])


net = nn.Conv1d(dilation=)