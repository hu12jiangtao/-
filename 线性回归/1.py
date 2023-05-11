import torch
import numpy as np
import random

a = torch.tensor([np.random.uniform(0.8,1.1) for _ in range(10)])
print(a)