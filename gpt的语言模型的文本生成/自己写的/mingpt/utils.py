import torch
import numpy as np
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class CfgNode:
    def __init__(self,**kwargs): # 不知道数量的 a = b 形式的键值对
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        parts = []
        for k,v in self.__dict__.items():
            if isinstance(v,CfgNode):
                parts.append(f'{k}:\n')
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append(f'{k}:{v}\n')
        parts = [' ' * (indent * 4) + p for p in parts]
        return ''.join(parts)

    def merge_from_dict(self, d): # 输入的应当是字典
        self.__dict__.update(d)

