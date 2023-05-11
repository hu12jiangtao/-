# 自己创建的随机网格搜索,其判别方式为K-折交叉验证的方式,随机抽取的参数组合应当是不同的
# 随机网格搜索对于网格搜索来说的优势:
# 随机网格搜索可以接受连续的线性分布，而网格搜索只能接受离散的分布，原因是随机网格搜索是进行随机采样的
# 经过实验的证明:随机网格搜索得到的最右参数的准确率近似于网格搜索(训练集上的准确率和测试集上的准确率),但是所消耗的时间大大的减少了
from torch import nn
import torch
import numpy as np
import random
import K_fold

def set_seed(seed): # 防止网络参数的初始化不一致而产生的影响
    torch.manual_seed(seed) # 用来固定torch中的随机种子,这句话包含了cpu上的随机数
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # 针对所有的gpu上的所有随机数
        torch.backends.cudnn.enabled = False # 禁止cuda使用非确定性算法
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.enabled = True + torch.backends.cudnn.benchmark = True
        # 这两句话在输入的形状不变的情况下，网络结构固定的时候自动选择最为合适的卷积算法可以加快运行的效率


if __name__ == '__main__':
    set_seed(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    param_grid={'dropout':list(np.linspace(0.2,0.9,8)),'lr':list(np.linspace(0.02,1,50)),'num_epoch':list(range(0,11))}
    choice_num = 10
    give_param = [] # 创建了一个子搜索空间
    for i in range(choice_num):
        random_param = {k: random.sample(v, 1)[0] for k, v in param_grid.items()}  # 取出了一个随机的参数组合
        if random_param not in give_param:
            give_param.append(random_param)
        else:
            i -= 1
    result_param,test_acc = K_fold.func(give_param, device)
    print(result_param)
    print(test_acc) # 在测试集上的准确率




