import torch

device = torch.device('cuda')
root = 'D:\\python\\pytorch作业\\all_data\\cifar100'
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
train_batch_size = 128
test_batch_size = 64