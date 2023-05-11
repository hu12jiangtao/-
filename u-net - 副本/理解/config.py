import torch

device = torch.device('cuda')
base_size = 565
crop_size = 480
# root = 'D:\\python\\pytorch作业\\计算机视觉\\u-net\\理解\\DRIVE'
root = 'DRIVE'
train_batch_size = 3
test_batch_size = 1
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
epochs = 200
num_classes = 2