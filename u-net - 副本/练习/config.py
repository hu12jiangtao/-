import torch

in_channel = 3
num_classes = 2
base_channel = 64
device = torch.device('cuda')
data_path = 'DRIVE'
train_batch_size = 3
test_batch_size = 1
lr = 0.01
momentum = 0.9
weight_decay = 1e-4
epochs = 300
