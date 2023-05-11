import torch
import torchvision
from torch import nn


# 不同的学习率训练网络
pre_net = torchvision.models.resnet18(pretrained=True)
net = nn.Sequential(pre_net)
net.add_module('linear',nn.Linear(pre_net.fc.out_features,3))
lr = 1e-3
# net.named_parameters()会显示整一个网络的所有参数的名字和其对应的参数值
pre_param = [param for name,param in net.named_parameters() if name not in ['linear.weight','linear.bias']] # 此时为除了自己添加的那一层的所有的学习参数
trainer = torch.optim.SGD([{'params':pre_param},{'params':net.linear.parameters(),'lr':lr*1000}],lr=lr) # 分成两个不同的学习率
print(trainer.param_groups)


# 不同的学习率训练网络2
net = nn.Sequential(nn.Linear(4,3),nn.Linear(3,1))
forward_net = nn.Sequential(*[child for name,child in net.named_children() if name in ['0']])
backward_net = nn.Sequential(*[child for name,child in net.named_children() if name not in ['0']])
trainer = torch.optim.Adam([{'params':forward_net.parameters(),'lr':0.01},{'params':backward_net.parameters()}],lr=0.1)
print(trainer.param_groups)



'''
# 冻结部分层的权重
net = torchvision.models.resnet18(pretrained=True)
for i in net.layer2.modules():
    print(i)
for name,block in net.named_children():
    if name != 'layer3':
        block.requires_grad_(False)
        for linear in block.modules():
            if isinstance(linear,nn.BatchNorm2d):
                linear.eval()
    else:
        break
'''
