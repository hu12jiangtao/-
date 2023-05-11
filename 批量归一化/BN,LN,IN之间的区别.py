from torch import nn
import torch
# 首先我们将一个批量的图片[batch,channel,w,h]看成一个高为w*h，长为batch,宽为channel的正方体
# nn.BatchNorm2d为批量归一化，从batch所在的面观察，对 batch*(w,h)的正方形进行归一化（对每个通道进行归一化）,之后在利用gamma，beta去归一化
# 一般用于判别模型(对图片进行分类)，对batch_size比较敏感，需要较大的batch_size才能代表整体数据
a = torch.arange(96,dtype=torch.float32).reshape(3,2,4,4)
batch = nn.BatchNorm2d(2)
y = batch(a)
print(torch.mean(y,[0,2,3]))
print(torch.std(y,[0,2,3]))
print('*'*50)
# 首先我们将一个批量的图片[batch,channel,一个词元的embedding:h]看成一个高为h，长为batch,宽为channel的正方体
# nn.LayerNorm从channel所在的面观察，对 c*h的正方形进行归一化（对一个样本）,之后在利用gamma，beta去归一化，因此nn.LayerNorm的输入维度应该使3维的
# 输入的参数为一个样本中的所有参数
# 一般用于语言模型，此时由于每个batch之间的序列长度不同，因此不在使用BatchNorm2d，而使用每个batch相互独立的nn.LayerNorm
a = torch.arange(24,dtype=torch.float32).reshape(3,2,4)
layer = nn.LayerNorm([2,4])
y = layer(a)
print(torch.mean(y,[1,2]))
print(torch.std(y,[1,2]))
print('*'*50)

# 首先我们将一个批量的图片[batch,channel,w,h]看成一个高为w*h，长为batch,宽为channel的正方体
# 一般用于图片的风格迁移(batch中的单张图片变换成风格转变的其他图片，和输入的batch中的其他图片无关)
# 在风格迁移过程中每个channel的均值和方差会影响最后的结果，但是由于是单对单的关系，因此只是一个样本从batch所在的面观察，计算一个样本中每个通道的均值
# 之后在利用gamma，beta去归一化
a = torch.arange(96,dtype=torch.float32).reshape(3,2,4,4)
Instance = nn.InstanceNorm2d(2)
y = Instance(a)
print(torch.mean(y,[2,3]))
print(torch.std(y,[2,3]))
print('*'*50)