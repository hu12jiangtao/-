from torch import nn
import torch
# from torch.nn import functional as F
from PIL import Image
import numpy as np
from torch.utils import data
import torchvision
from torchvision.transforms import functional
import pandas as pd
import ComVid_config


# x = torch.arange(64,dtype=torch.float32).reshape(1,1,8,8)
# b, index = F.max_pool2d(x, kernel_size=2,stride=2,return_indices=True) # b为返回的最大值,c代表返回的最大值的序列
#
# x = F.max_unpool2d(b,index,kernel_size=2,stride=2,output_size=x.shape)
# print(x)

# path = 'D:\\python\\pytorch作业\\计算机视觉\\SegNet\\VOCdevkit\\VOC2007\\JPEGImages\\2007_000392.jpg'
# image = Image.open(path)
# # image = functional.resize(image, (224,300),interpolation=torchvision.transforms.InterpolationMode.NEAREST)
# image = np.transpose(image.resize((224,300)),(0,1,2))
# image = Image.fromarray(image)
# image.show()
# image = np.array(image)
# print(image.shape)


# a = np.array([[1,2,3,9],[4,5,6,8]])
# print(a.shape)
# image_PIL = Image.fromarray(a)
# a = image_PIL.resize((2,3))
# print(np.array(a).shape) # [3,2]


# torch.manual_seed(1)
# loss = nn.CrossEntropyLoss()
# x = torch.randn(size=(1,3,2,3)) # [batch,c,h,w]
# y = torch.tensor([[[1,0],[2,1],[1,1]]]).permute(0,2,1) # [batch,w,h]
# print(loss(x,y))

root = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\FCN资料合集\\FCN(模板)\\Datasets\\CamVid\\train_labels\\0001TP_006690_L.png'
a = np.array(Image.open(root))
print(a.shape)

a = [3,2,1]
a.sort()
print(a)

