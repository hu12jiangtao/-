import torchvision.transforms as transforms
from torch import nn
import torch
from PIL import Image
import numpy as np

# path = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\FCN资料合集\\FCN\\Datasets\\CamVid\\test_labels\\0001TP_008550_L.png'
# image = Image.open(path)
# print(np.array(image).shape)


# kernel_size = 4
# a = np.ogrid[:kernel_size,:kernel_size+5]
# print(a)


# x = np.array([0, 1, 1, 3, 2, 1, 4])
# print(np.bincount(x))

# num_classes = 4
# label_true = np.array([0,2,3,1,0,2])
# label_pred = np.array([1,3,2,1,0,2])
# mask = (label_true >= 0) & (label_true < num_classes)
# print(mask)
# print(label_pred[mask])
# print(num_classes * label_true[mask].astype(int))
# print(num_classes * label_true[mask].astype(int) + label_pred[mask])
# hist = np.bincount(
#             num_classes * label_true[mask].astype(int) + label_pred[mask], minlength=num_classes ** 2
#         ).reshape(num_classes, num_classes)
# print(hist)

a = torch.tensor([[1,0],[2,3]])
b = torch.tensor([[0,0,0],[0,0,1],[0,1,0],[0,1,1]])
b.type(torch.float32)
print(b)