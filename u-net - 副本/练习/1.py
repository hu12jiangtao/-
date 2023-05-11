import random
import torch
import numpy as np
import os

root = 'DRIVE'
train = True
root_path = os.path.join(root,'training' if train is True else 'test')
mask_path = os.path.join(root_path,'mask')
roi_mask = [os.path.join(mask_path,i) for i in os.listdir(mask_path)]
input_path = os.path.join(root_path,'images')
img_list = [os.path.join(input_path,i) for i in os.listdir(input_path)]
label_path = os.path.join(root_path,'1st_manual')
manual = [os.path.join(label_path,i) for i in os.listdir(label_path)]
print(img_list[12])
print(roi_mask[12])
print(manual[12])

flag = 'training' if train else 'test'
root_path = os.path.join(root, flag)
img_lst = [i for i in os.listdir(os.path.join(root_path, 'images')) if i.endswith('.tif')]
img_list = [os.path.join(root_path, 'images', i) for i in img_lst]  # 输入模型的图片的路径列表
manual = [os.path.join(root_path, '1st_manual', i.split('_')[0] + '_manual1.gif') for i in img_lst]
roi_mask = [os.path.join(root_path, 'mask', i.split('_')[0] + f'_{flag}_mask.gif') for i in img_lst]

print(img_list[12])
print(roi_mask[12])
print(manual[12])