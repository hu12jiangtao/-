import torch
import os
device = torch.device('cuda')
root = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\FCN资料合集\\FCN(模板)\\Datasets\\CamVid'
# root = 'CamVid'
class_dict_path = os.path.join(root,'class_dict.csv')
in_channel = 3
out_channel = 12
num_classes = 12
crop_size=(224,224)
train_image_path = os.path.join(root,'train')
train_label_path = os.path.join(root,'train_labels')
test_image_path = os.path.join(root,'test')
test_label_path = os.path.join(root,'test_labels')
num_epochs = 200
