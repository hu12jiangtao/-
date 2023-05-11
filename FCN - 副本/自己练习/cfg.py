import os
import torch
file_path = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\FCN资料合集\\FCN\\Datasets\\CamVid'
TRAIN_ROOT = os.path.join(file_path,'train')
TRAIN_LABEL = os.path.join(file_path,'train_labels')
TEST_ROOT = os.path.join(file_path,'test')
TEST_LABEL = os.path.join(file_path,'test_labels')
class_dict_path = os.path.join(file_path,'class_dict.csv')
crop_size = (352,480)
BATCH_SIZE = 4
num_classes = 12
device = torch.device('cuda')
num_epochs = 200
test_batch_size = 2