import os

BATCH_SIZE = 4
EPOCH_NUMBER = 200
DATASET = ['CamVid', 12]
crop_size = (352, 480)
path = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\FCN资料合集\\FCN\\Datasets\\CamVid'
class_dict_path = os.path.join(path,'class_dict.csv')
TRAIN_ROOT = os.path.join(path,'train')
TRAIN_LABEL = os.path.join(path,'train_labels')
VAL_ROOT = os.path.join(path,'val')
VAL_LABEL = os.path.join(path,'val_labels')
TEST_ROOT = os.path.join(path,'test')
TEST_LABEL = os.path.join(path,'test_labels')





