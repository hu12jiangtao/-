# 根据VOC_dataset同样需要求解每类在所有图片中所占的权重
# 此时是针对于ComVid的数据集
from torch.utils import data
import os
import pandas as pd
import ComVid_config
import numpy as np
from PIL import Image
from torchvision import transforms as T
import torch

class LabelProcessor:
    def __init__(self,file_path):
        self.colormap = self.read_csv_file(file_path)
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_csv_file(file_path):
        color_lst = []
        file = pd.read_csv(file_path,sep=',',engine='python')
        for i in range(len(file.index)):
            tmp = file.iloc[i]
            color_lst.append([tmp['r'],tmp['g'],tmp['b']])
        return color_lst

    @staticmethod
    def encode_label_pix(color_map): # 对于每个像素点进行赋值
        color_dict = np.zeros(256**3)
        for idx,color_content in enumerate(color_map):
            color_dict[(color_content[0] * 256 + color_content[1]) * 256 + color_content[2]] = idx
        return color_dict

    def encode_label_img(self, img): # 将验证图片转换为标签(此时输入的应当是PIL图片的格式)
        img = np.array(img,dtype=np.int32) # [h,w,3]
        idx = img[:,:,0] * 256**2 + img[:,:,1] * 256 + img[:,:,2]
        return np.array(self.cm2lbl[idx],dtype=np.int64)

# 此时需要将输入模型的图片和标签的图片一对一对的输出(成双成对的出现)，同时需要数据增强(此时的标签矩阵同样进行Normalize和ToTensor操作)
class LoadDataset(data.Dataset):
    def __init__(self, filter_size, crop_size): # filter_size为一个列表，其中分别是训练图片和标签图片所存放的文件夹
        super(LoadDataset, self).__init__()
        image_path = filter_size[0]
        label_path = filter_size[1]
        self.image_path_lst = self.read_path(image_path)
        self.label_path_lst = self.read_path(label_path)
        self.crop_size = crop_size

    def read_path(self,path):
        name = os.listdir(path)
        path = [os.path.join(path,i) for i in name]
        path.sort()
        return path

    def __len__(self):
        return len(self.image_path_lst)

    def __getitem__(self, item):
        image_path = self.image_path_lst[item]
        label_path = self.label_path_lst[item]
        image = Image.open(image_path).convert('RGB') # [h,w,3]的PIL图
        label = Image.open(label_path).convert('RGB') # [h,w,3]的PIL图
        # 进行数据的增强
        transform1 = [T.CenterCrop(self.crop_size)]
        transform2 = [T.ToTensor(),T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        image = T.Compose(transform1 + transform2)(image) # [3,224,224]
        label = T.Compose(transform1)(label) # [224,224,3]的PIL图
        label = a.encode_label_img(label) # 转换为了标签
        label = torch.as_tensor(label)
        return image,label

def compute_class_probability(labels,num_classes): # labels应为一个列表，其中的元素为一个形状为[h,w]的标签矩阵
    count = {i:0 for i in range(num_classes)}
    for label in labels:
        label = label.reshape(-1)
        for i in count.keys():
            count[i] += torch.sum(label == i)
    each_num = np.array(list(count.values()))
    all_sum = torch.as_tensor(each_num / np.sum(each_num))
    return all_sum


a = LabelProcessor(ComVid_config.class_dict_path)

if __name__ == '__main__':
    root = 'D:\\python\\pytorch作业\\计算机视觉\\FCN\\FCN资料合集\\FCN(模板)\\Datasets\\CamVid\\train_labels\\0001TP_006690_L.png'
    label_image = Image.open(root)
    label = a.encode_label_img(label_image)
    for i in range(12):
        print(i in label)

