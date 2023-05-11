import torch
import torchvision
import pandas as pd
import cfg
import numpy as np
from torch.utils import data
import os
from PIL import Image

# 首先需要求解出每个类别所对应的颜色，将标签的图片转换为标签的类别
class LabelProcessor:
    def __init__(self,file_path):
        self.colormap = self.read_color_map(file_path) # 得到每个类别对应的RGB的像素值
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(file_path):
        read_data = pd.read_csv(file_path,sep=',',engine='python')
        color_lst = []
        for i in range(len(read_data.index)):
            tmp = read_data.iloc[i]
            color_lst.append([tmp['r'],tmp['g'],tmp['b']])
        return color_lst

    @staticmethod
    def encode_label_pix(color_map): # 用来创建一个对应关系（每个像素值对应的类别序列）
        color = np.zeros(256 ** 3)
        for idx,content in enumerate(color_map):
            color[(content[0] * 256 + content[1]) * 256 + content[2]] = idx
        return color

    def encode_label_img(self, img): # 将图片转换为标签，img=[h,w,3]
        img = np.array(img,dtype='int32')
        idx = (img[:,:,0] * 256 + img[:,:,1]) * 256 + img[:,:,2] # [h,w]
        return np.array(self.cm2lbl[idx],dtype='int64')

class LoadDataset(data.Dataset):
    def __init__(self,file_path,crop_size): # file_path中应当有训练的图片和标签的路径
        super(LoadDataset, self).__init__()
        assert len(file_path) == 2
        self.crop_size = crop_size
        self.image_path = self.read_file(file_path[0])
        self.label_path = self.read_file(file_path[1])

    def read_file(self,root):
        root_path = os.listdir(root)
        root_path = [os.path.join(root,i) for i in root_path]
        root_path = sorted(root_path)
        return root_path

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self, item):
        image_path, label_path = self.image_path[item],self.label_path[item]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        # 只进行了一次的数据增强(center crop的作用是取出图片的中心区域，这个可以不算随机的数据增强，这个是固定的)
        common_trans = [torchvision.transforms.CenterCrop(cfg.crop_size)]
        image_trans = [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])]
        image = torchvision.transforms.Compose(common_trans + image_trans)(image)
        label = torchvision.transforms.Compose(common_trans)(label) # [h,w,3],需要对label进行类变换
        label = label_process.encode_label_img(label) # [h,w]
        label = torch.from_numpy(label)
        return image,label

label_process = LabelProcessor(cfg.class_dict_path)

if __name__ == '__main__':
    train_dataset = LoadDataset(file_path=[cfg.TRAIN_ROOT,cfg.TRAIN_LABEL],crop_size=cfg.crop_size)
    for x,y in train_dataset:
        print(x.shape)
        print(y.shape)
        break