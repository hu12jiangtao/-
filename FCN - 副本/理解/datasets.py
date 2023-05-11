import pandas as pd
from torchvision import transforms
from torch.utils import data
import os
from PIL import Image
import numpy as np
import torch
import cfg

class LabelProcessor:
    def __init__(self,file_path):
        self.colormap = self.read_color_map(file_path) # 得到每个类别对应的RGB的像素值
        self.cm2lbl = self.encode_label_pix(self.colormap)

    @staticmethod
    def read_color_map(path):
        pd_label_color = pd.read_csv(path,sep=',',engine='python') # 导入了表格
        lst = []
        for i in range(len(pd_label_color.index)):
            color = pd_label_color.iloc[i]
            lst.append([color['r'],color['g'],color['b']])
        return lst

    @staticmethod
    def encode_label_pix(colormap):
        color_dict = np.zeros(256 ** 3)
        for idx,color_num in enumerate(colormap):
            color_dict[(color_num[0] * 256 + color_num[1]) * 256 + color_num[2]] = idx
        return color_dict

    def encode_label_img(self, img): # 标签图片转换为平面的标签
        img = np.array(img,dtype='int32')
        idx = (img[:,:,0] * 256 + img[:,:,1]) * 256 + img[:,:,2] # [w,h]
        return np.array(self.cm2lbl[idx], dtype='int64')

class LoadDataset(data.Dataset):
    def __init__(self,file_path,crop_size): # file_path为一个列表，其中存放输入图片的文件路径和输出图片文件的路径
        super(LoadDataset, self).__init__()
        assert len(file_path) == 2
        self.crop_size = crop_size
        self.img_path = file_path[0]
        self.label_path = file_path[1]
        self.images = self.read_file(self.img_path)
        self.labels = self.read_file(self.label_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image_path = self.images[item]
        label_path = self.labels[item]
        image = Image.open(image_path).convert('RGB')
        label = Image.open(label_path).convert('RGB')
        transform_lst1 = [transforms.CenterCrop(self.crop_size)]
        transform_lst2 = [transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        image = transforms.Compose(transform_lst1 + transform_lst2)(image) # [3,h,w]
        label = transforms.Compose(transform_lst1)(label) # 一张PIL类别的图片
        # 将标签图片转换为标签
        label = np.array(label)
        label = Image.fromarray(label) # 这两句话是为了确认标签图片就是PIL格式的图片
        out_label = label_processor.encode_label_img(label)
        out_label = torch.from_numpy(out_label)
        sample = {'img': image, 'label': out_label}
        return sample

    def read_file(self,path):
        img_path = os.listdir(path)
        img_path = [os.path.join(path, i) for i in img_path]
        img_path.sort()
        return img_path

# 导入数据集(训练数据集，验证的数据集)
label_processor = LabelProcessor(cfg.class_dict_path)

if __name__ == '__main__':
    label_processor = LabelProcessor(cfg.class_dict_path)
    Load_train = LoadDataset([cfg.TRAIN_ROOT, cfg.TRAIN_LABEL], cfg.crop_size)
    train_data = data.DataLoader(Load_train, batch_size=1, shuffle=False)
    for sample in train_data:
        img_data = sample['img']
        img_label = sample['label']  # 标签出问题
        print(img_data)
        print(img_label)
        break





