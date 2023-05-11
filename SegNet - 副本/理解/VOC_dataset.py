# 在dataset中需要确定两个部分的内容:1.所有的训练图片的类别对应的像素点出现的频率 2.进行数据增强的输出的对应的mask和image图片
import numpy as np
from torch.utils import data
import os
from PIL import Image
from torchvision.transforms import functional as F
import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt

VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
NUM_CLASSES = len(VOC_CLASSES) + 1

class PascalVOCDataset(data.Dataset):
    def __init__(self, list_file, img_dir, mask_dir):
        super(PascalVOCDataset, self).__init__()
        # list_file 存放输入的图片的名称
        self.image_name_lst = open(list_file,'r').read().split('\n')[:-1]
        self.image_end = '.jpg'
        self.mask_end = '.png'
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.count = self.__compute_class_probability()

    def __compute_class_probability(self): # 用来统计所有输入图片中的各个类别的像素点的个数
        count = {i:0 for i in range(NUM_CLASSES)}
        for name in self.image_name_lst:
            mask = os.path.join(self.mask_dir, name + self.mask_end)
            mask = Image.open(mask)
            mask = F.resize(mask,(224,224),interpolation=T.InterpolationMode.NEAREST)
            mask = np.array(mask).reshape(224 * 224) # [224*224,]
            mask[mask == 255] = len(VOC_CLASSES) # 对应边界标签
            for i in range(NUM_CLASSES):
                count[i] += np.sum(mask == i)
        return count

    def get_class_probability(self): # 得到每类像素点的占比
        value = np.array(list(self.count.values()))
        return torch.as_tensor(value / np.sum(value))

    def __len__(self):
        return len(self.image_name_lst)

    def __getitem__(self, item):
        name = self.image_name_lst[item]
        image_path = os.path.join(self.img_dir, name + self.image_end)
        mask_path = os.path.join(self.mask_dir, name + self.mask_end)
        image = self.load_image(image_path)
        mask = self.load_mask(mask_path)
        data_1 = {
            'image':torch.FloatTensor(image), # [c,h,w]
            'mask':torch.LongTensor(mask) # [h,w]
        }
        return data_1

    def load_image(self,path): # 需要进行resize和归一化
        image = Image.open(path)
        # image = F.resize(image,(224,300))即是生成h=224,w=300的PIL图片, image.resize(image,(224,300))即是生成w=224,h=300的PIL图片
        # 但是利用这两种方法生成PIL图片都是只改变大小，不会进行翻转的
        image = F.resize(image, (224,224)) # [h=224,w=224,c=3]
        image = np.transpose(image,(2,0,1)) # [c,h,w]
        image = np.array(image,dtype=np.float32) / 255.0
        return image # [c,h,w]

    def load_mask(self,path):
        mask = Image.open(path)
        mask = F.resize(mask,(224,224),interpolation=T.InterpolationMode.NEAREST) # [h,w]
        mask = np.array(mask)
        mask[mask == 255] = len(VOC_CLASSES)
        return mask

if __name__ == '__main__':
    list_file = 'D:\\python\\pytorch作业\\计算机视觉\\SegNet\\VOCdevkit\\VOC2007\\ImageSets\\Segmentation\\train.txt'
    img_dir = 'D:\\python\\pytorch作业\\计算机视觉\\SegNet\\VOCdevkit\\VOC2007\\JPEGImages'
    mask_dir = 'D:\\python\\pytorch作业\\计算机视觉\\SegNet\\VOCdevkit\\VOC2007\\SegmentationClass'
    objects_dataset = PascalVOCDataset(list_file,img_dir,mask_dir)
    image = objects_dataset[0]['image'].numpy().transpose(1,2,0)
    plt.imshow(image)
    plt.show()
    # mask中应当是每一个像素都根据其类别映射至其对应的颜色
    # 此时飞机的类别由于标签是1，和背景非常相似，因此图片中看不出变化
    mask = objects_dataset[0]['mask']
    plt.imshow(mask)
    plt.show()
