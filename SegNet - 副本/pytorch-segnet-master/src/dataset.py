"""Pascal VOC Dataset Segmentation Dataloader"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image


VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')
NUM_CLASSES = len(VOC_CLASSES) + 1



class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1] # 给出参与训练的图片对的编号(对应的mask和image只有后缀是不同的)
        self.transform = transform
        # 在mask图片中，像素为255的代表着类别之间的分割线，且根据VOC_CLASSES给mask图片的每一个像素赋予标签
        self.img_extension = ".jpg" # 显示输入模型的图片的后缀格式
        self.mask_extension = ".png" # 显示图片类别的mask图片的后缀格式

        self.image_root_dir = img_dir # 输入模型的图片所保存的文件夹的路径
        self.mask_root_dir = mask_dir # 显示图片类别的mask图片的文件夹路径

        # self.counts = self.__compute_class_probability() # 得到训练图片中每个类别的像素点的占比

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index] # 给出此次载入图片的名称
        image_path = os.path.join(self.image_root_dir, name + self.img_extension) # 此次图片的image的路径
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension) # 此次标签的路径

        image = self.load_image(path=image_path) # 转换为numpy类型矩阵，通道数在前，形状变为(224,224)
        gt_mask = self.load_mask(path=mask_path) # 形状变为(224,224)

        data = {
                    'image': torch.FloatTensor(image),
                    'mask' : torch.LongTensor(gt_mask)
                    }

        return data

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES)) # 当i = NUM_CLASSES - 1的时候，代表着类别之间的分界线
        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension) # 获取一张mask图片的路径

            raw_image = Image.open(mask_path).resize((224, 224)) # 将这张mask图片数据增强为相应的尺寸，mask为黑白图，只有[h,w]两个维度
            imx_t = np.array(raw_image).reshape(224*224) # 转换为行向量
            imx_t[imx_t==255] = len(VOC_CLASSES) # 将原色的白色的像素点转换为len(VOC_CLASSES)这个类标签，这个类标签对应VOC_CLASSES类别以外的类别

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)  # 得到所有训练图片中每个类别的像素点的个数

        return counts

    def get_class_probability(self): # 得到每个类别的像素点的占比
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)

    def load_image(self, path=None):
        raw_image = Image.open(path) # 转换为PIL类型
        raw_image = np.transpose(raw_image.resize((224, 224)), (2,1,0)) # 此时的输出应为[w,h,c]->[c,h,w]
        # 转换为通道数放置在前面的格式 # 此时经过Image.resize中的(224, 224)第一个参数对应矩阵的宽，第二个参数对应矩阵的高
        imx_t = np.array(raw_image, dtype=np.float32)/255.0 # 归一化

        return imx_t

    def load_mask(self, path=None):
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224)) # 此时的线性分类中出现了因为线性差值导致的不存在于VOC_CLASSES的像素值
        imx_t = np.array(raw_image) # 此时的形状为[w,h]
        # border
        imx_t[imx_t==255] = len(VOC_CLASSES) # 给每个类别边界部分类别标签

        return imx_t


if __name__ == "__main__":
    root = 'D:\\python\\pytorch作业\\计算机视觉\\SegNet'
    data_root = os.path.join(root, "VOCdevkit", "VOC2007")
    list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt") # 每行记录着训练使用的图片编号
    img_dir = os.path.join(data_root, "JPEGImages") # 输入模型未经数据增强的原始图片
    mask_dir = os.path.join(data_root, "SegmentationClass") # 所关注的类别轮廓的黑白图，类别的轮廓用白色表示


    objects_dataset = PascalVOCDataset(list_file=list_file_path,
                                       img_dir=img_dir,
                                       mask_dir=mask_dir)
    # print(objects_dataset.get_class_probability())  # 此时可用于确认的每个类别所造成的权重


    sample = objects_dataset[0]
    image, mask = sample['image'], sample['mask']
    image.transpose_(0, 2)  # 此时输出的图片为[224,224,3]
    print(image.shape)
    fig = plt.figure()

    a = fig.add_subplot(1,2,1)
    plt.imshow(image)

    a = fig.add_subplot(1,2,2)
    plt.imshow(mask)

    plt.show()

