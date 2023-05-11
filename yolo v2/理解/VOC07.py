# 此文件对VOC07中的标签和图片进行抽取，并且修正VOC标签的格式和对图片数据进行数据的增强
# 输出的内容为images, labels([xmin,ymin,xmax,ymax,label]) -> 产生的是相对的坐标

from xml.etree import ElementTree as ET

import torch
from torch.utils import data
import numpy as np
import cv2
import os.path as osp

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')  # VOC中含有的类别

class VOCAnnotationTransform(object):
    def __init__(self): # 给每个类别一个标签
        self.class_idx = {name:idx for name,idx in zip(VOC_CLASSES, list(range(len(VOC_CLASSES))))}

    def __call__(self,target,weight,height): # target为一个xml文件经过ET.parse()处理
        # 训练的数据集中应当不包含较难分辨的检测目标, 同时应当由绝对坐标转换为相对的坐标
        res = []
        for obj in target.findall('object'):
            if int(obj.find('difficult').text) == 1: # 说明是较难检测的物体
                continue
            bbox = obj.find('bndbox')
            label = obj.find('name').text
            axis = ['xmin', 'ymin', 'xmax', 'ymax']
            box = []
            for idx,ax in enumerate(axis):
                box.append((int(bbox.find(ax).text) - 1) / weight if idx % 2 == 0 else (int(bbox.find(ax).text) - 1) / height)
            box.append(self.class_idx[label])
            res.append(box)
        return res

class VOCDetection(data.Dataset):
    def __init__(self, root, img_size=None,
                 image_sets=[('2007', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 mosaic=False):
        self.root = root
        self.img_size = img_size
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform # 从xml文件中提取出标签
        self._annopath = osp.join('%s', 'Annotations', '%s.xml') # 用于存储之后表示选中图片的标签信息的路径
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg') # 用于存储之后表示选中图片的路径
        self.ids = []
        for name,dataset in image_sets:
            second_root = osp.join(self.root, 'VOC' + name)
            for i in open(osp.join(second_root,'ImageSets','Main',dataset + '.txt')):
                self.ids.append((second_root, i.strip()))
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def pull_item(self, index):
        choice_idx = self.ids[index]
        xml_path = self._annopath % (choice_idx)
        image_path = self._imgpath % (choice_idx)
        # 读取图片信息
        images = cv2.imdecode(np.fromfile(image_path,dtype=np.int),-1)
        H, W, c = images.shape
        # 读取标签文件
        if self.target_transform is not None:
            target = ET.parse(xml_path)
            target = self.target_transform(target,W,H)
        # 进行数据的增强
        if self.transform is not None:
            target = np.array(target)
            images, bboxes, labels = self.transform(images, target[:,:4], target[:,4])
            # 转换为tensor矩阵的形式
            images = images[:,:,(2,1,0)] # 转换为RGB的格式
            targets = np.concatenate((bboxes, labels.reshape(-1,1)), axis=1)
            return torch.from_numpy(images).permute(2,0,1), targets, H, W

    def pull_image(self, index):
        img_id = self.ids[index]
        return cv2.imdecode(np.fromfile(self._imgpath % img_id, dtype=np.uint8),-1), img_id

if __name__ == '__main__':
    from transforms import Augmentation, BaseTransform
    np.random.seed(2)
    img_size = 640
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    data_root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
    train_transform = Augmentation(img_size, pixel_mean, pixel_std)
    val_transform = BaseTransform(img_size, pixel_mean, pixel_std)

    # dataset
    dataset = VOCDetection(
        root=data_root,
        img_size=img_size,
        image_sets=[('2007', 'trainval')],
        transform=train_transform
        )
    im, gt = dataset[0]

    print(gt)

