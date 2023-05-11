import sys
import torch
import os
from torch.utils import data
import os.path as osp
import xml.etree.ElementTree as ET
import cv2
import numpy as np

VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')  # VOC中含有的类别

class VOCAnnotationTransform(object):
    def __init__(self):
        # 用于从给定的标签文件中提取出当前图片中含有的类别以及在图片中标签框的相对坐标
        self.class_to_end = {i:j for i, j in zip(VOC_CLASSES,list(range(len(VOC_CLASSES))))}

    def __call__(self,target,weight,height): # target为一个xml文件经过ET.parse()处理
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1 # 若是为检测困难的物体则不将其作为训练集(将一张图片中检测难度较大的物体给舍弃掉)
            if difficult:
                continue
            name = obj.find('name').text.lower().strip() # 获得当前这个对象的名字
            bbox = obj.find('bndbox') # 获得当前这张图象中obj这个对象的标签框
            pst = ['xmin','ymin','xmax','ymax'] # x对应weight，y对应height
            box = [] # 用于存储宽高的坐标的
            for i, pt in enumerate(pst):
                cur_pt = int(bbox.find(pt).text) - 1 # 获取当前的一个坐标(左上角的宽高或者右下角的宽高值的其中一个)
                cur_pt = cur_pt / weight if i % 2 == 0 else cur_pt / height # 转换为相对的坐标
                box.append(cur_pt)
            label_idx = self.class_to_end[name] # 当前图片的标签
            box.append(label_idx)
            res.append(box)
        return res # 一张图片中的所有对象的相对坐标和标签


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
        self.mosaic = mosaic
        self.ids = []
        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def pull_item(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot() # 此时得到的是一个xml文件
        img = cv2.imdecode(np.fromfile(self._imgpath % img_id, dtype=np.uint8),-1) # 此时路径中可以存在中文名字
        height, weight, channel = img.shape
        if self.target_transform is not None: # 对标签进行处理
            target = self.target_transform(target, weight, height) #
        if self.transform is not None:
            # 对图片进行处理(此时需要标签框随着图像一起进行改动(例如图片的旋转，但是尺寸的扩展、颜色的改变并不需要对box进行操作))
            target = np.array(target) # [num_obj, 5]
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # 将图片转换为RGB的格式
            img = img[:,:,(2,1,0)] # [h,w,c]
            # 将经过处理的图片方框坐标信息和标签信息结合在一起
            target = np.concatenate([boxes, labels.reshape(-1,1)], axis=1)
        return torch.from_numpy(img).permute(2,0,1),target, height, weight


if __name__ == "__main__":
    from transform import Augmentation, BaseTransform

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