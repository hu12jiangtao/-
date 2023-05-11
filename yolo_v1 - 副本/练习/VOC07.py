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
        self.class_dict = {i:j for i, j in zip(VOC_CLASSES,list(range(len(VOC_CLASSES))))}

    def __call__(self,target, weight, height): # target为一个xml文件经过ET.parse()处理
        result = []
        tree = target
        for obj in tree.findall('object'):
            if int(obj.find('difficult').text) == 1: # 说明此时的检测目标较难分配
                continue
            name = obj.find('name').text
            bbox = obj.find('bndbox')
            pst = ['xmin', 'ymin', 'xmax', 'ymax']
            box = []
            for i, ps in enumerate(pst):
                cur_pt = int(bbox.find(ps).text) - 1
                cur_pt = cur_pt / weight if i % 2 == 0 else cur_pt / height
                box.append(cur_pt)
            box.append(self.class_dict[name])
            result.append(box)
        return result # 此时的result中返回的是相对坐标



class VOCDetection(data.Dataset):
    def __init__(self, root, img_size,
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
        for year, name in image_sets:
            root_path = osp.join(self.root, f'VOC{year}')
            for i in open(osp.join(root_path, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((root_path, i.strip()))
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)

        return im, gt

    def pull_item(self, index):
        # 获取对应的标签和图片
        idx = self.ids[index]
        image_path = self._imgpath % idx
        annopath = self._annopath % idx
        image = cv2.imdecode(np.fromfile(image_path,dtype=np.uint8),-1) #
        height, weight, _ = image.shape
        target = ET.parse(annopath)
        # 对标签进行处理
        if self.target_transform is not None:
            target = self.target_transform(target, weight,height) # [n_obj, 5]
        if self.transform is not None:
            target = np.array(target)
            image, bbox, label = self.transform(image, target[:,:4], target[:,4]) # label = [n_obj,]
            # 经过CV的变换后需要将通道数放在最前面
            image = image[:,:,(2,1,0)] # 将BGR通道转换为了RGB通道
            # 将bbox、label进行整合
            target = np.concatenate([bbox,label.reshape(-1,1)],axis=1) # [n_obj, 5]
        return torch.from_numpy(image).permute(2,0,1),target,height,weight


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



