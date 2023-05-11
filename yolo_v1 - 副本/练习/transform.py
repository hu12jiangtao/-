# 注意在cv2中读取图片的时候其格式由RGB->BGR
import cv2
import numpy as np
import torch
from numpy import random

class BaseTransform(object):
    # 对于验证集的数据主要是进行两个事情:resize和归一化
    # 此时的box中存储的坐标是相对坐标
    def __init__(self,size,mean,std):
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.size = size

    def __call__(self,img, box, label):
        # resize
        img = cv2.resize(img,(self.size, self.size)).astype(np.float32)
        # normal
        img /= 255.
        img = (img - self.mean) / self.std
        return img,box,label

class Augmentation(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        # 此时的mean或者std应当是BGR格式的
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),             # 将int类型转换为float32类型
            ToAbsoluteCoords(),            # 将归一化的相对坐标转换为绝对坐标
            Expand(self.mean),             # 扩充增强(此时输出的每一张图片的尺寸不一定相同)
            RandomSampleCrop(),            # 随机剪裁
            RandomMirror(),                # 随机水平镜像
            ToPercentCoords(),             # 将绝对坐标转换为归一化的相对坐标
            Resize(self.size),             # resize操作（将图片的尺寸给固定下来）
            Normalize(self.mean, self.std) # 图像颜色归一化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)


class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,img, boxes, labels):
        for transform in self.transforms:
            img, boxes, labels = transform(img, boxes, labels)
        return img, boxes, labels


class ConvertFromInts(object):
    def __call__(self, img, box, label):
        return img.astype(np.float32), box, label

class ToAbsoluteCoords(object):
    def __call__(self,img, boxes, label):
        height, width, _ = img.shape
        boxes[:, 0] = boxes[:, 0] * width
        boxes[:, 1] = boxes[:, 1] * height
        boxes[:, 2] = boxes[:, 2] * width
        boxes[:, 3] = boxes[:, 3] * height
        return img, boxes, label

class Expand(object):
    def __init__(self,mean):
        self.mean = mean

    def __call__(self, img, boxes, labels):
        # 此时的目标是对图像进行扩充(50%的概率不进行扩充),扩充的部分用均值进行填充
        if np.random.randint(2):
            return img, boxes, labels
        else:
            height, width, c = img.shape
            ratio = np.random.uniform(1,4)
            # 获取起始点的坐标
            left = np.random.uniform(0, ratio * width - width)
            top = np.random.uniform(0, ratio * height - height)
            # 对image进行处理
            pad_img = np.zeros(shape=(int(ratio * height), int(ratio * width), c),dtype=img.dtype)
            pad_img[:,:,:] = self.mean
            pad_img[int(top): int(top + height), int(left): int(left + width), :] = img
            img = pad_img
            # 对box的绝对坐标进行处理
            boxes_copy = boxes.copy()
            boxes_copy[:, :2] += (int(left), int(top))
            boxes_copy[:, 2:] += (int(left), int(top))
            boxes = boxes_copy
            return img, boxes, labels

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),  # 中间存储着min_iou 和 max_iou
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, image, box, label):
        # 首先确认裁剪的起始点，左上角的坐标，并且确定裁剪后的宽高，获取裁剪后的图形
        # 对box进行筛选和修正(取出中心点不在裁剪后的图形中的box，对中心点在裁剪后的图片的box的坐标进行修改)
        height, width, c = image.shape
        while True:
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, box, label
            min_iou ,max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')


            for _ in range(50): # 选择出一次符合所有要求的裁剪
                cur_image = image
                w = np.random.uniform(0.3 * width, width) # 裁剪后的宽度
                h = np.random.uniform(0.3 * height, height) # 裁剪后的高度
                if h / w < 0.5 or h / w > 2:
                    continue
                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)
                # 裁剪后的坐标
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # 裁剪后的所有的box和裁剪的图形是否匹配(要求所有的box和裁剪后的图形的iou都应该在[min_iou, max_iou]之间)
                overlap = jaccard_numpy(box, rect)
                if overlap.max() > max_iou and overlap.min() < min_iou:
                    continue

                # 裁剪后的图形
                choice_image = cur_image[rect[1]: rect[3], rect[0]: rect[2], :]
                # 对bbox进行筛选(中心点在裁剪后图形外面的需要删除)
                center_bbox = (box[:,:2] + box[:,2:]) / 2 # [n,2]
                mask_left = (center_bbox[:,0] > rect[0]) * (center_bbox[:,1] > rect[1])
                mask_right = (center_bbox[:,0] < rect[2]) * (center_bbox[:,1] < rect[3])
                mask = mask_right * mask_left
                if any(mask) == 0:
                    continue
                cur_box = box[mask]
                label = label[mask]


                # 对保留下来的box的边框坐标进行调整
                cur_box[:,:2] = np.maximum(rect[:2], cur_box[:,:2])
                cur_box[:,:2] -= rect[:2]
                cur_box[:,2:] = np.minimum(rect[2:], cur_box[:,2:])
                cur_box[:,2:] -= rect[:2]
                return choice_image, cur_box, label


def jaccard_numpy(box, rect): # 正确的
    # rect = [4,] , box = [n, 4]
    # 求解公共部分的区域
    x_min = np.maximum(box[:,0], rect[0])
    y_min = np.maximum(box[:,1], rect[1])
    x_max = np.maximum(box[:,2], rect[2])
    y_max = np.maximum(box[:,3], rect[3])
    inter = (x_max - x_min) * (y_max - y_min)
    # 求解两者的交集
    union = (box[:,2] - box[:,0]) * (box[:,3] - box[:,1]) + (rect[2] - rect[0]) * (rect[3] - rect[1]) - inter
    iou = inter / union
    return iou

class RandomMirror(object): # 进行水平的翻转
    def __call__(self,img, box, label):
        if np.random.randint(2):
            height, width ,c = img.shape
            mirror_image = img[:,::-1,:]
            mirror_box = box.copy()
            mirror_box[:,0] = width - box[:,0]
            mirror_box[:,2] = width - box[:,2]
            img = mirror_image
            box = mirror_box
        return img,box,label

class ToPercentCoords(object):
    def __call__(self,img, box, label):
        height, width, c = img.shape
        box[:, 0] /= width
        box[:, 1] /= height
        box[:, 2] /= width
        box[:, 3] /= height
        return img, box, label


class Resize(object):
    def __init__(self,size):
        self.size = size

    def __call__(self, img, box, label):
        # 此时的box为相对坐标
        img = cv2.resize(img, (self.size, self.size))
        return img, box, label

class Normalize(object):
    def __init__(self, mean, std):
        # 此时的通道的顺序为GBR
        self.mean = mean
        self.std = std

    def __call__(self,img, box, label):
        # 只有图片需要进行归一化
        img = img.astype(np.float)
        img /= 255
        img = (img - self.mean) / self.std
        return img, box, label




