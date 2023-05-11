# 注意在cv2中读取图片的时候其格式由RGB->BGR
import cv2
import numpy as np
import torch
from numpy import random

class BaseTransform(object):
    def __init__(self,size,mean,std):
        self.size = size
        self.mean = np.array(mean,dtype=np.float32)
        self.std = np.array(std,dtype=np.float32)

    def __call__(self,img, box, label):
        # 此时的box并不需要跟着img进行缩放，因为img中存储的是相对的位置信息的相对坐标
        # 进行大小的重构
        img = cv2.resize(img,(self.size,self.size)).astype(np.float32)
        # 进行归一化处理
        img /= 255.
        img = (img - self.mean) / self.std
        return img,box,label

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,img, box, label):
        for transform in self.transforms:
            img, box, label = transform(img, box, label)
        return img, box, label

class ConvertFromInts(object):
    def __call__(self, img, box, label):
        return img.astype(np.float32), box, label

class ToAbsoluteCoords(object): # 将box中的相对坐标转换为绝对坐标
    def __call__(self, img, box, label):
        height, width, channel = img.shape
        box[:,0] *= width
        box[:,1] *= height
        box[:,2] *= width
        box[:,3] *= height
        return img, box, label


class Expand(object):  # 相当于进行了padding操作(padding的部分填充的是图片的均值)
    def __init__(self,mean):
        self.mean = mean

    def __call__(self,image,box,label):
        # 首先是需要判断是否进行padding
        if np.random.randint(2):
            return image, box, label
        else:
            # 当前图片的尺寸
            height, width, channel = image.shape
            ratio = np.random.uniform(1,4) # 随即扩充至原来图片的整数倍
            # 确定当前图片的起点
            left = np.random.uniform(0, ratio * width - width)
            top = np.random.uniform(0, ratio * height - height)
            # 对图片进行处理
            padding_image = np.zeros((int(height * ratio), int(width * ratio), channel),dtype=image.dtype)
            padding_image[:,:,:] = self.mean
            padding_image[int(top):int(top + height),int(left):int(left + width)] = image
            image = padding_image
            # 对box进行处理
            boxes = box.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))
            return image, boxes, label


def intersect(box_a, box_b): # 用于获取每个标签框和裁剪后的图片之间的公共区域的值
    max_xy = np.minimum(box_a[:,2:], box_b[2:]) # [n, 2] 逐元素进行比较，取最小值
    min_xy = np.maximum(box_a[:,:2], box_b[:2]) # [n, 2]
    inter = np.clip(max_xy - min_xy, a_min=0, a_max=np.inf)
    return inter[:,0] * inter[:,1] # [n,]

def jaccard_numpy(boxes, rect):
    # boxes = [n, 4], rect = [4, ]
    inter = intersect(boxes, rect)
    part1 = (rect[2] - rect[0]) * (rect[3] - rect[1]) # value
    part2 = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1]) # [n,]
    union = part2 + part1 - inter
    return inter / union # [n, ]

class RandomSampleCrop(object):
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            (0.1, None),  # 中间存储着min_iou 和 max_iou
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self,img, box, label):
        height, width, _ = img.shape
        while True:
            # 确认随机裁剪的方式
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return img, box, label
            min_iou ,max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                # 确定裁剪后的尺寸和起始坐标
                current_img = img
                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)
                if h / w < 0.5 or h / w > 2:
                    continue
                left = np.random.uniform(0,width - w)
                top = np.random.uniform(0,height - h)
                # 获得裁剪后的图片的坐标
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])
                # 得到当前裁剪后的图片和原先标签框的iou值
                overlap = jaccard_numpy(box, rect)
                if overlap.min() < min_iou and overlap.max() > max_iou:
                    continue
                # 删减标签框(中心点在裁剪后图片之外的)
                current_img = current_img[rect[1]:rect[3], rect[0]:rect[2], :]
                # 获得中心点
                center = (box[:,2:] + box[:,:2]) / 2 # [n, 2]
                m1 = (center[:,0] > rect[0]) * (center[:,1] > rect[1])  # 标签框的中心点的坐标在修建后图片之外
                m2 = (center[:,0] < rect[2]) * (center[:,1] < rect[3]) # [n,]
                mask = m1 * m2 # [n,],其中元素为True的说明在裁剪后的图片里面
                if not mask.any(): # 所有的标签框都在裁剪后的图片之外
                    continue
                # 修正标签框的标签
                label = label[mask]
                # 修正标签框的坐标
                current_box = box[mask,:]
                current_box[:,:2] = np.maximum(current_box[:,:2], rect[:2])
                current_box[:,:2] -= rect[:2]
                current_box[:,2:] = np.minimum(current_box[:,2:], rect[2:])
                current_box[:,2:] -= rect[:2]
                return current_img, current_box, label

class RandomMirror(object):
    def __call__(self, img, box, label):
        if np.random.randint(2):
            _, width, _ = img.shape
            img = img[:,::-1]
            box_copy = box.copy()
            box_copy[:,0::2] = width - box[:,2::-2] # 2::-2代表从列表索引2向前两个两个推
            box = box_copy
        return img, box, label



class ToPercentCoords(object):
    def __call__(self, img, box, label):
        height, width, _ = img.shape
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



