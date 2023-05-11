import random

import cv2
import numpy as np

class BaseTransform(object): # 只进行缩放和归一化
    def __init__(self, size, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)): # 此时的mean、std的通道为BGR
        self.size = size
        self.mean = mean
        self.std = std

    def __call__(self,images, bboxes=None, labels=None):
        images = images.astype(np.float32)
        images = cv2.resize(images,(self.size,self.size))
        images /= 255.
        images = (images - self.mean) / self.std
        return images, bboxes, labels

class Compose(object):
    def __init__(self,transforms):
        self.transforms = transforms

    def __call__(self,images, bboxes, labels):
        for transform in self.transforms:
            images, bboxes, labels = transform(images, bboxes, labels)
        return images, bboxes, labels

class ConvertFromInts(object):
    def __call__(self,images, bboxes, labels):
        images = images.astype(np.float32)
        return images, bboxes, labels

class ToAbsoluteCoords(object):
    def __call__(self,images, bboxes, labels):
        H, W, c = images.shape
        bboxes[:,0] *= W
        bboxes[:,1] *= H
        bboxes[:,2] *= W
        bboxes[:,3] *= H
        return images, bboxes, labels

class Expand(object):
    def __init__(self,mean):
        self.mean = mean

    def __call__(self,images,bboxes,labels):
        if np.random.randint(2):
            return images, bboxes, labels
        else:
            # 随机指定一个ratio，宽高扩充至该倍率
            H, W, c = images.shape
            ratio = np.random.uniform(1,4)
            pad_image = np.zeros(shape=[int(H * ratio), int(W * ratio), c])
            pad_image[:,:,:] = self.mean
            # 指定扩充的起点
            left = np.random.uniform(0, ratio * W - W)
            top = np.random.uniform(0, ratio * H - H)
            # 对图片进行扩充
            pad_image[int(top):int(top + H), int(left): int(left + W),:] = images
            # 对标签进行扩充(每个坐标都加上填充的长度)
            bboxes[:,0] += int(left)
            bboxes[:,1] += int(top)
            bboxes[:,2] += int(left)
            bboxes[:,3] += int(top)
            return pad_image, bboxes, labels

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
    def __call__(self,images, bboxes, labels):
        H, W, _ = images.shape
        while True:
            # 确认随机裁剪的方式
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return images, bboxes, labels
            min_iou ,max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')
            for _ in range(50):
                cur_image = images
                # 裁剪后的宽高
                width = np.random.uniform(0.3 * W, W)
                height = np.random.uniform(0.3 * H, H)
                if height / width < 0.5 or height / width > 2:
                    continue
                # 起始的坐标
                left = np.random.uniform(W - width)
                top = np.random.uniform(H - height)
                # 获得此时裁剪后的左上角和右下角的坐标
                rect = np.array([int(left), int(top), int(left + width), int(top + height)])
                # 和所有的boxes进行iou的计算
                overlap = jaccard_numpy(bboxes, rect)
                if overlap.min() < min_iou and overlap.max() > max_iou:
                    continue
                # 获得裁剪后的图形
                crop_image = cur_image[rect[1]:rect[3],rect[0]:rect[2],:]

                # 排除掉中心点不在裁剪后的图形里面的bbox
                bbox_w = (bboxes[:,0] + bboxes[:,2]) / 2
                bbox_h = (bboxes[:,1] + bboxes[:,3]) / 2
                mask_w = (bbox_w < rect[2]) * (bbox_w > rect[0])
                mask_h = (bbox_h < rect[3]) * (bbox_h > rect[1])
                mask = mask_h * mask_w
                if not any(mask):
                    continue
                # 裁剪后所保留的标签框
                cur_bbox = bboxes[mask]
                cur_label = labels[mask]
                # 对标签框进行处理
                cur_bbox[:,:2] = np.maximum(cur_bbox[:,:2], rect[:2])
                cur_bbox[:,:2] -= rect[:2]
                cur_bbox[:,2:] = np.minimum(cur_bbox[:,2:], rect[2:])
                cur_bbox[:,2:] -= rect[:2]
                return crop_image, cur_bbox, cur_label

def jaccard_numpy(bboxes,rect):
    # bboxes = [n, 4], rect=[4, ]
    x1 = np.maximum(bboxes[:,0], rect[0])
    y1 = np.maximum(bboxes[:,1], rect[1])
    x2 = np.minimum(bboxes[:,2], rect[2])
    y2 = np.maximum(bboxes[:,3], rect[3])
    inter = (x2 - x1) * (y2 - y1)
    area1 = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])
    area2 = (rect[2] - rect[0]) * (rect[3] - rect[1])
    union = area2 + area1 - inter
    iou = inter / union
    return iou


class RandomMirror(object): # 进行水平的镜像操作
    def __call__(self,images, bboxes, labels):
        if np.random.randint(2):
            H, W, c = images.shape
            # 对图片进行翻转
            mirror_image = images.copy()
            mirror_image[:,:] = images[:,::-1]
            # 对bboxes进行翻转
            mirror_bboxes = bboxes.copy()
            mirror_bboxes[:,0] = W - bboxes[:,2] # 左右镜像需要注意这里是反的
            mirror_bboxes[:,2] = W - bboxes[:,0]
            return mirror_image, mirror_bboxes,labels
        else:
            return images, bboxes, labels

class ToPercentCoords(object):
    def __call__(self, images, bboxes, labels):
        H, W, c = images.shape
        bboxes[:,0] /= W
        bboxes[:,1] /= H
        bboxes[:,2] /= W
        bboxes[:,3] /= H
        return images,bboxes,labels

class Resize(object):
    def __init__(self,size):
        self.size = size
    def __call__(self,images, bboxes, labels):
        images = cv2.resize(images, (self.size, self.size))
        return images, bboxes, labels

class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self,images,bboxes,labels):
        images = images.astype(np.float32)
        images = images / 255.
        images = (images - self.mean) / self.std
        return images,bboxes,labels

class Augmentation(object):
    def __init__(self, size=640, mean=(0.406, 0.456, 0.485), std=(0.225, 0.224, 0.229)):
        # 此时的mean或者std应当是BGR格式的
        self.size = size
        self.mean = mean
        self.std = std
        self.augment = Compose([
            ConvertFromInts(),  # 将int类型转换为float32类型
            ToAbsoluteCoords(),  # 将归一化的相对坐标转换为绝对坐标
            Expand(self.mean),  # 扩充增强(此时输出的每一张图片的尺寸不一定相同)
            RandomSampleCrop(),  # 随机剪裁
            RandomMirror(),  # 随机水平镜像
            ToPercentCoords(),  # 将绝对坐标转换为归一化的相对坐标
            Resize(self.size),  # resize操作（将图片的尺寸给固定下来）
            Normalize(self.mean, self.std)  # 图像颜色归一化
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)