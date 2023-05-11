import numpy as np
import random
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
# 此时要完成的是图片(输入模型的图片和标签图片)的大小重构，水平、垂直方向的图片翻转，图片的随机的裁剪，图片转换为矩阵，输入模型图片的归一化

class Compose(object):
    def __init__(self,transform_lst):
        self.transform_lst = transform_lst

    def __call__(self,image,target):
        for transform in self.transform_lst:
            image,target = transform(image,target)
        return image,target

class RandomHorizontalFlip(object):
    def __init__(self,prob):
        self.prob = prob

    def __call__(self,image,target):
        if random.random() > self.prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target

class RandomVerticalFlip(object):
    def __init__(self,prob):
        self.prob = prob

    def __call__(self,image,target):
        if random.random() > self.prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image,target

class ToTensor(object):
    def __call__(self,image,target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target,dtype=np.int64))
        return image,target

class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self,image,target):
        return F.normalize(image,mean=self.mean,std=self.std),target

class RandomResize(object):
    def __init__(self,min_size,max_size):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self,image,target):
        size = random.randint(self.min_size,self.max_size)
        image = F.resize(image,size)
        # 此时必须有interpolation=T.InterpolationMode.NEAREST，否则采用的线性插值会导致target中出现0，1，255之外的值
        target = F.resize(target,size,interpolation=T.InterpolationMode.NEAREST)
        return image,target

def pad_if_small(x,size,fill): # 正确
    min_size = min(x.size) # x.size既适用于灰度图像又适用于彩色图像，返回一个元组，第一个元素为宽，第二个元素为高
    if size > min_size:
        ow,oh = x.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        x = F.pad(x,[0,0,padw,padh],fill)  # F.pad的填充顺序为 左上右下
    return x

class RandomCrop(object): # 进行随机的剪裁 正确
    def __init__(self, size):
        self.size = size

    def __call__(self,image,target):
        image = pad_if_small(image,self.size,fill=255) # target填充值必须是任意的 # 若图片尺寸小于裁剪的尺寸时，会对原来的图片进行pad
        target = pad_if_small(target,self.size,255) # target填充值必须是255，在损失函数中丢弃填充部分需要利用
        # # (self.size, self.size)代表这输出图片的尺寸的大小，同时此时是随机选择初始坐标点
        crop_inform = T.RandomCrop.get_params(image,(self.size,self.size))
        image = F.crop(image,*crop_inform)
        target = F.crop(target,*crop_inform)
        return image,target

class SegmentationPresetTrain(object): # 正确
    def __init__(self,base_size, crop_size, hflip_prob=0.5, vflip_prob=0.5,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        transform_lst = [RandomResize(min_size,max_size)]
        if hflip_prob > 0:
            transform_lst.append(RandomHorizontalFlip(hflip_prob))
        if vflip_prob > 0:
            transform_lst.append(RandomVerticalFlip(vflip_prob))
        transform_lst.append(RandomCrop(crop_size))
        transform_lst.append(ToTensor())
        transform_lst.append(Normalize(mean,std))
        self.transform = Compose(transform_lst)

    def __call__(self,image,target):
        return self.transform(image,target)

class SegmentationPresetEval(object):  # 正确
    def __init__(self,mean, std):
        self.transform = Compose([ToTensor(),Normalize(mean,std)])

    def __call__(self,image,target):
        return self.transform(image,target)


def get_transform(train, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):  # 正确
    base_size = 565
    crop_size = 480

    if train:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    a = np.array([[[1,2,3,4],[5,6,7,8],[5,3,4,1]],[[1,3,2,4],[5,7,6,8],[5,4,3,1]]],dtype=np.uint8)
    a = a.transpose((0,2,1))
    a = Image.fromarray(a)
    target = np.array([[1,0,255,1],[0,0,1,255]])
    target = Image.fromarray(target)
    mean = (0,0,0)
    std = (1,1,1)
    base_size = 3
    crop_size = 2
    transformer = SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    a, target = transformer(a,target) # 得到的两个结果不相同
    print(a)
    print(target)





