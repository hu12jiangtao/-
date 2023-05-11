from torchvision.transforms import functional as F
import torchvision.transforms as T
import random
import torch
import numpy as np
from PIL import Image
# 对于训练集的数据需要完成:1.尺寸的重构 2.水平翻转 3.垂直翻转 4.随机裁剪 5.转换为tensor矩阵 6.进行归一化处理
# 验证的数据集的数据增强只要求:1.转换为tensor矩阵 2.进行归一化处理

class RandomResize(object):
    def __init__(self,min_size,max_size):
        if max_size is None:
            max_size = min_size
        self.max_size = max_size
        self.min_size = min_size

    def __call__(self,image,target):
        size = random.randint(self.min_size,self.max_size)
        image = F.resize(image, size)
        target = F.resize(target,size, T.InterpolationMode.NEAREST)
        return image,target

class RandomHorizontalFlip(object):
    def __init__(self,prob):
        self.prob = prob

    def __call__(self,image,target):
        if random.random() > self.prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image,target

class RandomVerticalFlip(object):
    def __init__(self,prob):
        self.prob = prob

    def __call__(self,image,target):
        if random.random() > self.prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomCrop(object): # 标签和输入矩阵的图片一起进行变动
    def __init__(self,crop_size):
        self.crop_size = crop_size

    def __call__(self,image,target): # 需要考虑到原先的图片尺寸没有crop_size这么大
        image = pad(image, self.crop_size,fill_value=255) # 填充的值可以是任意一个
        target = pad(target, self.crop_size, fill_value=255) # 填充值必须是255
        crop_param = T.RandomCrop.get_params(image,(self.crop_size,self.crop_size))
        image = F.crop(image,*crop_param)
        target = F.crop(target,*crop_param)
        return image,target

def pad(x, crop_size,fill_value): # 此时的x为PIL类型的图片
    x_w, x_h = x.size
    if min(x_h,x_w) < crop_size:
        pad_h = crop_size - x_h if x_h < crop_size else 0
        pad_w = crop_size - x_w if x_w < crop_size else 0
        x = F.pad(x,[0, 0, pad_w, pad_h],fill_value) # 填充的顺序为左上右下
    return x


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
        image = T.Normalize(mean=self.mean,std=self.std)(image)
        return image,target


class Compose(object):
    def __init__(self,transform_lst):
        self.transform_lst = transform_lst

    def __call__(self,image,target):
        for transform in self.transform_lst:
            image, target = transform(image,target)
        return image,target


def get_transform(train,mean,std):
    base_size = 565
    crop_size = 480
    if train is True:
        return SegmentationPresetTrain(base_size, crop_size, mean=mean, std=std)
    else:
        return SegmentationPresetEval(mean=mean, std=std)

class SegmentationPresetEval(object):
    def __init__(self,mean,std):
        self.transform = Compose([ToTensor(),Normalize(mean,std)])

    def __call__(self,image,target):
        image,target = self.transform(image,target)
        return image,target

class SegmentationPresetTrain(object):
    def __init__(self,base_size, crop_size, mean, std, h_prob=0.5, v_prob = 0.5):
        min_size = int(0.5 * base_size)
        max_size = int(1.2 * base_size)
        transform_lst = [RandomResize(min_size,max_size)]
        if h_prob > 0:
            transform_lst.append(RandomHorizontalFlip(h_prob))
        if v_prob > 0:
            transform_lst.append(RandomVerticalFlip(v_prob))
        transform_lst.append(RandomCrop(crop_size))
        transform_lst.append(ToTensor())
        transform_lst.append(Normalize(mean,std))
        self.transform = Compose(transform_lst)

    def __call__(self,image,target):
        image,target = self.transform(image,target)
        return image,target

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
    transformer = SegmentationPresetTrain(base_size, crop_size, mean, std)
    a, target = transformer(a,target)
    print(a)
    print(target)

