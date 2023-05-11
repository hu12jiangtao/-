import torchvision.transforms as T
from torchvision.transforms import functional as F
import torch
import numpy as np
import random
from PIL import Image

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target),dtype=torch.long)
        return image, target

class Compose(object):
    def __init__(self,transform_lst):
        self.transform_lst = transform_lst

    def __call__(self,image,target):
        for transform in self.transform_lst:
            image,target = transform(image,target)
        return image,target

class Normalize(object):
    def __init__(self,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self,image,target):
        image = F.normalize(image,mean=self.mean,std=self.std)
        return image,target

class RandomHorizontalFlip(object):
    def __init__(self,pred=0.5):
        self.pred = pred

    def __call__(self,image,target):
        if random.random() < self.pred:
            image = F.hflip(image)
            target = F.hflip(target)
        return image,target

class RandomResize(object):
    def __init__(self,min_size,max_size):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self,image,target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image,size)
        target = F.resize(target,size,interpolation=T.InterpolationMode.NEAREST)
        return image,target

def pad_if_small(x,crop_size,fill=0):
    w, h = x.size
    if min(w, h) < crop_size:
        pad_w = crop_size - w if w < crop_size else 0
        pad_h = crop_size - h if h < crop_size else 0
        x = F.pad(x,[0, 0, pad_w, pad_h],fill=fill) # 顺序为左上右下
    return x


class RandomCrop(object):
    def __init__(self,size):
        self.size = size

    def __call__(self,image,target):
        image = pad_if_small(image,self.size,fill=0) # 确保h,w大于crop_size
        target = pad_if_small(target,self.size,fill=255)
        crop_param = T.RandomCrop.get_params(image,(self.size,self.size))
        image = F.crop(image, *crop_param)
        target = F.crop(target, *crop_param)
        return image,target


def get_transform(mode):
    base_size = 520
    crop_size = 480
    if mode == 'train':
        return SegmentationPresetTrain(base_size, crop_size)
    else:
        return SegmentationPresetEval(base_size)

class SegmentationPresetTrain(object):
    def __init__(self,base_size,crop_size,hflip_prob=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        max_size = int(1.2 * base_size)
        min_size = int(0.5 * base_size)
        transforms = [RandomResize(min_size,max_size)]
        if hflip_prob > 0:
            transforms.append(RandomHorizontalFlip(hflip_prob))
        transforms.extend([RandomCrop(crop_size),ToTensor(),Normalize(mean=mean,std=std)])
        self.transforms = Compose(transforms)

    def __call__(self,img,target):
        img,target = self.transforms(img,target)
        return img,target

class SegmentationPresetEval(object):
    def __init__(self,base_size,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.transforms = [RandomResize(base_size,base_size),ToTensor(),Normalize(mean=mean,std=std)]
        self.transforms = Compose(self.transforms)

    def __call__(self,img,target):
        img,target = self.transforms(img,target)
        return img,target


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    image = np.array([[[1,2,31,45],[8,6,7,61],[21,4,67,55]],[[1,24,3,5],[82,61,1,6],[1,4,6,5]],
                      [[1,2,1,4],[8,6,7,6],[1,41,37,25]]],dtype=np.uint8) # [3,3,4]
    target = np.array([[1,0,3,2],[6,2,4,3],[1,3,5,7]],dtype=np.uint8) # [3,4]
    image = image.transpose((0,2,1))
    image = Image.fromarray(image)
    target = Image.fromarray(target)
    transform = Compose([RandomResize(3,6),RandomHorizontalFlip(),RandomCrop(3),ToTensor(),Normalize()])
    image,target = transform(image,target)
    print(image)
    print(target)

