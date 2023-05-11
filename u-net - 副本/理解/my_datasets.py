import os
import random

import torchvision
import cv2
import numpy as np
import torch
from torch.utils import data
from PIL import Image
# tif格式的图片为高清无压缩的图片，需要用cv.imread进行打开(形状为[h,w,3],此时的通道顺序为BGR，进行形状变换需要转为RGB格式)，利用PIL.Image无法打开

def BGR_to_RGB(cv_img):
    pil_img = cv_img.copy()
    pil_img[:,:,0] = cv_img[:,:,2]
    pil_img[:,:,2] = cv_img[:,:,0]
    return pil_img

def cv_read(file_path):
    return cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1) # 利用CV导入路径中含有中文的图片

class DriveDataset(data.Dataset):
    def __init__(self,root,train,transforms=None):
        super(DriveDataset, self).__init__()
        self.transform = transforms
        # 创建三个列表，其中分别存放三个不同图片的路径
        self.flag = 'training' if train else 'test'
        self.root_path = os.path.join(root, self.flag)
        img_lst = [i for i in os.listdir(os.path.join(self.root_path,'images')) if i.endswith('.tif')]
        self.img_list = [os.path.join(self.root_path,'images',i) for i in img_lst] # 输入模型的图片的路径列表
        self.manual = [os.path.join(self.root_path,'1st_manual',i.split('_')[0] + '_manual1.gif') for i in img_lst]
        self.roi_mask = [os.path.join(self.root_path,'mask',i.split('_')[0] + f'_{self.flag}_mask.gif') for i in img_lst]
        print(self.roi_mask[10])
        print(self.manual[10])
        print(self.img_list[10])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, item):
        # 最终的目标是mask的圈外的像素为255(损失时需要剔除)，mask圈内的神经的像素值对应的标签为1，非神经的像素对应的标签为0
        image = BGR_to_RGB(cv_read(self.img_list[item])) # 转换为RGB顺序的彩色图()
        manual = Image.open(self.manual[item]).convert('L') # 转换为灰度图(代表神经的白色的值为255，代表背景的黑色的值为0)
        manual = np.array(manual) / 255 # 圈内神经为1，其余部分为0
        roi_mask = Image.open(self.roi_mask[item]).convert('L') # 转换为灰度图(代表圈内为白色，圈外区域为黑色)
        roi_mask = 255 - np.array(roi_mask) # 圈外的像素值为255，圈内的像素值为0
        mask = np.clip(manual + roi_mask,a_max=255,a_min=0) # 此时相当于将mask当作了标签
        mask = Image.fromarray(mask) # 转换为PIL图片用于之后的数据增强
        image = Image.fromarray(image)
        # 进行数据增强(此处出现了问题)
        if self.transform is not None:
            image, mask = self.transform(image,mask)
        return image, mask # 此时的mask应当是圈外的像素值为255，圈内的神经的值为1，其余部分的像素值为0

def collate_fn(batch): # 解决dataset输出的图片的尺寸不一样的情况
    # 此时输入的batch是一个长度为batch，每个元素为(image,mask)
    images,target = list(zip(*batch)) # 此时的images,target都是一个列表
    images = pad_content(images,pad_value=0) # 此时的pad的值可以是任意的
    target = pad_content(target,pad_value=255) # 此时pad的值应为255，之后在计算损失时忽略像素值为255造成的损失
    return images,target

def pad_content(images, pad_value): # 由于u-net利用的是全卷积的网络，因此输入的每个批量输入输入的尺寸大小可以是不相同的
    # 首先得到输入的批量的最大尺寸
    max_size = tuple([max(s) for s in zip(*[img.shape for img in images])]) # 得到这个批量中的最大尺寸
    max_shape = (len(images),) + max_size
    batches_image = images[0].new(*max_shape).fill_(pad_value)
    for image, pad_image in zip(images,batches_image):
        pad_image[...,:image.shape[-2],:image.shape[-1]].copy_(image)
    return batches_image



if __name__ == '__main__':
    # trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # train_dataset = DriveDataset(root='D:\\python\\pytorch作业\\计算机视觉\\u-net\\DRIVE',train=True,transform=trans)
    root = 'DRIVE'
    train = True
    a = DriveDataset(root, train)





