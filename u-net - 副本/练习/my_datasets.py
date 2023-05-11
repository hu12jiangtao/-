from torch.utils import data
import os
from PIL import Image
import cv2
import numpy as np

def read_cv(path):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1) # 此时转换为numpy的矩阵的格式

def BGR_to_RGB(x):
    y = x.copy()
    y[...,0] = x[...,2]
    y[...,2] = x[...,0]
    return y

class DriveDataset(data.Dataset):
    def __init__(self,root,train,transforms=None):
        super(DriveDataset, self).__init__()
        # 目标为给出 用到的 mask_image, input_image, label_image的路径列表
        # 不行，下面的表现较好
        # self.transform = transforms
        # root_path = os.path.join(root,'training' if train is True else 'test')
        # mask_path = os.path.join(root_path,'mask')
        # self.roi_mask = [os.path.join(mask_path,i) for i in os.listdir(mask_path)]
        # input_path = os.path.join(root_path,'images')
        # self.img_list = [os.path.join(input_path,i) for i in os.listdir(input_path)]
        # label_path = os.path.join(root_path,'1st_manual')
        # self.manual = [os.path.join(label_path,i) for i in os.listdir(label_path)]

        self.transform = transforms
        # 创建三个列表，其中分别存放三个不同图片的路径
        self.flag = 'training' if train else 'test'
        self.root_path = os.path.join(root, self.flag)
        img_lst = [i for i in os.listdir(os.path.join(self.root_path,'images')) if i.endswith('.tif')]
        self.img_list = [os.path.join(self.root_path,'images',i) for i in img_lst] # 输入模型的图片的路径列表
        self.manual = [os.path.join(self.root_path,'1st_manual',i.split('_')[0] + '_manual1.gif') for i in img_lst]
        self.roi_mask = [os.path.join(self.root_path,'mask',i.split('_')[0] + f'_{self.flag}_mask.gif') for i in img_lst]

    def __len__(self):
        return len(self.roi_mask)

    def __getitem__(self, item):
        # 此时需要输出的目标是:彩色输入模型的图片经过了数据增强，并且根据mask和label给出标签矩阵
        #
        # 读取tif类的图片需要利用到cv2(此时的通道顺序为BGR)
        image = BGR_to_RGB(read_cv(self.img_list[item])) # 得到彩色图片的numpy的矩阵
        # mask标签矩阵的创建
        label = Image.open(self.manual[item]).convert('L')
        # 对于细胞外的部分在计算损失的时候应当不用考虑，因此其标签设置成255
        mask = Image.open(self.roi_mask[item]).convert('L')
        label = np.array(label) / 255 # 此时神经为1，其余部分的标签为0
        mask = np.array(mask) # 此时细胞内的部分为255，细胞外的部分为0
        mask = 255 - mask # 此时细胞内的部分为0，细胞外的部分为255
        label = np.clip(mask + label, a_min=0, a_max=255) # 此时细胞外的像素标签255，细胞内的神经的标签为1，细胞内的其余部分的标签为0
        # 转换为PIL格式的图片进行数据增强
        image = Image.fromarray(image)
        label = Image.fromarray(label)
        if self.transform is not None:
            image,label = self.transform(image,label)
        return image, label



def collate_fn(batch): # 此时的目标为使一个批量的图片都扩展至最大
    # 此时的batch传入的应当是长度为batch，元素为一个元组(包含输入的image和label)
    # 首先将target和label分开
    image,target = list(zip(*batch))
    # 将image，target扩展至最大的尺寸大小
    image = pad_content(image,fill_value=0) # fill_value的值可以是任意的
    target = pad_content(target,fill_value=255) # 由于之后计算损失，因此fill_value的值必须是255
    return image, target

def pad_content(x,fill_value):
    shape = tuple([max(k) for k in zip(*[i.shape for i in x])])
    max_shape = (len(x),) + shape
    pad_image = x[0].new(*max_shape).fill_(fill_value)
    for img, p_img in zip(x,pad_image):
        p_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return pad_image



if __name__ == '__main__':
    root = 'DRIVE'
    train = True
    a = DriveDataset(root,train)




