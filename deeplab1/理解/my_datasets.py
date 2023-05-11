from torch.utils import data
import torch
import os
from PIL import Image

class VOCSegmentation(data.Dataset):
    def __init__(self,voc_root, year="2012", transforms=None, txt_name: str = "train.txt"):
        super(VOCSegmentation, self).__init__()
        assert year in ['2007', '2012']
        root = os.path.join(voc_root, 'VOCdevkit', f'VOC{year}')
        image_path = os.path.join(root, 'JPEGImages')
        mask_path = os.path.join(root, 'SegmentationClass')
        txt = os.path.join(root,'ImageSets','Segmentation',txt_name)
        file_name = open(txt,'r').read().split('\n')[:-1]
        self.image_path = [os.path.join(image_path,i + '.jpg') for i in file_name]
        self.mask_path = [os.path.join(mask_path,i + '.png') for i in file_name]
        assert len(self.image_path) == len(self.mask_path)
        self.transforms = transforms

    def __len__(self):
        return len(self.mask_path)

    def __getitem__(self, item):
        image = Image.open(self.image_path[item]).convert('RGB')
        mask = Image.open(self.mask_path[item])
        # 进行数据的增强后输出图片和标签
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        return image,mask

def collect_fn(batch): # 目标为每个batch中输出
    # 此时的batch是一个长度为batch，其中的元素为image,mask的元组
    images,masks = [i for i in zip(*batch)]
    images = pad_for_max(images,0)
    masks = pad_for_max(masks,255) # 标签值为255的计算损失时应当忽略
    return images,masks

def pad_for_max(x,fill_value):
    batch = len(x)
    max_shape = tuple([max(j) for j in zip(*[i.shape for i in x])])
    max_shape = (batch,) + max_shape
    batches_x = x[0].new(*max_shape).fill_(fill_value)
    for now_x,pad_x in zip(x,batches_x):
        pad_x[...,:now_x.shape[-2],:now_x.shape[-1]].copy_(now_x)
    return batches_x

if __name__ == '__main__':
    voc_root = 'D:\\python\\pytorch作业\\计算机视觉\\SegNet'
    a = VOCSegmentation(voc_root,year='2007')
