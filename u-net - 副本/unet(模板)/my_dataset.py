import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data
import torchvision
import cv2
import torch

def BGR_to_RGB(image):
    a = image.copy()
    a[:,:,0] = image[:,:,2]
    a[:,:,2] = image[:,:,0]
    return a

def read_cv(path):
    return cv2.imdecode(np.fromfile(path),-1)

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        data_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names] # 输入网络的图片的具体路径所组成的列表
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names] # 标签的路径(标签和图片对应)
        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names] #
        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        img = BGR_to_RGB(read_cv(self.img_list[idx])) # 已经进行了归一化的操作（read_cv已经将其转换为了np的矩阵了）
        manual = Image.open(self.manual[idx]).convert('L') # 转换为灰度图(没有进行归一化的操作)，在标签中神经是用白色表示，其他部分都是黑色的
        manual = np.array(manual) / 255 # 进行了归一化的处理，神经的颜色为白色的值为1，其余的为黑色的，值为0(相当于两个类别的标号)
        roi_mask = Image.open(self.roi_mask[idx]).convert('L') # 转换为灰度图(白的值为255，黑的值为0)
        roi_mask = 255 - np.array(roi_mask) # 此时mask中圆内的颜色变为了黑色(值为0)，圆之外的颜色变为了白色(值为255)
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)
        # 圆之外的颜色变为了白色(值为255)，圆内神经的像素点的值为1，圆内其余像素点的值为0(为之后去除圈外像素造成的损失做铺垫)
        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)
        img = Image.fromarray(img)
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask # 此时的mask就是标签
    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch): # 针对于train_dataset[i]的形状不相同的情况，将其pad至同样的宽高
        # 此时的batch应当为元素为一个val_dataset[i],i为[0,len(val_dataset))中随机一个数，大小为传入的batch_size的值
        images, targets = list(zip(*batch)) # 将输入的image和targets给分开，此时images, targets都是一个长度为batch的列表
        batched_imgs = cat_list(images, fill_value=0) # 此时可以填充任意的值，因为在计算损失的时候不会对标签值为255的进行考虑
        batched_targets = cat_list(targets, fill_value=255) # 外部边界的标签为255，白色
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images])) # 此函数的作用就是用来处理导入的图片的尺寸大小不相同的情况下进行填充
    batch_shape = (len(images),) + max_size # batch_shape=[batch,3,480,480]
    # 此时的images[0].shape=[3,480,480]，此时的目标为创建一个形状为batch_shape的全0矩阵，之后填充fill_value
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img) # 将pad_img中填充img
    return batched_imgs


if __name__ == '__main__':
    torch.manual_seed(1)
    a1 = torch.randn(size=(4, 3, 4, 4))
    label1 = torch.randn(size=(4, 4, 4))

    dataset = data.TensorDataset(a1, label1)
    loader = data.DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=DriveDataset.collate_fn)

    for x, y in loader:
        print(x)
        print(y)



