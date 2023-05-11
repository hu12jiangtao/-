from torch import nn
import torch
import config
import numpy as np
from PIL import Image
import cv2
import torchvision
import collections
from torch.utils import data
from torch.nn import functional as F



def cv_read(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return cv_img

def BGR_to_RGB(cv_img):
    pil_img = cv_img.copy()
    pil_img[:,:,0] = cv_img[:,:,2]
    pil_img[:,:,2] = cv_img[:,:,0]
    return pil_img


if __name__ == '__main__':
    # x = torch.randn(size=(1,1024,30,30))
    # up_sample = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
    # print(up_sample(x).shape)

    # L_root = 'D:\\python\\pytorch作业\\计算机视觉\\u-net\\DRIVE\\training\\mask\\21_training_mask.gif'
    # L = Image.open(L_root).convert('L')
    # print(np.array(L)[200][200])


    # RGB_root = 'D:\\python\\pytorch作业\\计算机视觉\\u-net\\DRIVE\\training\\images\\21_training.tif'  # cv读取的路径中是没有中文的
    # arr = cv_read(RGB_root)  # 此时利用cv2读取的图片的通道应为BGR,第一个维度为高，第二个维度为宽
    # print(arr.shape)

    # root = 'D:\\python\\pytorch作业\\计算机视觉\\u-net\\DRIVE\\training\\1st_manual\\21_manual1.gif'
    # root = 'D:\\python\\pytorch作业\\计算机视觉\\u-net\\DRIVE\\training\\mask\\21_training_mask.gif'
    # image = Image.open(root).convert('L')
    # image.show()
    # image1 = np.array(image)
    # print(image1[100][100])
    # print(image1.shape)
    # trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # data = np.array(trans(image).squeeze())
    # print(data.shape)
    # print(data[100][100])
    # image = Image.fromarray(data)
    # image.show()

    # a = np.array([[1,2],[3,4]])
    # b = a.copy() #
    # a[0][0] = 4
    # print(b)

    # root = 'D:\\python\\pytorch作业\\计算机视觉\\u-net\\DRIVE\\training\\1st_manual\\21_manual1.gif'
    # image = Image.open(root).convert('L')
    # trans = torchvision.transforms.ToTensor()
    # a = trans(image).squeeze()
    # b = torch.as_tensor(np.array(image))
    # c = 0
    # d = (a == b).reshape(-1)
    # print(d[100])
    # for i in (a == b).reshape(-1):
    #     print(i)
    # print(c)

    # def f(x):
    #     return 0.1 * x
    #
    # net = nn.Linear(4,3)
    # opt = torch.optim.Adam(net.parameters(),lr=0.01)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(opt,lambda epoch:f(epoch)) # 此时的学习率为返回的内容乘上当前的学习率
    # for i in range(10):
    #     print(opt.param_groups[0]['lr'])
    #     scheduler.step()

    # class AddMachine(object):
    #     def __init__(self,n=2):
    #         self.data = [0.] * n
    #
    #     def add(self,*args):
    #         self.data = [float(i) + j for i,j in zip(self.data,args)]
    #
    # d3 = collections.defaultdict(AddMachine)  # 此时说明键值一定是这个AddMachine类对象
    # d3['x'] = AddMachine(2)
    # d3['x'].add(3,4) # 此时d3['x']就是类对象
    # print(d3['x'].data)

    # delimiter = '\t'
    # header = f'Epoch:[{1}]'
    # space_fmt = ':' + str(len(str(10))) + 'd'
    #
    # log_msg = delimiter.join([
    #     header,
    #     '[{0' + space_fmt + '}/{1}]',
    #     'eta: {eta}',
    #     '{meters}',
    #     'time: {time}',
    #     'data: {data}',
    #     'max mem: {memory:.0f}'
    #     ])
    # print(log_msg)

    from collections import deque

    # # 创建一个空的deque, 并指定最大的元素是3个
    # data = deque(maxlen=3)
    # data.append(1)
    # data.append(2)
    # data.append(3)
    # print(data)
    # data.append(4)
    # print(data)

    # batch_shape = (1,3,4,4)
    # image = torch.zeros(size=(3,4,4))
    # a = image.new(*batch_shape)
    # print(a)

    # batched_imgs = torch.zeros(size=(2,3,4,4))
    # images = [torch.arange(12).reshape(3,2,2),torch.arange(27).reshape(3,3,3)]
    # for img, pad_img in zip(images, batched_imgs):
    #     pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img) # 将pad_img中填充img
    # print(batched_imgs)

    # x = torch.randn(size=(2,2,1,3))
    # y = torch.tensor([[[0,255,1]],[[1,0,255]]],dtype=torch.long) # [2,1,3]
    # weights = torch.tensor([1,1],dtype=torch.float32)
    # l = F.cross_entropy(x, y, ignore_index=255, weight=weights)
    # print(l)
    #
    # x_1 = x.permute(0,2,3,1)
    # x_1 = x_1.reshape(-1,x_1.shape[-1])
    # y_1 = y.reshape(-1)
    # print(x_1)
    # print(y_1)
    # loss = nn.CrossEntropyLoss(reduction='none',ignore_index=255, weight=weights)
    # print(loss(x_1,y_1))
    #
    #
    # a = torch.randn(size=(4,2))
    # b = torch.tensor([1,0,255,0])
    # loss1 = nn.CrossEntropyLoss(reduction='none', ignore_index=255, weight=weights)
    # print(loss1(a,b))

    # mask = np.array([False, False ,False ,False ,False ,False ,False ,False ,False ,False, False ,False,
    #                     False ,False ,False ,False, False, False ,False, False])
    # a = np.array([255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255 ,255,
    #                 255 ,255])
    # print(a[mask])

    # torch.manual_seed(1)
    # x = torch.tensor([[1,3,2],[5,3,1]],dtype=torch.float32)
    # x = F.softmax(x,dim=1)
    # y = torch.tensor([0,2])
    # loss1 = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    # l1 = loss1(x,y).mean()
    # loss2 = nn.CrossEntropyLoss()
    # l2 = loss2(x,y)
    # print(l1,l2)

    a = torch.randn(size=(4,3))
    b = torch.tensor([1,0,2,1])
    loss1 = nn.CrossEntropyLoss(reduction='none',ignore_index=0)
    loss2 = nn.CrossEntropyLoss(ignore_index=0)
    print(loss2(a,b))
    print(sum(loss1(a,b)) / 3)








