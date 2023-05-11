# 此文件中的目标是随机抽取一种彩色图片输入模型区分出其细胞中的是否是神经(白色)还是其他的内容(黑色)
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision
import UNET
import config

def BGR_to_RGB(x):
    y = x.copy()
    y[...,0] = x[...,2]
    y[...,2] = x[...,0]
    return y

def read_cv(path):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)

if __name__ == '__main__':
    input_image_path = 'DRIVE\\test\\images\\01_test.tif'
    mask_image_path = 'DRIVE\\test\\mask\\01_test_mask.gif'
    print(BGR_to_RGB(read_cv(input_image_path)))
    # 将两者转换为矩阵的形式
    mask_image = np.array(Image.open(mask_image_path).convert('L')) # [584,565]
    input_image = BGR_to_RGB(read_cv(input_image_path))
    input_image = Image.fromarray(input_image) # PIL格式图片
    # 对数据进行数据增强
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    # 注意此时的标签图片的的mean、std和输入模型的彩色图片的mean、std相同
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=mean,std=std)])
    input_image = transform(input_image).unsqueeze(0) # [1,3,584,565]
    # 导入模型参数
    model = UNET.Unet(in_channel=3,num_class=2,base_channel=64).to(config.device)
    model.load_state_dict(torch.load('params(200次迭代).param'))
    # 开始进行验证
    with torch.no_grad():
        model.eval()
        input_image = input_image.to(config.device)
        y_hat = model(input_image) # [1,2,584,565],只对细胞内的像素进行预测(神经的标签为1)
        pred = torch.argmax(y_hat,dim=1).squeeze() # [584,565]
        # 此时应该在mask的图片上加上预测的细胞内的像素标签,mask中细胞区域为255
        pred[pred == 1] = 255
        pred[mask_image == 0] = 0
        pred = pred.cpu().numpy().astype(np.uint8)
        pred = Image.fromarray(pred)
        pred.show()







