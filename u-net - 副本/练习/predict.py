import torch
import Model
import torchvision
import config
from PIL import Image
import cv2
import numpy as np
import train_and_eval
import my_datasets
from torch.utils import data
import transforms

def read_cv(path):
    return cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1) # 此时转换为numpy的矩阵的格式

def BGR_to_RGB(x):
    y = x.copy()
    y[...,0] = x[...,2]
    y[...,2] = x[...,0]
    return y

if __name__ == '__main__':
    # 导入训练好的模型
    model = Model.Unet(in_channel=config.in_channel,num_classes=config.num_classes,base_channel=config.base_channel)
    model.load_state_dict(torch.load('params(300).param'))
    model.to(config.device)
    # 导入一张图片
    image_path = 'DRIVE\\test\\images\\01_test.tif'
    mask_path = 'DRIVE\\test\\mask\\01_test_mask.gif'
    label_path = 'DRIVE\\training\\1st_manual\\21_manual1.gif'
    mask = Image.open(mask_path).convert('L') # [565,584]
    label = Image.open(label_path).convert('L') # [565,584]
    image = BGR_to_RGB(read_cv(image_path))
    # 对输入模型的图片进行数据增强
    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    dice_label = np.array(label) / 255  # [565,584] # 需要利用到mask进行处理
    mask_image = np.array(mask)
    dice_mask = 255 - mask_image
    dice_label = np.clip(dice_mask + dice_label, a_min=0, a_max=255)  # [565,584]
    dice_label = Image.fromarray(dice_label)
    image = Image.fromarray(image)
    image,dice_label = transforms.get_transform(False,mean,std)(image,dice_label)
    image = image.unsqueeze(0).to(config.device)
    dice_label = dice_label.unsqueeze(0)
    # 开始进行验证
    model.eval()
    with torch.no_grad():
        y_hat = model(image) # [1,num_class,565,584]
        pred = torch.argmax(y_hat,dim=1).squeeze() # [565,584]
        # 保存预测的图片
        # 此时应该在mask的图片上加上预测的细胞内的像素标签,mask中细胞区域为255
        pred[pred == 1] = 255
        pred[mask_image == 0] = 0
        pred = pred.cpu().numpy().astype(np.uint8)
        pred = Image.fromarray(pred)
        pred.show()

    # 参数验证
    # mean = (0.709, 0.381, 0.224)
    # std = (0.127, 0.079, 0.043)
    # test_dataset = my_datasets.DriveDataset(config.data_path,train=False,
    #                                         transforms=transforms.get_transform(train=False,mean=mean,std=std))
    # print(test_dataset[0][0][:10][:10])
    # test_loader = test_dataset[0]
    # dice = train_and_eval.DiceCoefficient(num_classes=config.num_classes, ignore_index=255)
    # x, y = test_loader[0].unsqueeze(0).to(config.device), test_loader[1].unsqueeze(0).to(config.device)
    # print(y.shape)
    # outputs = model(x) # [batch,num_class,h,w]
    # pred_dice = torch.argmax(outputs,dim=1).cpu()
    # y_dice = y.cpu()
    # dice.update(pred_dice,y_dice)
    # cumulative_dice = dice.get_score()
    # print(cumulative_dice)




