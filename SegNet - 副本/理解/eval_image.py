# 在测试集中取出一张图片进行可视化
import numpy as np
import torch
import ComVid_config
import ComVid_dataset
import model
import pandas as pd
from PIL import Image

def read_file(path):
    color_lst = []
    file = pd.read_csv(path,sep=',',engine='python')
    for i in range(len(file.index)):
        tmp = file.iloc[i]
        color_lst.append([tmp['r'],tmp['g'],tmp['b']])
    return color_lst


if __name__ == '__main__':
    # 导入模型
    model = model.SegNet(in_channel=3,output_channel=ComVid_config.num_classes).to(ComVid_config.device)
    model.load_state_dict(torch.load('params(200).param'))
    # 导入数据参数
    test_dataset = ComVid_dataset.LoadDataset([ComVid_config.test_image_path,
                                               ComVid_config.test_label_path], ComVid_config.crop_size)
    test_batch = test_dataset[0]
    test_image = test_batch[0].unsqueeze(0)
    test_label = test_batch[1].unsqueeze(0)
    # 开始进行验证
    model.eval()
    test_image = test_image.to(ComVid_config.device) # [1,3,h,w]
    test_label = test_label.squeeze() # [h,w],此时的test_label也是标签图,不是像素图
    y_hat, _ = model(test_image) # [1,num_class,h,w]
    pred = torch.argmax(y_hat,dim=1).squeeze() # [h,w]
    # 每个标签对应的像素
    color_lst = np.array(read_file(ComVid_config.class_dict_path))
    # 将预测标签转换为像素
    pred = pred.cpu().numpy()
    color_pred_image = color_lst[pred].astype(np.uint8) # 预测的标签图像
    color_pred_image = Image.fromarray(color_pred_image)
    color_pred_image.show()
    # 将真实标签转换为像素
    fact = test_label.numpy()
    color_label_image = color_lst[fact].astype(np.uint8)
    color_label_image = Image.fromarray(color_label_image)
    color_label_image.show()



