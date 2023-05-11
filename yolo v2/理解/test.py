# 此文件中的目标:将预测的框标注到测试的图片上去
# 首先随机抽取一张图片输入模型得到通过模型预测的bbox、score、labels
# 给出一个score的阈值，确定需要可视化的bbox的颜色(根据标签进行确定)
# 根据bbox在图片中画出bbox，并且给打上标签

import os
import numpy as np
import torch
import VOC07
import transforms
from models import YOLOv2
import cv2

class Config(object):
    def __init__(self):
        self.root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
        self.device = torch.device('cuda')
        self.image_size = 416
        self.num_classes = 20
        self.save_path = 'results/outputs'
        self.vis_thresh = 0.3 #

def visualize(img, bboxes, scores, labels, vis_thresh, class_colors,class_names):
    ts = 0.4
    # imgs代表当前这张图片的CV的形式进行读取
    for i,bbox in enumerate(bboxes):
        if scores[i] > vis_thresh: # 说明当前大于分数的阈值，需要在图片中利用方框将其标出来
            cls_id = int(labels[i])
            cls_color = class_colors[cls_id] # 给出当前类别的颜色
            if len(class_names) > 1: # 当前的目标的种类大于一种
                mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            else:
                cls_color = [255, 0, 0]
                mess = None
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)
    return img


def plot_bbox_labels(img, bbox, label=None, cls_color=None, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # cv2.getTextSize的第一个参数为字符串、第二个参数为字体、第三个参数为字符比例、第四个参数为字符的粗细
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0] # 得到这个字符串的宽高
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    if label is not None:
        # t_size[0] * text_scale代表字符串的实际长度
        cv2.rectangle(img, (x1, y1 - t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
        # 写入字符串:(int(x1), int(y1 - 5))代表的是起始的坐标,字体类型为0, (0, 0, 0)代表字体的颜色
        cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0))
    return img








if __name__ == '__main__':
    config = Config()
    # 导入数据集
    test_dataset = VOC07.VOCDetection(config.root, img_size=None,image_sets=[('2007', 'val')],transform=None)
    test_transform = transforms.BaseTransform(size=config.image_size)
    # 导入模型和权重
    model = YOLOv2(
                    device=config.device,
                    input_size=config.image_size,
                    num_classes=20,
                    trainable=False,
                    topk=100,
                    anchor_size=[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
                   ).to(config.device)
    model.load_state_dict(torch.load(os.path.join('checkpoints', 'epoch_21_57.3.pth')))
    model.eval()
    # 标注边框的颜色
    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(config.num_classes)]
    # 在图片上进行标注
    num_images = len(test_dataset)
    for index in range(num_images):
        if index == 1: # 只对10张图进行可视化
            break
        image, _ = test_dataset.pull_image(index)
        # 对图片进行数据的增强
        x = test_transform(image)[0] # 此时通道数在后，同时通道为gbr
        # 将图片转换为RGB且通道数在前的tensor格式
        x = torch.from_numpy(x[:,:,(2,1,0)]).permute(2,0,1)
        h, w = x.shape[1], x.shape[2]
        x = x.unsqueeze(0).to(config.device)
        x = x.type(torch.float32)
        # 进行前向的传播
        bboxes, scores, labels = model(x) # 此时的bboxes为相对的左上角和右下角的坐标
        scale = np.array([w, h, w, h])
        bboxes = bboxes * scale
        class_names = VOC07.VOC_CLASSES
        print(scores)
        img_processes = visualize(image, bboxes, scores, labels, config.vis_thresh, class_colors, class_names)
        cv2.imshow('detection', img_processes)
        cv2.waitKey(0)




