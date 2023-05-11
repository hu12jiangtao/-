import cv2
import numpy as np
import torch
import os
import transforms
import VOC07
from models import YOLOv2

class Config(object):
    def __init__(self):
        self.val_size = 416
        self.device = torch.device('cpu')
        self.data_root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
        self.vis_thresh = 0.33

def visualize(img, bboxes, scores, labels, vis_thresh, class_colors, class_names):
    ts = 0.4
    for idx, bbox in enumerate(bboxes):
        if scores[idx] > vis_thresh:
            cls_color = class_colors[idx]
            class_name = class_names[labels[idx]]
            mess = '%s: %.2f' % (class_name, scores[idx])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)
    return img

def plot_bbox_labels(img, bbox, mess, cls_color, text_scale):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img,(x1,y1), (x2,y2), cls_color, 2)
    test_size = cv2.getTextSize(mess, fontFace=0, fontScale=1, thickness=2)[0]
    if labels is not None:
        cv2.rectangle(img, (x1, y1-test_size[0]), (int(x1 + text_scale * test_size[1]), y1), -1)
        cv2.putText(img, mess, (int(x1), int(y1 - 5)), 0, text_scale,  (0,0,0))
    return img


if __name__ == '__main__':
    config = Config()
    # 导入yolo v2的模型
    model = YOLOv2(
        device=config.device,
        input_size=config.val_size,
        num_classes=20,
        trainable=False,
        topk=100,
        anchor_size=[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    ).to(config.device)
    model.load_state_dict(torch.load('checkpoints/epoch_21_57.3.pth'))
    model.eval()
    # 导入数据集
    val_dataset = VOC07.VOCDetection(
        root=config.data_root,
        img_size=config.val_size,
        image_sets=[('2007', 'trainval')],
        transform=None
        )
    val_transform = transforms.BaseTransform(config.val_size)
    num_image = len(val_dataset)

    # 给定每个标签的颜色
    class_names = VOC07.VOC_CLASSES
    np.random.randint(1)
    class_colors = [(np.random.randint(255),np.random.randint(255),
                     np.random.randint(255)) for _ in range(len(class_names))]

    for index in range(num_image):
        if index == 5:
            break
        img, _ = val_dataset.pull_image(index)
        x = val_transform(img)[0] # [H, W, c]
        x = torch.from_numpy(x[:,:,(2,1,0)]).permute(2,0,1) # [c, H, W]
        h, w = x.shape[1], x.shape[2]
        x = x.unsqueeze(0)
        x = x.type(torch.float32)
        bboxes, scores, labels = model(x)
        scale = np.array([w,h,w,h])
        bboxes *= scale
        processes_img = img_processes = visualize(img, bboxes, scores, labels, config.vis_thresh, class_colors, class_names)
        cv2.imwrite(os.path.join('results/out', str(index).zfill(6) + '.jpg'), processes_img)





