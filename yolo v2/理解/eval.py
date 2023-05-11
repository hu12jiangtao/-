# 此文件下的函数是用来计算验证集中的map
# 此时求解出的内容和yolo v1 中求解出的相同(person:0.0064,map:0.9497)，但是和yolo v2的不相同
import time
import torch
import numpy as np
import transforms
from models import YOLOv2
from VOC07 import VOC_CLASSES,VOCDetection
import os
from xml.etree import ElementTree as ET
import pickle


class VOCAPIEvaluator():
    def __init__(self,data_root, img_size, device, transform, set_type='val'):
        # 首先给出用于验证的数据集
        self.dataset = VOCDetection(root=data_root, img_size=img_size,
                                    image_sets=[('2007', set_type)],
                                    transform=transform)  # 此时用val中的图片进行测式
        self.VOC_CLASSES = VOC_CLASSES
        self.device = device
        self.save_root = 'results' # 用于存储不同的类别的相关信息
        self.class_save_root = os.path.join(self.save_root,'label_informer','det_val_%s.txt') # 存储预测的每一类预测的置信度和坐标信息
        self.det_file = os.path.join(self.save_root, 'detections.pkl') # 存储self.all_boxes，用于获得self.class_save_root
        self.image_lst = os.path.join(data_root,'VOC2007','ImageSets', 'Main', 'val.txt')
        self.annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        self.all_object_label = os.path.join(self.save_root, 'annots.pkl') # 用于存储所有真实的图片的所有检测目标的信息
        self.output_dir = os.path.join(self.save_root,'maps')

    def evaluate(self,net):
        net.eval()
        net = net.to(self.device)
        net.trainable = False

        # all_boxes[i][j]代表第i类的第j张图片存储的内容
        self.all_boxes = [[[] for _ in range(len(self.dataset))] for _ in range(len(self.VOC_CLASSES))]
        for i in range(len(self.dataset)):
            # 首先进行前向传播得到经过筛选得到的预测框
            im, gt, h, w = self.dataset.pull_item(i) # [3, H, W]
            scale = np.array([[w, h, w, h]])
            im = im.unsqueeze(0).to(self.device) # [1, 3, H, W]
            im = im.type(torch.float32)
            t0 = time.time()
            bboxes, scores, labels = net(im)  # 此时的bboxes中存储的是相对于整张图的相对坐标 bboxes.shape=[n_objs,4]
            pred_time = time.time() - t0
            bboxes *= scale # 将其转换为绝对坐标
            for j in range(len(self.VOC_CLASSES)): # 每一个类
                choice = np.where(labels == j)[0]
                if len(choice) == 0:
                    self.all_boxes[j][i] = np.empty([0,5],dtype=np.float32)
                    continue
                else:
                    c_bbox = bboxes[choice] # [n, 4]
                    c_score = scores[choice] # [n, ]
                    c_det = np.concatenate([c_bbox, c_score.reshape(-1,1)],axis=1) # [n, 5]
                    self.all_boxes[j][i] = c_det
            if i % 500 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, len(self.dataset), pred_time))

        # 将all_boxes中的信息按照类别存储到txt文件中
        for cls_ind, cls in enumerate(self.VOC_CLASSES):
            save_file = self.class_save_root % (cls)
            with open(save_file, 'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids):
                    det = self.all_boxes[cls_ind][im_ind]
                    if det == []:
                        continue
                    else:
                        for k in range(det.shape[0]):
                            f.write(f'{index[1]:s} {det[k][-1]:.3f} {det[k][0]+1:.1f} '
                                    f'{det[k][1]+1:.1f} {det[k][2]+1:.1f} {det[k][3]+1:.1f}\n')

        # 按类获取标签框信息
        aps = []
        for cls in self.VOC_CLASSES:
            filename = self.class_save_root % (cls)
            r, p, ap = self.voc_eval(detpath=filename,  # 指定路径的txt文件
                                     classname=cls,  # 类别的名称
                                     ovthresh=0.5)  # 设置的IOU阈值
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': r, 'prec': p, 'ap': ap}, f)
        self.map = np.mean(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))


    def parse_rec(self,filename): # 导入一个xml的文件
        tree = ET.parse(filename)
        objs = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
            objs.append(obj_struct)
        return objs

    def voc_eval(self, detpath, classname, ovthresh):
        # 第一步将所有的标签框放入recs中
        with open(self.image_lst, 'r') as f:
            lines = f.readlines()
        image_names = [image_name.strip() for image_name in lines]

        if not os.path.exists(self.all_object_label):
            recs = {}
            for image_name in image_names:
                recs[image_name] = self.parse_rec(self.annopath % (image_name))
            with open(self.all_object_label, 'wb') as f:
                pickle.dump(recs, f)
        else:
            recs = pickle.load(open(self.all_object_label, 'rb'))

        # 得到当前类的标签框信息，存储在中class_recs中
        npos = 0 # 用于记录当前类别实际应当检测的目标个数
        class_recs = {}
        for image_name in image_names:
            R = [obj for obj in recs[image_name] if obj['name'] == classname]
            bbox = np.array([obj['bbox'] for obj in R])
            difficult = np.array([obj['difficult'] for obj in R]).astype(np.bool) # 其中的元素只有1或者0, 1代表的是较难检测
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[image_name] = {'bbox': bbox, 'difficult': difficult, 'det': det}  # 已经验证其是正确的

        # 得到当前类别的预测框的内容
        with open(detpath, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:
            split_lines = [line.strip().split(' ') for line in lines]
            image_ids = [line[0] for line in split_lines]
            confidence = np.array([float(line[1]) for line in split_lines])
            BB = np.array([[float(i) for i in line[2:]] for line in split_lines]) # 这个类别的所有预测框的坐标
            # 所有的预测框都是要按照置信度的大小进行排队
            sorted_ind = np.argsort(-confidence)
            image_ids = [image_ids[i] for i in sorted_ind]
            BB = BB[sorted_ind]
            # 进行该类别预测框和标准框的ap值
            nd = len(image_ids)
            tp = np.zeros((nd,))
            fp = np.zeros((nd,)) # []
            # 遍历每一个预测框，找出与这个预测框在同一张图片上的所有的标签框，计算两者的iou
            for d in range(nd):
                # 当前预测框的坐标
                cur_pred_axis = BB[d]
                cur_tg_obj = class_recs[image_ids[d]]
                cur_tg_axis = cur_tg_obj['bbox']
                # 两者进行iou的计算
                if cur_tg_axis.shape[0] != 0:
                    xx1 = np.maximum(cur_pred_axis[0], cur_tg_axis[:, 0])
                    yy1 = np.maximum(cur_pred_axis[1], cur_tg_axis[:, 1])
                    xx2 = np.minimum(cur_pred_axis[2], cur_tg_axis[:, 2])
                    yy2 = np.minimum(cur_pred_axis[3], cur_tg_axis[:, 3])
                    iw = np.maximum(xx2 - xx1, 0)
                    ih = np.maximum(yy2 - yy1, 0)
                    inter = iw * ih
                    union = (cur_pred_axis[2] - cur_pred_axis[0]) * (cur_pred_axis[3] - cur_pred_axis[1]) + \
                            (cur_tg_axis[:,2] - cur_tg_axis[:,0]) * (cur_tg_axis[:,3] - cur_tg_axis[:,1]) - inter
                    iou = inter / union
                    # 取出iou最大的时候
                    jmax = np.argmax(iou)
                    ovmax = np.max(iou)
                    if ovmax > ovthresh:
                        if not cur_tg_obj['difficult'][jmax]:
                            if not cur_tg_obj['det'][jmax]:
                                cur_tg_obj['det'][jmax] = True
                                tp[d] = 1
                            else:
                                fp[d] = 1
                    else:
                        fp[d] = 1
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            p = tp / np.maximum(np.finfo(np.float64).eps, tp + fp) # (nd,)
            r = tp / float(npos) # (nd,)
            # 根据pr求解ap值
            ap = self.voc_ap(r, p)
        else:
            p = -1
            r = -1
            ap = -1
        return r, p, ap

    def voc_ap(self, r, p):
        p_pad = np.concatenate(([0.], p, [0.]))
        r_pad = np.concatenate(([0.], r, [1.]))
        # 第二个部分 r值相同的情况下取p最大的那一行
        r_pad_first = r_pad[:-1]
        r_pad_last = r_pad[1:]
        choice = np.where(r_pad_last != r_pad_first)[0]
        r_pad_first = r_pad_first[choice]
        r_pad_last = r_pad_last[choice]
        diff_r_pad = r_pad_last - r_pad_first
        # 第一个部分为当前p以及之后部分的p的最大值
        for i in range(len(p_pad) - 1, 0, -1):
            p_pad[i - 1] = np.maximum(p_pad[i - 1], p_pad[i])
        p_pad = p_pad[choice + 1]
        ap = sum(p_pad * diff_r_pad)
        return ap


if __name__ == '__main__':
    torch.manual_seed(1)
    val_size = 416
    data_root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
    device = torch.device('cuda')
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    val_transform = transforms.BaseTransform(val_size, pixel_mean, pixel_std)
    model = YOLOv2(device, 416, 20, trainable=False, # # 输入的大小为416
                   anchor_size=[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]).to(device)
    model.eval()
    model.set_grid(val_size)
    evaluator = VOCAPIEvaluator(
        data_root=data_root,
        img_size=val_size,
        device=device,
        transform=val_transform
    )
    evaluator.evaluate(model)