# 此文件可以给出目标检测的评估指标(MAP)
import time
import numpy as np
import VOC07
import transform
from models import myYOLO
import torch
import os
import pickle
import xml.etree.ElementTree as ET

class VOCAPIEvaluator():
    def __init__(self,data_root, img_size, device, transform, set_type='val'):
        # 首先给出用于验证的数据集
        self.dataset = VOC07.VOCDetection(root=data_root, img_size=img_size,
                                    image_sets=[('2007', set_type)],
                                    transform=transform)
        self.VOC_CLASSES = VOC07.VOC_CLASSES
        self.device = device
        self.save_root = 'results' # 用于存储不同的类别的相关信息
        self.class_save_root = os.path.join(self.save_root,'label_informer','det_val_%s.txt') # 存储预测的每一类预测的置信度和坐标信息
        self.det_file = os.path.join(self.save_root, 'detections.pkl') # 存储self.all_boxes，用于获得self.class_save_root
        self.image_lst = os.path.join(data_root,'VOC2007','ImageSets', 'Main', 'val.txt')
        self.annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        self.all_object_label = os.path.join(self.save_root, 'annots.pkl') # 用于存储所有真实的图片的所有检测目标的信息
        self.output_dir = os.path.join(self.save_root,'maps')

    def evaluate(self, net):
        if not os.path.exists(os.path.join(self.save_root, 'label_informer')):
            # 首先获得网络预测得到的bbox(bbox[i][j]代表着第i类，第j张图的检测目标的信息)
            self.all_boxes = [[[] for _ in range(len(self.dataset))] for _ in range(len(self.VOC_CLASSES))]
            # 利用for循环将预测的标签框填入
            for i in range(len(self.dataset)):
                # 从验证集中导出一张图片,输入模型中得到预测边框bbox(相对的位置)
                im, gt, h, w = self.dataset.pull_item(i) # im.shape=[c,h,w]
                im = im.unsqueeze(0) # [1,c,h,w]
                im = im.type(torch.float32)
                im = im.to(self.device)
                t0 = time.time()
                bboxes, save_score, save_label = net(im)  # 得到了预测的锚框[n, 4], save_label.shape=[n,]
                pred_time = time.time() - t0
                # 将bbox从相对坐标转换为相对坐标
                scale = np.array([w,h,w,h])
                bboxes *= scale
                for j in range(len(self.VOC_CLASSES)): # 类别
                    choice = np.where(save_label == j)[0]
                    if len(choice) == 0:
                        self.all_boxes[j][i] = np.empty([0,5],dtype=np.float32)
                        continue
                    else:
                        c_bboxes = bboxes[choice] # [n, 4]
                        c_score = save_score[choice] # [n, ]
                        c_det = np.concatenate([c_bboxes,c_score.reshape(-1,1)],axis=1) # [n, 5]
                        self.all_boxes[j][i] = c_det
                if i % 500 == 0:
                    print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, len(self.dataset), pred_time))

            # 第二步将self.all_boxes中的内容按照类放入不同的txt文件中(已经进行验证，验证正确)
            for j, cls in enumerate(self.VOC_CLASSES):
                filename = self.class_save_root % (cls) # 获得当前的路径
                with open(filename,'wt') as f:
                    for im_ind, index in enumerate(self.dataset.ids):
                        dets = self.all_boxes[j][im_ind]
                        if dets == []:
                            continue
                        else:
                            for k in range(dets.shape[0]):
                                f.write(f'{index[-1]:s} {dets[k,-1]:.3f} {dets[k,0]+1:.1f} '
                                        f'{dets[k,1]+1:.1f} {dets[k,2]+1:.1f} {dets[k,3]+1:.1f}\n')
        # 进行每个类之间的AP运算
        aps = []
        for i, cls in enumerate(self.VOC_CLASSES):
            # 导入当前类别预测的bbox
            filename = self.class_save_root % (cls)
            r, p, ap = self.voc_eval(filename, cls, 0.5)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': r, 'prec': p, 'ap': ap}, f)
        # 计算得到map的值
        self.map = np.mean(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))




    def voc_eval(self,detpath, classname, ovthresh):
        # 第三步对于验证集中的所有编号的图片，读取相对应的xml文件，并将检测目标的信息存储到recs中
        # recs为一个列表，列表的每一个元素为一张图片中的所有检测目标(每个检测目标中的信息都包含在一个字典中)
        with open(self.image_lst, 'r') as f:
            lines = f.readlines()
        imagenames = [line.strip() for line in lines]
        if not os.path.exists(self.all_object_label):
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
            with open(self.all_object_label, 'wb') as f:
                pickle.dump(self.all_object_label, f)
        else:
            recs = pickle.load(open(self.all_object_label, 'rb'))
        # 第四步将recs中的检测目标进行按类的区分(同时得到每个类别的标签框的个数npos)
        # class_recs[imagename]中存放着在imagename中类别为classname的检测目标的信息
        class_recs = {}
        npos = 0
        for i, imagename in enumerate(imagenames):
            R = [obj for obj in recs[imagename] if obj['name'] == classname] # imagename这张图片中检测目标的类别是当前类的信息(R为列表)
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox':bbox, 'difficult':difficult, 'det':det}
        # 计算标签框和预测框之间的AP值
        # 1.导入预测框
        with open(detpath, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1: # 当前类存在预测框的情况
            splitlines = [line.strip().split(' ') for line in lines]
            image_idx = [x[0] for x in splitlines] # 给出该类别的预测框的编号[pred_num, ]
            confidence = np.array([x[1] for x in splitlines]) # 每个预测框的置信度[pred_num, ]
            BB = np.array([[float(y) for y in x[2:]] for x in splitlines]) # 预测的边框的坐标[pred_num, 4]
            # 将预测框按照confidence的顺序进行排列
            idx = np.argsort(confidence)[::-1]
            image_idx = [image_idx[i] for i in idx]
            BB = BB[idx]

            nd = len(image_idx)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_idx[d]] # 获得当前图片编号的标签框的信息(为一个字典)

                BB = BB[d] # 选中的预测框的坐标
                BBGT = R['bbox'] # 当前图片下该类的标签框 [num,4]
                if BBGT.shape[0] != 0:
                    # 计算预测框和所有当前图片下该类的标签框的信息
                    xmin = np.maximum(BB[0], BBGT[:, 0])
                    ymin = np.maximum(BB[1], BBGT[:, 1])
                    xmax = np.minimum(BB[2], BBGT[:, 2])
                    ymax = np.minimum(BB[3], BBGT[:, 3])
                    iw = np.maximum(0., xmax - xmin)
                    ih = np.maximum(0., ymax - ymin)
                    inter = iw * ih
                    union = (BBGT[:, 3] - BBGT[:, 1]) * (BBGT[:, 2] - BBGT[:, 0]) + (BB[2] - BB[0]) * (BB[3] - BB[1]) - inter
                    iou = inter / union # [num, ]
                    # 找出iou值最大的标签框，将这个标签框的det设置为True，若大于阈值则将此时的tp设置为1，否则fp设置为1
                    over_max = np.max(iou)
                    j_max = np.argmax(iou)
                    if over_max > ovthresh:
                        if R['det'][j_max] is False: # 当前的标签框无对应的预测框
                            R['det'][j_max] = True
                            tp[d] = 1
                        else:
                            fp[d] = 1
                    else:
                        fp[d] = 1
            # 获取tp和fp值，fn的值就是npos
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            fn = float(npos)
            # 获得PR的值
            p = tp / np.maximum(np.finfo(np.float64).eps, tp + fp) # [pred_n, ]
            r = tp / fn # [pred_n, ]
            # 根据PR值计算这个类的AP值
            ap = self.voc_ap(r, p)
        else:
            p = -1
            r = -1
            ap = -1
        return r, p, ap


    def voc_ap(self, r, p): # 计算AP的值
        p = np.concatenate(([0.], p, [0.]))
        r = np.concatenate(([0.], r, [0.]))
        # 计算ap的part1(R的处理，当前项减去前面一项)
        p_forward = p[:-1]
        p_backward = p[1:]
        p_result = p_backward - p_forward
        choice = np.where(p_result != 0)[0]
        part1 = p_result[choice]
        # 计算ap的part2，p的处理，取当前项
        for i in range(p.shape[0] - 1, 0, -1):
            p[i - 1] = np.maximum(p[i], p[i - 1])
        part2 = p[choice + 1]
        ap = sum(part1 * part2)
        return ap



    def parse_rec(self,image_name):  # 输入一个xml文件(图片的编号)，需要返回这个编号的图片信息
        tree = ET.parse(image_name)
        objs = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['difficult'] = obj.find('difficult').text # 当前的类别是否好检测
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
            objs.append(obj_struct)
        return objs










if __name__ == '__main__':
    torch.manual_seed(1)
    val_size = 416
    data_root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
    device = torch.device('cuda')
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    val_transform = transform.BaseTransform(val_size, pixel_mean, pixel_std)
    model = myYOLO(device, 416, 20, trainable=False).to(device) # 输入的大小为416
    model.eval()
    model.set_grid(val_size)
    evaluator = VOCAPIEvaluator(
        data_root=data_root,
        img_size=val_size,
        device=device,
        transform=val_transform
    )
    evaluator.evaluate(model)
