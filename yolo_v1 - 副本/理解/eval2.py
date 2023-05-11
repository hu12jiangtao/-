# 此文件可以给出目标检测的评估指标(MAP)
import time
import numpy as np
from VOC07 import VOCDetection, VOC_CLASSES
import transform
from models import myYOLO
import torch
import os
import pickle
import xml.etree.ElementTree as ET

class VOCAPIEvaluator():
    def __init__(self,data_root, img_size, device, transform, set_type='val'):
        # 首先给出用于验证的数据集
        self.dataset = VOCDetection(root=data_root, img_size=img_size,
                                    image_sets=[('2007', set_type)],
                                    transform=transform)
        self.VOC_CLASSES = VOC_CLASSES
        self.device = device
        self.save_root = 'results' # 用于存储不同的类别的相关信息
        self.class_save_root = os.path.join(self.save_root,'label_informer','det_val_%s.txt') # 存储预测的每一类预测的置信度和坐标信息
        self.det_file = os.path.join(self.save_root, 'detections.pkl') # 存储self.all_boxes，用于获得self.class_save_root
        self.image_lst = os.path.join(data_root,'VOC2007','ImageSets', 'Main', 'val.txt')
        self.annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        self.all_object_label = os.path.join(self.save_root, 'annots.pkl') # 用于存储所有真实的图片的所有检测目标的信息
        self.output_dir = os.path.join(self.save_root,'maps')

    def evaluate(self, net):
        # 首先获得网络预测得到的bbox
        net.eval()
        num_images = len(self.dataset)
        # 此时的self.all_boxes[i][j]中存储的是第i个类别的第j张图片的目标的 坐标和confidence
        self.all_boxes = [[[] for _ in range(num_images)] for _ in range(len(self.VOC_CLASSES))]
        # 向self.all_boxes中填充入内容
        for i in range(num_images):
            # 获得当前增强后的图片以及坐标信息
            im, gt, h, w = self.dataset.pull_item(i) # [channel, H, W]
            img = im.unsqueeze(0).to(self.device) # [1, channel, H, W]
            img = img.type(torch.float32)
            t0 = time.time()
            # bboxes.shape = [n_obj, 4], scores.shape=labels.shape=[n_obj, ],scores, labels为np格式
            # 此时保留的预测框的confidence都是大于阈值
            bboxes, scores, labels = net(img)
            pred_time = time.time() - t0
            scale = np.array([[w, h, w, h]])
            bboxes *= scale  # 将相对的坐标转换为了绝对的坐标
            # 根据标签将结果放入self.all_boxes中去
            for j in range(len(self.VOC_CLASSES)):
                ind = np.where(labels == j)[0] # 获得第i中图片中第j类的bbox的索引
                if len(ind) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                else:
                    c_bboxes = bboxes[ind] # [n,4]
                    c_scores = scores[ind] # [n, ]
                    c_det = np.concatenate([c_bboxes, c_scores.reshape(-1,1)],axis=1) # [n, 5]
                    self.all_boxes[j][i] = c_det
            if i % 500 == 0:
                print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, pred_time))

        # # 以验证self.all_boxes得到的结果和标准文件是相同的
        # with open(self.det_file,'wb') as f:
        #     pickle.dump(self.all_boxes,f)
        #
        # 第2步将all_box中的内容存储到以类别进行区分的txt文件中(存储的内容应当是图片的标签、置信度、边框的绝对坐标)
        # 第2步验证为正确的
        for cls_ind, cls in enumerate(self.VOC_CLASSES):
            filename = self.class_save_root % (cls) # 确认了文件的名称
            with open(filename,'wt') as f:
                for im_ind, index in enumerate(self.dataset.ids): # 此时index为图片的编号, im_ind代表此时为第几张图片
                    dets = self.all_boxes[cls_ind][im_ind] # 导入第im_ind张图片的第cls_ind类的信息()
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        # 此时的+1后之前VOCAnnotationTransform中int(bbox.find(pt).text) - 1对应
                        # 顺序为图片编号、置信度，坐标的绝对位置
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
        # 第三步:抽取出标签(遍历xml文件),并且进行AP的计算
        aps = []
        for i, cls in enumerate(self.VOC_CLASSES):
            # 取出当前类所有图片的目标信息
            filename = self.class_save_root % (cls) # 通过网络预测为这一类的所有的信息
            r, p, ap = self.voc_eval(detpath=filename,  # 指定路径的txt文件
                                     classname=cls,   # 类别的名称
                                     ovthresh=0.5) # 设置的IOU阈值
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': r, 'prec': p, 'ap': ap}, f)
        self.map = np.mean(aps)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))

    def voc_eval(self,detpath, classname, ovthresh):
        with open(self.image_lst, 'r') as f:
            lines = f.readlines()
        imagenames = [line.strip() for line in lines]
        if not os.path.exists(self.all_object_label):
            recs = {} # 代表所有的验证图片下的所有检测目标的信息，recs中的每一个键值为一个列表(列表中包含当前图片所有的检测目标)
            # 抽取出这一类的标签的真实信息(首先是读取编号，以后是根据编号在xml中1取出信息)
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(self.annopath % (imagename)) # 获得了当前的图片中的所有检测物体
            with open(self.all_object_label, 'wb') as f:
                pickle.dump(recs,f)
        else:
            recs = pickle.load(open(self.all_object_label,'rb'))  # 经过检验其是正确的
        # 根据每个类别的所有预测信息和所有的标签信息计算这个类的AP值

        # 以下是标签信息的求解
        # 获得每一个类别每一张图片下的检测目标
        npos = 0
        class_recs = {} # 用于存储包含当前类的图片的检测目标的信息,每个元素为一个字典，存放当前类别的一个检测目标的所有信息
        for i, imagename in enumerate(imagenames):
            # recs[imagename]代表当前图片中所有类别的检测目标，R的元素为一个字典，存放当前图片的当前类别检测目标的所有信息(列表，元素为字典)
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['box'] for x in R]) # 所有当前类别的检测目标的坐标,shape=[n_obj,4],n_obj代表的当前图片中这个类的检测目标的个数
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool) # shape=[n_obj,]
            det = [False] * len(R) # 用于之后的TP的判定
            npos = npos + sum(~difficult) # 在总的标签框个数中取出了较难进行判断的标签框
            # class_recs[imagename]中存放当前图片下的当前类别的bbox、difficult信息
            class_recs[imagename] = {'bbox':bbox, 'difficult':difficult, 'det':det}  # 已经验证其是正确的
        # 以下是求解预测的信息
        with open(detpath,'r') as f: # 此时得到的是预测的信息
            lines = f.readlines()  # lines存储着所有预测为当前类的检测目标信息
        if any(lines) == 1: # 预测时有这个类的目标
            splitlines = [line.strip().split(' ') for line in lines] # 有的时候应当为一个二维的列表
            image_ids = [x[0] for x in splitlines] # 预测图片的编号，其为列表，[pred_n, ]
            confidence = np.array([float(x[1]) for x in splitlines]) # 预测的置信度[pred_n, ],pred_n代表预测时为这个类的检测目标的数量
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines]) # 预测的坐标信息,[pred_n, 4]

            sorted_ind = np.argsort(-confidence)
            BB = BB[sorted_ind, :] # 按照置信度的顺序对预测的坐标进行重新排序
            image_ids = [image_ids[x] for x in sorted_ind]

            nd = len(image_ids) # 预测出的检测目标的个数
            tp = np.zeros(nd) # 真实的(预测真实，实际真实) 实际真实就是指预测的坐标和真实的坐标之间的IOU大于阈值
            fp = np.zeros(nd) # 假阳性(预测真实，实际为假)
            # 此时对于预测的目标信息存在 这个类别的 图片的标签、置信度、坐标(self.class_save_root的类别的txt文件，之后得到的BB和image_ids)
            # 对于标签信息来说有 这个类别的 图片的标签、坐标、是否较难检测的标志(class_recs)
            for d in range(nd):
                # image_ids[d]代表的意思是 第d个confidence对应的预测bbox是属于那一张图的，R代表这张下的实际的标签框的信息(其中包含bbox,difficult)
                R = class_recs[image_ids[d]]
                # bb代表着第d个confidence对应的预测bbox的坐标
                bb = BB[d,:]
                # 代表着bb这张图片中实际的标签框的坐标
                BBGT = R['bbox']
                if BBGT.shape[0] != 0:
                    # 计算当前预测框和对应的标签框之间的IOU值
                    ixmin = np.maximum(BBGT[:,0], bb[0]) # [n, ]
                    iymin = np.maximum(BBGT[:,1], bb[1]) # [n, ]
                    ixmax = np.minimum(BBGT[:,2], bb[2]) # [n, ]
                    iymax = np.minimum(BBGT[:,3], bb[3]) # [n, ]
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inter = iw * ih # [n, ] 交集
                    union = (bb[2] - bb[0]) * (bb[3] - bb[1]) + (BBGT[:,2] - BBGT[:,0]) * (BBGT[:,3] - BBGT[:,1]) - inter # [n,]
                    overlaps = inter / union
                    ovmax = np.max(overlaps) # 取出最大的交并比和阈值进行比较
                    jmax = np.argmax(overlaps) # 取出适合当前的标签框中哪一个的iou值最大
                    # # ovthresh为最终的交并比阈值(实际为正确的),此时的预测框的confidence都是大于阈值的
                    if ovmax > ovthresh:
                        # 此时的bbox和所在的图片的标签框的iou值大于阈值代表此时即是对应tp
                        # R['difficult'][jmax]代表的是此时的bbox对应的图片的最大iou标签框是否是较难测量的，较难测量的为1
                        if not R['difficult'][jmax]:
                            # 最初的情况下R['det']都是False的
                            if not R['det'][jmax]:
                                tp[d] = 1
                                # 当此时第d个bbox和当前图片中的jmax标签框对应的时候，其他的bbox应该不会与当前图片中的jmax标签框对应(一对一的关系)
                                R['det'][jmax] = 1  # 若不是一对一对应，说明预测出现错误
                            else:
                                fp[d] = 1
                    else: # 此时预测的confidence是大于阈值的，实际的错误的，因此为放在fp
                        fp[d] = 1

            fp = np.cumsum(fp) # fp[i] = sum(fp[:i])
            tp = np.cumsum(tp) # tp.shape=fp.shape=(nd,)

            p = tp / np.maximum(np.finfo(np.float64).eps, tp + fp) # (nd,)
            r = tp / float(npos) # (nd,)
            # 求解这个类别的AP值
            ap = self.voc_ap(r, p)
        else:
            r = -1
            p = -1
            ap = -1
        return r, p, ap

    def voc_ap(self, r, p): # 计算AP的值
        # 此时求解的方式为 p取当前项以及之后项的最大值(记为part1)， 同时删去R重复的项(仅保留p最大的那一项,记为part2)
        # p = [1., 1., 1., 1., 0.8, 0.66, 0.71] r = [0.14, 0.28, 0.42, 0.57, 0.57, 0.57, 0.71]
        # 此时AP计算的为1*(0.14-0)+1*(0.28-0.14)+1*(0.42-0.28)+1*(0.57-0.42)+0.71*(0.71-0.57)=0.6694
        # part1 项的求解
        r = np.concatenate(([0.], r, [1.]))
        p = np.concatenate(([0.], p, [0.]))
        # part2的求解
        max_part2 = r[1:]
        min_part2 = r[:-1]
        index = np.where(max_part2 != min_part2)[0]
        part2 = max_part2[index] - min_part2[index]
        # part1的求解
        for i in range(p.shape[0] - 1, 0, -1):
            p[i - 1] = np.maximum(p[i - 1], p[i])
        part1 = p[index + 1]
        ap = sum(part1 * part2)
        return ap

    def parse_rec(self,image_name):  # 输入一个xml文件(图片的编号)，需要返回这个编号的图片信息
        # 在验证的过程中并没有使用到目标对象的det(判断目标物体是否被遮挡)
        tree = ET.parse(image_name)
        objects = []
        for obj in tree.findall('object'):
            # 此时obj代表的是这张图片中的一个检测目标
            obj_struct = {}
            obj_struct['difficult'] = int(obj.find('difficult').text)
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['name'] = obj.find('name').text
            bbox = obj.find('bndbox')
            obj_struct['box'] = [int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                                 int(bbox.find('xmax').text), int(bbox.find('ymax').text)]
            objects.append(obj_struct)
        return objects # 此时返回一个列表，其元素为图片中的每一个检测目标(字典)










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
