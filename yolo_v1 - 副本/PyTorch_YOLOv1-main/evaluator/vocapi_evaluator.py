import torch
from torch.autograd import Variable
from data.voc0712 import VOCDetection, VOC_CLASSES
import os
import time
import numpy as np
import pickle

import xml.etree.ElementTree as ET


class VOCAPIEvaluator():
    """ VOC AP Evaluation class """
    def __init__(self, data_root, img_size, device, transform, set_type='val', year='2007', display=False):
        self.data_root = data_root  # 'VOCdevkit'
        self.img_size = img_size
        self.device = device
        self.transform = transform
        self.labelmap = VOC_CLASSES  # 为所有类别的名称
        self.set_type = set_type
        self.year = year
        self.display = display

        # path
        self.devkit_path = data_root + 'VOC' + year  #
        self.annopath = os.path.join(data_root, 'VOC2007', 'Annotations', '%s.xml')
        self.imgpath = os.path.join(data_root, 'VOC2007', 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', set_type+'.txt')
        self.output_dir = self.get_output_dir('voc_eval/', self.set_type) # 此时的路径为'evaluator/voc_eval/val'
        print('devkit_path:',self.devkit_path)
        print('output_dir:',self.output_dir)

        # dataset
        self.dataset = VOCDetection(root=data_root, 
                                    image_sets=[('2007', set_type)],
                                    transform=transform
                                    )

    def evaluate(self, net):
        net.eval()
        num_images = len(self.dataset) # 得到验证集的样本个数
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        # 此时的self.all_boxes[j][i]中存储着第j个类别的第i张图片的置信度和边框的位置
        self.all_boxes = [[[] for _ in range(num_images)]
                        for _ in range(len(self.labelmap))]

        # timers
        # det_file用于存储all_boxes(预测的all_box为一个二维的列表，all_box[j][i]代表着第j类第i张图片存储的bbox的坐标和置信度)
        det_file = os.path.join(self.output_dir, 'detections.pkl')
        print(det_file)
        if not os.path.exists(det_file):
            for i in range(num_images):  # 遍历每一张图片
                # 将样本进行处理后输出 经过数据增强的图片和标签(label.shape=[n_obj,5],前四个为左上右下角的相对坐标)
                im, gt, h, w = self.dataset.pull_item(i)
                x = im.unsqueeze(0).to(self.device)
                t0 = time.time() # 经过次数是正确的
                # forward
                bboxes, scores, labels = net(x) # 预算出预测且经过筛选后保留的锚框的置信度、左上右下的相对坐标，以及预测的标签
                detect_time = time.time() - t0 # 得到预测一张图片所需要的时间
                scale = np.array([[w, h, w, h]])
                bboxes *= scale # 得到bbox的左上角右下角的绝对位置坐标
                for j in range(len(self.labelmap)):
                    inds = np.where(labels == j)[0] # 判断第i张图片上预测的bbox中是否含有j类，若有则给出这个bbox的索引
                    if len(inds) == 0:
                        self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                        continue
                    c_bboxes = bboxes[inds] # 假设i张图片上预测的bbox中j类含有n个，c_bboxes.shape=[n,4]
                    c_scores = scores[inds] # c_scores.shape=[n,]
                    c_dets = np.hstack((c_bboxes,
                                        c_scores[:, np.newaxis])).astype(np.float32,
                                                                        copy=False) # c_scores[:, np.newaxis]相当于将行向量拉成列向量
                    self.all_boxes[j][i] = c_dets
                if i % 500 == 0: # 给出一张图片的预测速度，且500轮显示一次
                    print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1, num_images, detect_time))
            # with open(det_file, 'wb') as f: # 对all_boxes进行存储
            #     pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)
        else:
            self.all_boxes = pickle.load(open(det_file,'rb'))

        print('Evaluating detections')
        self.evaluate_detections(self.all_boxes) # 对其MAP的指标进行求解

  

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text) # 用于判断检测的物体是否有被遮挡(只为目标整体的一部分)
            obj_struct['difficult'] = int(obj.find('difficult').text)  # 用于判断检测的困难程度
            bbox = obj.find('bndbox')
            # 此时保留的是绝对坐标
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
            objects.append(obj_struct)

        return objects


    def get_output_dir(self, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir


    def get_voc_results_file_template(self, cls): # cls代表的是类别的名称
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + self.set_type + '_%s.txt' % (cls) # det_val_类别名称.txt文件
        filedir = os.path.join(self.devkit_path, 'results') # /VOCdevkitVOC2007/results
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes):
        # all_boxes = self.all_boxes, self.all_boxes[i][j]代表的意思为第i类的第j张图片的 bbox坐标和置信度
        for cls_ind, cls in enumerate(self.labelmap):
            if self.display:
                print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(cls)  # 此时为最开始的创建的新的txt文件
            with open(filename, 'wt') as f: # wt代表向其中不断写入内容
                # self.dataset.ids为一个列表其中的元素为一个元组,元组中的内容图片的根目录VOCdevkit/VOC2007和一张图片的编号
                for im_ind, index in enumerate(self.dataset.ids):
                    dets = all_boxes[cls_ind][im_ind]  # 得到第j类的第im_ind张图片的 bbox坐标和置信度
                    if dets == []: # 说明第im_ind张图片中不存在第j类图片
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]): # dets.shape = [n, 5],假设第im_ind张图片中存在第j类图片目标为n
                        # 往txt文件中写入 验证图片的编号 + 置信度 + bbox的绝对坐标
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))


    def do_python_eval(self, use_07=True):
        cachedir = os.path.join(self.devkit_path, 'annotations_cache') # VOCdevkitVOC2007/annotations_cache,一个空的路径
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        for i, cls in enumerate(self.labelmap):  # self.labelmap为所有类别的名称
            # 得到一个指定路径的txt文件(文件中已经存在内容了，在write_voc_results_file中已经写入了内容)
            filename = self.get_voc_results_file_template(cls)
            rec, prec, ap = self.voc_eval(detpath=filename,  # 指定路径的txt文件
                                          classname=cls,   # 类别的名称
                                          cachedir=cachedir,  # 一个空路径
                                          ovthresh=0.5,
                                          use_07_metric=use_07_metric # 默认为True
                                        )
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(self.output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        if self.display:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('Results:')
            for ap in aps:
                print('{:.3f}'.format(ap))
            print('{:.3f}'.format(np.mean(aps)))
            print('~~~~~~~~')
            print('')
            print('--------------------------------------------------------------')
            print('Results computed with the **unofficial** Python eval code.')
            print('Results should be very close to the official MATLAB eval code.')
            print('--------------------------------------------------------------')
        else:
            self.map = np.mean(aps)
            print('Mean AP = {:.4f}'.format(np.mean(aps)))


    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def voc_eval(self, detpath, classname, cachedir, ovthresh=0.5, use_07_metric=True):
        # detpath为一个指定路径的txt文件夹(指定着当前的类别)，classname为当前类别的名称，cachedir为一个空的路径
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl') # 获得了一个指定路径的存储文件
        # read list of images
        with open(self.imgsetpath, 'r') as f: # self.imgsetpath为存放验证图片的序号的txt文件
            lines = f.readlines() #
        imagenames = [x.strip() for x in lines] # 获得验证图片编号的列表
        if not os.path.isfile(cachefile):  # 用于判断是否存在指定的文件
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                # self.annopath % (imagename) = Annotations\imagename.xml xml文件中存储着这个标签图片的所有的绝对坐标和类别标签
                recs[imagename] = self.parse_rec(self.annopath % (imagename))
                if i % 100 == 0 and self.display:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            if self.display:
                print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)
        # extract gt objects for this class
        # recs是一个字典，其中的每对键名为图片的命名编号，键值为一个列表，列表中的每一个元素为一个字典，其中包含了图片的一个检测目标的相关信息
        class_recs = {}
        npos = 0
        for imagename in imagenames: # 给出当前图片的命名编号
            # 遍历当前标号的检测目标，并且保留名称为当前选中的类别的检测目标，此时R为一个列表，元素为一个字典
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R]) # 得到这张图片的这个类别的绝对坐标
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool) # 得到这张图片的这个类别的目标是否是较难检测的
            det = [False] * len(R)
            npos = npos + sum(~difficult) # sum(~difficult)代表这张图的这个类别的较好检测的目标的个数
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}
        # class_recs中存放着指定的类别，不同的图片中含有这个类别的相关信息(是否好检测以及标签框的坐标)
        # read dets
        detfile = detpath.format(classname) # 确保这个txt文件与当前这个类别相对应
        with open(detfile, 'r') as f:
            lines = f.readlines()  # 读取其中的内容信息
        if any(lines) == 1:
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1 #
                        else:
                            fp[d] = 1. # 预测出错
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    def evaluate_detections(self, box_list):
        self.write_voc_results_file(box_list)
        self.do_python_eval()


if __name__ == '__main__':
    from data import transform
    from models import build
    from models import yolo

    torch.manual_seed(1)
    val_size = 416
    data_root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
    device = torch.device('cuda')
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    val_transform = transform.BaseTransform(val_size, pixel_mean, pixel_std)
    model = yolo.myYOLO(device, 416, 20, trainable=False).to(device)
    model.eval()
    model.set_grid(val_size)
    evaluator = VOCAPIEvaluator(
        data_root=data_root,
        img_size=val_size,
        device=device,
        transform=val_transform
    )
    evaluator.evaluate(model)
