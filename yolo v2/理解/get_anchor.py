# 此文件用于获取anchor尺寸选取的大小
# 其主要的原理是利用k-means算法进行聚类

# 首先获取数据集中所有检测目标的标签框的宽高
# 其次在其中随机抽取k个标签框的宽高作为初始的簇，并且计算出所有的标签框和选中的簇之间的衡量距离(1-iou)
# 此时的iou的计算为 取簇与标签框两者之间最小的宽高相乘就是相交的面积
from xml.etree import ElementTree as ET
import numpy as np
import os
import pickle

def load_dataset(root_path): # 将所有的检测目标的标签框的宽高放在一个列表中
    listdir = os.listdir(root_path)
    objs = []
    for cur_dir in listdir:
        cur_listdir = os.path.join(root_path, cur_dir)
        tree = ET.parse(cur_listdir)
        for obj in tree.findall('object'):
            box = obj.find('bndbox')
            one_hw = [int(box.find('xmax').text) - int(box.find('xmin').text),
                      int(box.find('ymax').text) - int(box.find('ymin').text)]
            objs.append(one_hw)
    return objs


def iou(box, clusters):
    min_w = np.minimum(box[0], clusters[:,0])
    min_h = np.minimum(box[1], clusters[:,1])
    inter = min_h * min_w
    union = box[0] * box[1] + clusters[:,0] * clusters[:,1] - inter
    iou = inter / union
    return iou


def k_means(boxes, k, dist=np.median):  # np.median的目标是选取中位数
    # boxes = [n_obj, 2],其中存储的内容为检测物体的宽高
    n_obj= boxes.shape[0]
    distance = np.empty(shape=(n_obj, k))
    np.random.seed()
    clusters = boxes[np.random.choice(n_obj, k, replace=False)] # 不放回的随机抽取作为初始的簇心 [k, 2]
    last_clusters = np.zeros((n_obj,)) # 存储当前box属于哪一个类的
    while True:
        for row in range(n_obj):
            distance[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distance,axis=1) # 利用指标(1-iou)将所有的box归类于指标最小的簇上
        if all(nearest_clusters == last_clusters): # 当前的分类和上一次的分类相同的时候则跳出循环
            break
        for i in range(k):  # 更新每一个簇的宽高(此时利用的是中位数)
            clusters[i] = dist(boxes[nearest_clusters == i],axis=0)
        last_clusters = nearest_clusters
    return clusters

if __name__ == '__main__':
    root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit\\VOC2007\\Annotations'
    boxes = load_dataset(root)
    boxes = np.array(boxes)
    clusters = k_means(boxes, k=9, dist=np.mean)
    print(clusters)
