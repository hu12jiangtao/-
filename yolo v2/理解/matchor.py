# 此时需要对标签进行处理 [xmin, ymin, xmax, ymax, label],且其中的都是相对坐标
# 此时对于一个batch=3的target来说为[[n_obj1, 5], [n_obj2, 5], [n_obj3, 5]]
# 在yolo v1 中 需要输出的内容gt_tensor[batch_idx][grid_x][grid_y] 为 置信度(非0即1) 标签 中心点的坐标 宽高 标签框的权重
# 在yolo v2 中 获取标签的方式和yolo v1不相同
import numpy as np
import torch

def set_anchors(anchor_size):
    num_anchor = len(anchor_size)
    anchor_array = np.zeros(shape=(num_anchor,4))
    for index, size in enumerate(anchor_size):
        ws, hs = size
        anchor_array[index] = np.array([0,0,ws,hs])
    return anchor_array

def compute_iou(anchor_boxes, gt_box):
    # 计算两个boxes之间的iou
    # 求解anchor的左上角和右下角的坐标
    ab_x1y1_x2y2 = np.zeros([len(anchor_boxes), 4]) # 存储左上角和右下角的坐标
    ab_x1y1_x2y2[:,0] = anchor_boxes[:,0] - anchor_boxes[:,2] * 0.5
    ab_x1y1_x2y2[:,1] = anchor_boxes[:,1] - anchor_boxes[:,3] * 0.5
    ab_x1y1_x2y2[:,2] = anchor_boxes[:,0] + anchor_boxes[:,2] * 0.5
    ab_x1y1_x2y2[:,3] = anchor_boxes[:,1] + anchor_boxes[:,3] * 0.5
    w_ab, h_ab = anchor_boxes[:,2], anchor_boxes[:,3] # anchor的宽高
    # 求解标签框的左上角和右下角的坐标
    gt_x1y1_x2y2 = np.zeros(shape=[len(anchor_boxes), 4])
    gt_box_expand = np.repeat(gt_box, len(anchor_boxes), axis=0)
    gt_x1y1_x2y2[:,0] = gt_box_expand[:,0] - gt_box_expand[:,2] * 0.5
    gt_x1y1_x2y2[:,1] = gt_box_expand[:,1] - gt_box_expand[:,3] * 0.5
    gt_x1y1_x2y2[:,2] = gt_box_expand[:,0] + gt_box_expand[:,2] * 0.5
    gt_x1y1_x2y2[:,3] = gt_box_expand[:,1] + gt_box_expand[:,3] * 0.5
    w_gt, h_gt = gt_box[:,2], gt_box[:,3] # 标签框的宽高
    # 求解公共部分的面积
    inter_Lx = np.minimum(ab_x1y1_x2y2[:,2], gt_x1y1_x2y2[:,2])
    inter_Sx = np.maximum(ab_x1y1_x2y2[:,0], gt_x1y1_x2y2[:,0])
    inter_Ly = np.minimum(ab_x1y1_x2y2[:,3], gt_x1y1_x2y2[:,3])
    inter_Sy = np.maximum(ab_x1y1_x2y2[:,1], gt_x1y1_x2y2[:,1])
    inter = (inter_Lx - inter_Sx) * (inter_Ly - inter_Sy)
    union = w_ab * h_ab + w_gt * h_gt - inter
    iou = inter / union
    return iou


def generate_txtytwth(gt_label, w, h, s, anchor_size, ignore_thresh):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算出中心点绝对坐标和宽高
    c_x = (xmax + xmin) / 2 * w
    c_y = (ymax + ymin) / 2 * h
    box_w = (xmax - xmin) * w  # 此时求解出的内容可能是负数
    box_h = (ymax - ymin) * h
    # 求解出中心点和宽高相对于grid_cell尺寸的相对坐标
    if box_w < 1e-4 or box_h < 1e-4:  # 这个if语句是不能丢弃的
        print('not a valid data !!!')
        return False
    c_x_s = c_x / s
    c_y_s = c_y / s
    box_ws = box_w / s
    box_hs = box_h / s
    # 求解出所在的grid_cell的位置
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 将anchor点的中心记为(0,0),此时计算anchor和标签框之间的IOU
    anchor_boxes = set_anchors(anchor_size)  # 其中一项应当为[0,0,anchor_w,anchor_h],得到结果[num_anchor,4]
    gt_box = np.array([[0, 0,box_ws, box_hs]])
    # 计算anchor和标签框之间的iou
    iou = compute_iou(anchor_boxes, gt_box)
    # 只保留iou大于阈值ignore_thresh的anchor
    iou_mask = (iou > ignore_thresh)
    result = []
    if iou_mask.sum() == 0:
        # 如果所有的anchor和标签框的IOU都小于设定的阈值的情况下选择阈值最大的anchor
        index = np.argmax(iou)
        # 取出这个anchor的宽高
        p_w, p_h = anchor_size[index]
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_ws / p_w)
        th = np.log(box_hs / p_h)
        weight = 2. - (box_w / w) * (box_h / h)
        result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
    else:
        best_index = np.argmax(iou)
        for index, i in enumerate(iou_mask):
            if i: # 大于阈值的情况分类讨论（此时只能是i == True，不可以是i is True）is语句要求两者的内容和id必须都需要相同
                if index == best_index: # 最大的iou的情况
                    p_w, p_h = anchor_size[index]
                    tx = c_x_s - grid_x
                    ty = c_y_s - grid_y
                    tw = np.log(box_ws / p_w)
                    th = np.log(box_hs / p_h)
                    weight = 2. - (box_w / w) * (box_h / h)
                    result.append([index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax])
                else:
                    result.append([index, grid_x, grid_y, 0., 0., 0., 0., -1., 0., 0., 0., 0.])
    return result


def gt_creator(input_size, stride, label_lists, anchor_size, ignore_thresh):
    batch_size = len(label_lists)
    w, h = input_size, input_size
    s = stride
    ws, hs = w // stride, h // stride
    anchor_num = len(anchor_size)
    # 针对当前的grid_cell中不包含标签框中心点或者 标签框和anchor的iou小于阈值的情况时 置信度和权重都为0
    gt_tensor = np.zeros(shape=(batch_size, hs, ws, anchor_num, 1 + 1 + 4 + 1 + 4))
    for batch_idx in range(batch_size):
        for gt_label in label_lists[batch_idx]:  # gt_label = [5,] -> [xmin, ymin, xmax, ymax, label]相对坐标
            gt_class = int(gt_label[-1])
            results = generate_txtytwth(gt_label, w, h, s, anchor_size, ignore_thresh)
            if results:
                for result in results:
                    index, grid_x, grid_y, tx, ty, tw, th, weight, xmin, ymin, xmax, ymax = result
                    if weight > 0: # 针对和标签框的iou最大的anchor来说置信度和权重都是正数
                        gt_tensor[batch_idx][grid_y][grid_x][index][0] = 1. # 置信度为1
                        gt_tensor[batch_idx][grid_y][grid_x][index][1] = gt_class
                        gt_tensor[batch_idx][grid_y][grid_x][index][2:6] = np.array([tx, ty, tw, th])
                        gt_tensor[batch_idx][grid_y][grid_x][index][6] = weight
                        gt_tensor[batch_idx][grid_y][grid_x][index][7:] = np.array([xmin, ymin, xmax, ymax])
                    else:  # 针对于那些和标签框的iou大于阈值但是不是最大的iou的anchor来说将置信度和权重都设置为1
                        gt_tensor[batch_idx][grid_y][grid_x][index][0] = -1. # 置信度为-1
                        gt_tensor[batch_idx][grid_y][grid_x][index][6] = -1 # 权重也设置为-1
    gt_tensor = gt_tensor.reshape(batch_size, -1, 1 + 1 + 4 + 1 + 4)
    return gt_tensor


if __name__ == '__main__':
    input_size = 10
    stride = 5
    label_lists = [[[0.4,0.3,0.6,0.7,2],[0.2,0.1,0.9,0.5,4]], [[0.5,0.6,0.8,0.8,3]]]
    ignore_thresh = 0.6
    anchor_size = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    gt_tensor = gt_creator(input_size, stride, label_lists, anchor_size, ignore_thresh)
    print(gt_tensor)


