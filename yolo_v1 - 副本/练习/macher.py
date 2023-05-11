# 此文件用于对数据集进行处理
# 输入的内容为一个批量的标签(标签框的相对坐标以及类别)
# 输出每个grid_cell中的置信度(grid_cell中有标签框则为1，无标签框则为0)、标签框的绝对的宽高的对数、
# grid_cell的类别标签、标签框的中心点相对当前grid_cell中的相对坐标, 根据标签框的宽高得到的这个框的权重大小
import numpy as np
import torch

def generate_dxdywh(gt_label, w, h, s): # 用于计算h(log后), w(log后), grid_x, grid_y, weight
    xmin, ymin, xmax, ymax = gt_label[:-1]
    w_fact = (xmax - xmin) * w
    h_fact = (ymax - ymin) * h
    log_w = np.log(w_fact)
    log_h = np.log(h_fact)
    # 求解中心点的坐标
    center_x = (xmin + (xmax - xmin) * 0.5) * w
    center_y = (ymin + (ymax - ymin) * 0.5) * h
    # 中心点所在
    grid_x = center_x / s
    grid_y = center_y / s
    grid_cell_x = int(center_x / s)
    grid_cell_y = int(center_y / s)
    # 相对于当前的grid_cell的相对坐标
    grid_x = grid_x - grid_cell_x
    grid_y = grid_y - grid_cell_y
    # 所求的权重
    weight = 2 - (w_fact / w) * (h_fact / h)
    return grid_cell_x, grid_cell_y, grid_x, grid_y,log_w, log_h, weight

def gt_creator(input_size,stride,label_lists):
    # input_size代表输入图片的大小,stride是为一个grid_cell的大小，label_list为一个三维的列表
    # 返回的应该是[batch, h, w, (1 + 1 + 4 + 1)] -> 置信度，标签值，bbox的宽高的log值，标签框相对于当前grid_cell的坐标，权重weight
    w = input_size
    h = input_size
    h_s = input_size // stride
    w_s = input_size // stride
    batch = len(label_lists)
    gt_tensor = np.zeros(shape=(batch, h_s, w_s, 7))
    for i in range(batch):
        for k, gt_label in enumerate(label_lists[i]):
            obj_class = int(gt_label[-1])
            grid_cell_x, grid_cell_y, grid_x, grid_y, log_w, log_h, weight = generate_dxdywh(gt_label, w, h, stride)
            if grid_cell_x < gt_tensor.shape[2] and grid_cell_y < gt_tensor.shape[1]:
                gt_tensor[i][grid_cell_y][grid_cell_x][0] = 1.
                gt_tensor[i][grid_cell_y][grid_cell_x][1] = obj_class
                gt_tensor[i][grid_cell_y][grid_cell_x][2:6] = np.array([grid_x,grid_y,log_w,log_h])
                gt_tensor[i][grid_cell_y][grid_cell_x][6] = weight
    gt_tensor = gt_tensor.reshape(gt_tensor.shape[0], -1, 7)
    return torch.from_numpy(gt_tensor).float()

if __name__ == '__main__':
    input_size = 10
    stride = 5
    label_lists = [[[0.4,0.3,0.6,0.7,2],[0.2,0.1,0.9,0.5,4]], [[0.5,0.6,0.8,0.8,3]]]
    gt_tensor = gt_creator(input_size, stride, label_lists)
    print(gt_tensor)

