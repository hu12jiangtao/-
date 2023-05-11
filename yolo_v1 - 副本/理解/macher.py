# 此文件用于对数据集进行处理
# 输入的内容为一个批量的标签(标签框的相对坐标以及类别)
# 输出每个grid_cell中的置信度(grid_cell中有标签框则为1，无标签框则为0)、标签框的绝对的宽高的对数、
# grid_cell的类别标签、标签框的中心点相对当前grid_cell中的相对坐标, 根据标签框的宽高得到的这个框的权重大小
import numpy as np
import torch

def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    c_x = (xmin + xmax) / 2 * w
    c_y = (ymin + ymax) / 2 * h
    box_x = (xmax - xmin) * w
    box_y = (ymax - ymin) * h
    # 获得中心点的相对坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    # 获得这个标签框的中心点是那个grid_cell
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 中心点相对于grid_cell的相对宽高
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    # 获得标签框的绝对宽高的对数
    tw = np.log(box_x)
    th = np.log(box_y)
    weight = 2 - (box_x / w) * (box_y / h)
    return grid_x,grid_y,tx,ty,tw,th,weight



def gt_creator(input_size,stride,label_lists):
    # input_size代表输入图片的大小,stride是为一个grid_cell的大小，label_list为一个三维的列表
    # 返回的应该是[batch, h, w, (1 + 1 + 4 + 1)]
    batch_size = len(label_lists)
    h = input_size
    w = input_size
    s = stride
    hs = input_size // stride
    ws = input_size // stride
    gt_tensor = np.zeros((batch_size, hs, ws, 7))
    for batch_index in range(batch_size):
        for k,gt_label in enumerate(label_lists[batch_index]): # gt_label为第gt_tensor张图片中的k个标签框
            # 获取这个标签框的1 + 1 + 4 + 1
            gt_class = int(gt_label[-1]) # 获得这个标签框的类别
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight = result
                # 将求解的内容放入gt_tensor中去
                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index,grid_y,grid_x, 0] = 1.0 # 说明这个置信区间的置信度为1
                    gt_tensor[batch_index,grid_y,grid_x, 1] = gt_class # 这个grid_cell的标签
                    gt_tensor[batch_index,grid_y,grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index,grid_y,grid_x, -1] = weight
    gt_tensor = gt_tensor.reshape(batch_size, -1, 7)
    return torch.from_numpy(gt_tensor).float()


if __name__ == '__main__':
    input_size = 10
    stride = 5
    label_lists = [[[0.4,0.3,0.6,0.7,2],[0.2,0.1,0.9,0.5,4]], [[0.5,0.6,0.8,0.8,3]]]
    gt_tensor = gt_creator(input_size, stride, label_lists)
    print(gt_tensor)

