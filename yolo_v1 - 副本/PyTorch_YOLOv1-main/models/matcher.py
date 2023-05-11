import torch
import numpy as np


def generate_dxdywh(gt_label, w, h, s):
    xmin, ymin, xmax, ymax = gt_label[:-1]
    # 计算边界框的中心点
    c_x = (xmax + xmin) / 2 * w # 中心点的宽
    c_y = (ymax + ymin) / 2 * h # 中心点的高
    box_w = (xmax - xmin) * w # 宽
    box_h = (ymax - ymin) * h # 高

    if box_w < 1e-4 or box_h < 1e-4:
        # print('Not a valid data !!!')
        return False    

    # 计算中心点所在的网格坐标
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # 计算中心点偏移量和宽高的标签
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w)
    th = np.log(box_h)
    # 计算边界框位置参数的损失权重
    weight = 2.0 - (box_w / w) * (box_h / h) # 标签框占整张图片的多少比例

    return grid_x, grid_y, tx, ty, tw, th, weight


def gt_creator(input_size, stride, label_lists=[]):
    # model.stride = 32, label_lists为一个三维的列表, input_size=416
    # 必要的参数
    batch_size = len(label_lists)
    w = input_size # 416
    h = input_size # 416
    ws = w // stride # 13
    hs = h // stride # 13
    s = stride # 32
    gt_tensor = np.zeros([batch_size, hs, ws, 1+1+4+1])

    # 制作训练标签
    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]: # 得到第batch_index张图片的边框和标签信息(边框中的信息是相对位置坐标)
            gt_class = int(gt_label[-1]) # 得到这张图片的标签
            result = generate_dxdywh(gt_label, w, h, s)
            if result:
                # grid_x, grid_y指明了这个标签框所在的grid_cell是当前的第几行第几列
                # tx, ty代表标签框的中心点在这个grid_cell中的相对的位置
                # tw, th代表标签框的真实的宽高的对数值(原来取得开根号，现在改为了取对数，目标是为了使模型对小框更加的敏感)
                grid_x, grid_y, tx, ty, tw, th, weight = result

                if grid_x < gt_tensor.shape[2] and grid_y < gt_tensor.shape[1]:
                    gt_tensor[batch_index, grid_y, grid_x, 0] = 1.0
                    gt_tensor[batch_index, grid_y, grid_x, 1] = gt_class
                    gt_tensor[batch_index, grid_y, grid_x, 2:6] = np.array([tx, ty, tw, th])
                    gt_tensor[batch_index, grid_y, grid_x, 6] = weight
                # 此时的grid_x中第一个维度代表第i张图片，第二个和第三个维度表示grid_cell的位置，第四个维度中存储着标签框的相对信息
                # 针对于grid_cell中不存在标签框的内容其值都为0


    gt_tensor = gt_tensor.reshape(batch_size, -1, 1+1+4+1) # grid_cell从左到右从上到下依次进行排列

    return torch.from_numpy(gt_tensor).float()


if __name__ == '__main__':
    input_size = 10
    stride = 5
    label_lists = [[[0.4,0.3,0.6,0.7,2],[0.2,0.1,0.9,0.5,4]], [[0.5,0.6,0.8,0.8,3]]]
    gt_tensor = gt_creator(input_size, stride, label_lists)
    print(gt_tensor)