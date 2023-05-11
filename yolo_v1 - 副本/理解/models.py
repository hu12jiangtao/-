# 经过验证得到models也是对的
import numpy as np
from torch import nn
import torch
import torchvision
from torch.nn import functional as F
from copy import deepcopy
from thop import profile

class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()

    def forward(self,x):
        x1 = F.max_pool2d(x, 5, stride=1, padding=2)
        x2 = F.max_pool2d(x, 9, stride=1, padding=4)
        x3 = F.max_pool2d(x, 13, stride=1, padding=6)
        return torch.cat([x,x1,x2,x3],dim=1)

class Conv(nn.Module):
    def __init__(self,c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1,c2,k,stride=s,padding=p,dilation=d,groups=g),
                                  nn.BatchNorm2d(c2),
                                  nn.LeakyReLU(0.1, inplace=True) if act else nn.Identity())

    def forward(self,x):
        return self.conv(x)


class MSEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, target):
        # target.shape=logits.shape=[batch,H*W]
        # grid_cell中不存在标签框中心以及存在标签框中心的置信度损失给予不同的权重
        confidence = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1-1e-4)
        confidence_true = (target == 1.0).float()
        confidence_false = (target == 0.0).float()
        loss_true = 5 * confidence_true * (target - confidence)**2
        loss_false = confidence_false * confidence**2
        loss = loss_false + loss_true
        return loss # [batch, W * H]


def compute_loss(pred_conf,pred_cls,pred_txtytwth,targets):
    batch_size = pred_conf.shape[0]
    # target = [batch, H * W, 7]
    cls_loss_function = nn.CrossEntropyLoss(reduction='none') # 用于类别判断的损失
    conf_loss_function = MSEWithLogitsLoss() # grid_cell中不存在标签框中心以及存在标签框中心的置信度损失
    twth_loss_function = nn.MSELoss(reduction='none') # 标签框和预测框宽高之间的绝对值损失
    # 此时在yolo v1中是使用MSE损失函数的，但是由于MSELoss在最开始的迭代中标签中tx、ty都是小于1，但预测中可能会出现较大的值，造成的损失较大，因此使用BCELoss
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    # 将输入的标签进行拆分
    gt_obj = targets[...,0] #
    gt_cls = targets[...,1].long() # [batch, H * W]
    gt_txty = targets[...,2: 4] # 标签框的中心点相当于当前的grid_cell的位置[batch, H * W, 2]
    gt_twth = targets[...,4: 6] # 标签框的宽高的对数[batch, H * W, 2]
    gt_box_weight = targets[...,6] # 每个grid_cell的权重 [batch, H * W]
    # 对预测的内容进行处理
    pred_conf = pred_conf[:, :, 0] # 此时必须得是pred_conf[:, :, 0](shape=[1,169])，不能是pred_conf(shape=[1,169]),此时会对conf_loss_function有影响
    pred_cls = pred_cls.permute(0,2,1) # [batch, num_cls, H * W]
    pred_txty =pred_txtytwth[...,:2] # [batch, H * W, 2]
    pred_twth = pred_txtytwth[...,2:] # [batch, H * W, 2]
    # 标签的交叉熵损失函数
    cls_loss = cls_loss_function(pred_cls, gt_cls) * gt_obj # [batch, H * W]
    cls_loss = cls_loss.sum() / batch_size # 每张图片的平均类别损失
    # 置信度的损失
    conf_loss = conf_loss_function(pred_conf,gt_obj) # [batch, W * H]
    conf_loss = conf_loss.sum() / batch_size # 每张图片的平均置信度损失
    # 标签框和预测框宽高之间的绝对值损失
    twth_loss = twth_loss_function(pred_twth, gt_twth).sum(-1) * gt_box_weight * gt_obj # [batch,]
    twth_loss = twth_loss.sum() / batch_size
    # 标签框和预测框的中心点坐标之间的损失(在yolo v1中应该是中心坐标的MSELoss,但是此时变为了BCELoss,即用于分类的交叉熵损失)
    txty_loss = txty_loss_function(pred_txty, gt_txty).sum(-1) * gt_box_weight * gt_obj
    txty_loss = txty_loss.sum() / batch_size
    bbox_loss = twth_loss + txty_loss
    # total_loss
    total_loss = cls_loss + conf_loss + bbox_loss
    return conf_loss, cls_loss, bbox_loss,total_loss


class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5):
        super(myYOLO, self).__init__()
        # trainable代表此时是训练阶段还是测试阶段还是训练阶段
        self.trainable = trainable
        self.num_classes = num_classes
        self.stride = 32
        self.device = device
        self.grid_cell = self.create_grid(input_size)  # 网格坐标矩阵
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.input_size = input_size
        # 第一个阶段为backbone(resnet的卷积层的输出特征向量为512)
        self.backbone = torchvision.models.resnet18(pretrained=True)
        self.feat_dim = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # 第二个阶段为spp模块
        self.neck = nn.Sequential(SPP(), Conv(4 * self.feat_dim, self.feat_dim, k=1))
        # 卷积处理
        self.convsets = nn.Sequential(
            Conv(self.feat_dim, self.feat_dim // 2, k=1),
            Conv(self.feat_dim // 2, self.feat_dim, k=3, p=1),
            Conv(self.feat_dim, self.feat_dim // 2, k=1),
            Conv(self.feat_dim // 2, self.feat_dim, k=3, p=1)
        )
        # 对模型进行预测
        # 此时第一个参数是置信度，num_class中存放每个类别的条件概率，4中分别存放着标签框的中心点在这个grid_cell中的相对坐标以及宽高的log值
        self.pred = nn.Conv2d(self.feat_dim,  1 + num_classes + 4, kernel_size=1)
        # 对模型的参数进行初始化操作
        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1:1+self.num_classes], bias_value)

    def create_grid(self, input_size): # 给每个grid_cell分配坐标，例如第一行的第二个grid_cell的坐标应为(1,0)
        # 图片的尺寸
        w, h = input_size, input_size
        ws, hs = w // self.stride, h // self.stride
        # 分别得到所有grid_cell的x、y坐标
        grid_x = torch.arange(ws).unsqueeze(0).repeat((hs,1))
        grid_y = torch.arange(hs).unsqueeze(-1).repeat(1,ws)
        # 将坐标粘贴到一起
        grid_xy = torch.stack([grid_x, grid_y],dim=-1) # [H, W, 2]
        grid_xy = grid_xy.reshape(-1,2).to(self.device)
        return grid_xy  # [H * W, 2]

    def set_grid(self, input_size):
        self.input_size = input_size
        self.grid_cell = self.create_grid(input_size)

    def inference(self, x):
        # 与训练的过程相同首先得到所有的预测框
        feat = self.backbone(x) # [batch, feat_num, H, W]
        feat = self.neck(feat)
        feat = self.convsets(feat)
        pred = self.pred(feat) # [batch, 1 + num_classes + 4, H, W]
        pred = pred.permute(0,2,3,1) # [batch, H, W, 1 + num_classes + 4]
        pred = pred.reshape(pred.shape[0], -1, pred.shape[-1]) # [batch, H * W, 1 + num_classes + 4]
        # 分别提取出预测框中的内容
        conf_pred = pred[...,:1] # [batch, H * W, 1] # 进行sigmoid归一化后就是置信度
        cls_pred = pred[...,1: 1 + self.num_classes] # [batch, H * W, 20]
        txtytwth_pred = pred[...,1 + self.num_classes:] # [batch, H * W, 4] ，
        # 默认每次输入一张图片
        conf_pred = conf_pred[0] # [H * W, 1]
        cls_pred = cls_pred[0] # [H * W, 20]
        txtytwth_pred = txtytwth_pred[0] # [H * W,4] 此时其中的tx、ty经过sigmoid归一化后才是真正的预测框的中心点坐标相对于所在grid_cell的坐标
        # 获得所有的grid_cell的所有类别出现的概率
        score = torch.sigmoid(conf_pred) * torch.softmax(cls_pred,dim=-1)  # score检测正确
        # 求解边框的实际的边框的位置
        # [H * W, 4],其中存储着所有grid_cell的对应的bbox的左上、右下的相对坐标
        bboxes = self.decode_boxes(txtytwth_pred) / self.input_size
        bboxes = torch.clamp(bboxes, 0., 1.)  # bbox检测正确
        # 进行边框的后处理操作
        score = score.detach().to('cpu').numpy()
        bboxes = bboxes.detach().to('cpu').numpy()
        bboxes, scores, labels = self.postprocess(bboxes, score) # 经过mns筛选所剩下的bbox的信息
        return bboxes, scores, labels

    def postprocess(self,bbox, score):
        # bbox.shape=[H*W, 4], score=[H*W, num_classes]
        # 取出每个bbox最大可能的标签
        labels = np.argmax(score,axis=-1) # [H*W, ]
        # 取出当前标签最大可能的概率
        score = score[np.arange(labels.shape[0]), labels] # [H*W, ]
        # 取出大于设定的置信度阈值的bbox以及其置信度和标前
        keep = np.where(score >= self.conf_thresh)
        labels = labels[keep]
        score = score[keep]
        bbox = bbox[keep]
        # 对于每个类别进行非极大值抑制操作
        keep = np.zeros((len(bbox),),dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]  # 确认当前这个类别i在bbox中的索引
            if len(inds) == 0: # 符合要求的bbox中不存在标签为i的bbox
                continue
            c_score = score[inds] # 验证可以得到两者是相同的
            c_bbox = bbox[inds] # 验证可以得到两者是相同的
            # 每个类别之间进行极大值的抑制
            c_keep = self.nms(c_bbox, c_score)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0) # 得到此时保留的索引
        labels = labels[keep]
        score = score[keep]
        bbox = bbox[keep]
        return bbox, score, labels

    def nms(self, c_bbox, c_scores):
        # c_bbox.shape=[n,4],c_scores.shape=[n, ]
        # 首先根据c_scores的大小给c_bbox进行排序，之后从头开始遍历c_bbox的预测框，剔除与当前预测狂iou大于阈值的c_bbox
        x_1 = c_bbox[:, 0] # [n, ] 左上角的宽
        y_1 = c_bbox[:, 1] # [n, ] 左上角的高
        x_2 = c_bbox[:, 2] # 右下角的宽
        y_2 = c_bbox[:, 3] # 右下角的高
        area = (x_2 - x_1) * (y_2 - y_1) # 所有选中的bbox的面积求解(用于求解IOU的)
        order = np.argsort(c_scores)[::-1] # 从大到小得到最大的分数所在的索引  # order的检测是正确的
        keep = []
        while order.shape[0] > 0:
            i = order[0] # 获取当前confidence最大的预测框的编号
            keep.append(i) # 所保留的序列
            # 计算当前的bbox与其他的bbox之间的iou值
            # 计算两者的交集(以下以最开始的情况为例)
            xx1 = np.maximum(x_1[i], x_1[order[1:]]) # [n-1, ]
            yy1 = np.maximum(y_1[i], y_1[order[1:]]) # [n-1, ]
            xx2 = np.minimum(x_2[i], x_2[order[1:]]) # [n-1, ]
            yy2 = np.minimum(y_2[i], y_2[order[1:]]) # [n-1, ]

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h # 交集 [n-1, ]
            union = area[i] + area[order[1:]] - inter # 并集
            iou = inter / union # [n-1, ]
            # 除去超过 iou 阈值的bbox
            idx = np.where(iou <= self.nms_thresh)[0] # 此时的编号除去了第一个最大值
            order = order[idx + 1]
        return keep # keep中存放着保留的bbox的边框索引

    def decode_boxes(self, pred): #
        # 得到预测的bbox的中心点相对于整张图片的相对坐标
        outputs = torch.zeros_like(pred)
        pred[...,:2] = torch.sigmoid(pred[...,:2]) + self.grid_cell
        # 取得bbox的宽高的绝对长度
        pred[...,2:] = torch.exp(pred[...,2:])
        # 获得bbox的绝对坐标位置
        outputs[...,:2] = pred[...,:2] * self.stride - pred[...,2:] * 0.5
        outputs[...,2:] = pred[...,:2] * self.stride + pred[...,2:] * 0.5
        return outputs

    def forward(self,x, targets=None):
        if self.trainable:  # 训练过程已经经过检测，其时正确的
            feat = self.backbone(x) # [batch, feat_num, H, W], H=W=13,说明一共有 H * W个grid_cell，与target对应
            feat = self.neck(feat)
            feat = self.convsets(feat)
            pred = self.pred(feat) # [batch, 1 + num_classes + 4, H, W]
            pred = pred.permute(0,2,3,1) # [batch, H, W, 1 + num_classes + 4]
            pred = pred.reshape(pred.shape[0],-1,pred.shape[-1]) # [batch, H * W, 1 + num_classes + 4]
            # 根据x, targets计算当前的损失
            # 将置信度、类别，标签框的位置信息 从pred中提取出来
            conf_pred = pred[...,:1] # [batch, H*W, 1],此时需要对其进行归一化操作
            cls_pred = pred[...,1: 1 + self.num_classes] # [batch, H * W, num_classes]
            txtytwth_pred = pred[..., 1 + self.num_classes:] # [batch, H * W, 4]
            # 计算损失
            conf_loss,cls_loss,bbox_loss,total_loss= compute_loss(pred_conf=conf_pred,pred_cls=cls_pred,
                                                                  pred_txtytwth=txtytwth_pred,targets=targets)
            return conf_loss,cls_loss,bbox_loss,total_loss
        else:  # 验证集经过训练
            bbox, score, labels = self.inference(x)
            # 由于在验证的过程中输入的是一张图片，因此bbox.shape=[n_obj,4],score.shape=[n_obj,]
            return bbox, score, labels

if __name__ == '__main__':
    # # 测试集进行验证
    # torch.manual_seed(1)
    # device = torch.device('cuda')
    # model = myYOLO(device, 416, 20, trainable=False).to(device)  # 输入的大小为416
    # model.eval()
    # x = torch.randn(size=(1, 3, 416, 416), device=device)
    # bbox, score, labels = model(x)
    # print(bbox)
    # print(score)
    # print(labels)

    import macher
    torch.manual_seed(1)
    device = torch.device('cuda')
    model = myYOLO(device, 64, 20, trainable=True).to(device)  # 输入的大小为416
    x = torch.randn(size=(2, 3, 64, 64), device=device)


    label_lists = [[[0.4,0.3,0.6,0.7,2],[0.2,0.1,0.9,0.5,4]], [[0.5,0.6,0.8,0.8,3]]]
    gt_tensor = macher.gt_creator(input_size=64, stride=32, label_lists=label_lists)
    gt_tensor = gt_tensor.to(device)

    conf_loss, cls_loss, bbox_loss, total_loss = model(x, gt_tensor)

    print(conf_loss)
    print(cls_loss)
    print(bbox_loss)
    print(total_loss)


