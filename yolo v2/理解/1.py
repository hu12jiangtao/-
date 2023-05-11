import pickle

import numpy as np
from torch import nn
import torch
from darknet import build_darknet19

class Conv(nn.Module):
    def __init__(self, in_channel, out_channel, k=1, p=0, s=1, d=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=k,padding=p,stride=s,dilation=d,bias=False),
                                  nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.1,inplace=True) if act else nn.Identity())
    def forward(self,x):
        return self.conv(x)

class reorg_layer(nn.Module):
    def __init__(self,stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self,x): # x = [b, 64, h/16, w/16] -> out = [b, 64 * 4, h/32, w/32]
        b, c, h, w = x.shape
        hs = h // self.stride
        ws = w // self.stride
        x = x.reshape(b, c, hs, self.stride, ws, self.stride).transpose(3,4) # [b, c, hs, ws, self.stride * self.stride]
        x = x.reshape(b, c, hs * ws, -1).transpose(2,3) # [b, c, self.stride * self.stride, hs * ws]
        x = x.reshape(b, c, -1, hs, ws).transpose(1,2)
        x = x.reshape(b, -1, hs, ws)
        return x

def iou_score(box1, box2):
    # box1 = box2 = [b * hs * ws * num_anchors, 4]
    area1 = torch.prod(box1[:, 2:] - box1[:, :2], dim=1) # [b * hs * ws * num_anchors, ]
    area2 = torch.prod(box2[:, 2:] - box2[:, :2], dim=1) # [b * hs * ws * num_anchors, ]
    tl = torch.max(box1[:,:2], box2[:,:2])
    br = torch.min(box1[:,2:], box2[:,2:])
    flag = (tl < br).type(tl.dtype).prod(dim=-1)
    inter = flag * torch.prod(br - tl, dim=1) # [b * hs * ws * num_anchors, ]
    union = area2 + area1 - inter
    iou = inter / union
    return iou


class YOLOv2(nn.Module):
    def __init__(self,device,input_size=416,num_classes=20,trainable=False,conf_thresh=0.001, nms_thresh=0.6,
                 topk=100,anchor_size=None):
        super(YOLOv2, self).__init__()

        self.stride = 32
        self.reorg_dim = 64
        self.trainable = trainable
        self.num_classes = num_classes
        self.device = device
        self.input_size = input_size
        self.topk = topk
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        # 对于anchor的初始化
        self.num_anchors = len(anchor_size)
        self.anchor_size = torch.tensor(anchor_size)
        self.anchor_boxes = self.create_grid(input_size)
        # 输入骨干网络
        self.backbone,self.feat_dims = build_darknet19(pretrained=True)
        self.head_dim = self.feat_dims[-1]
        # c5的处理:通过检测头
        self.convsets_1 = nn.Sequential(Conv(self.head_dim, self.head_dim, k=3, p=1),
                                        Conv(self.head_dim, self.head_dim, k=3, p=1))
        # c4的处理
        self.route_layer = Conv(self.feat_dims[-2], self.reorg_dim, k=1)
        self.reorg = reorg_layer(2)
        # 融合后的特征的卷积
        self.convsets_2 = Conv(4 * self.reorg_dim + self.head_dim, self.head_dim, k=3, p=1)
        # 预测层
        self.pred = nn.Conv2d(self.head_dim, self.num_anchors * (1 + self.num_classes + 4), kernel_size=1)

        if self.trainable:
            self.init_bias()

    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.pred.bias[..., 1 * self.num_anchors:(1 + self.num_classes) * self.num_anchors],
                          bias_value)

    def set_grid(self, input_size):
        self.input_size = input_size
        self.anchor_boxes = self.create_grid(self.input_size)

    def create_grid(self, input_size):
        # 此时需要返回[batch, hs * ws * num_anchor, 4], 每个anchor的中心点坐标 和 anchor的宽高
        hs = input_size // self.stride
        ws = input_size // self.stride
        # 获取每个grid_cell的坐标
        h_axis = torch.arange(hs).unsqueeze(1).repeat((1,ws)) # [hs, ws]
        w_axis = torch.arange(ws).unsqueeze(0).repeat((hs, 1)) # [hs, ws]
        axis = torch.stack([w_axis, h_axis], dim=-1)
        axis = axis.reshape(-1, 2) # [hs * ws, 2]
        axis = axis.unsqueeze(1).repeat(1, self.num_anchors, 1).reshape(-1, 2) # [hs * ws * num_anchor, 2]
        # 每个位置的anchor的宽高
        anchor_wh = self.anchor_size.unsqueeze(0) # [1, num_anchor, 2]
        anchor_wh = anchor_wh.repeat((hs * ws, 1, 1)) # [hs * ws, num_anchor, 2]
        anchor_wh = anchor_wh.reshape(-1, 2)
        anchor_boxes = torch.cat([axis, anchor_wh],dim=-1) # [hs * ws * num_anchor, 2]
        return anchor_boxes.to(self.device)

    def decode_boxes(self, anchor_boxes, txtytwth_pred):
        # anchor_boxes.shape = [hs * ws * num_anchor, 4], txtytwth_pred.shape = [b, hs * ws * num_anchor, 4]
        # 首先得到预测框的中心点坐标
        anchor_center = anchor_boxes[...,:2] + torch.sigmoid(txtytwth_pred[...,:2]) # 相对于stride的坐标
        # 求解预测框的宽高
        anchor_wh = anchor_boxes[...,2:] * torch.exp(txtytwth_pred[...,2:]) # 相对于stride的坐标
        # 将其整合在一起并且换为绝对的长度
        anchor = torch.cat([anchor_center, anchor_wh],dim=-1) * self.stride
        # 将其转换为左上角和右下角的坐标
        x1y1x2y2_pred = torch.zeros_like(anchor)
        x1y1x2y2_pred[...,:2] = anchor[...,:2] - anchor[...,2:] * 0.5
        x1y1x2y2_pred[...,2:] = anchor[...,:2] + anchor[...,2:] * 0.5
        return x1y1x2y2_pred # [b, hs * ws * num_anchor, 4]

    def inference(self,x):
        # 在预测过程中同理先利用模型的前向传播获取预测的东西[b, hs * ws, num_anchor * (1 + num_class + 4)]
        # 之后进入后处理(首先选取前topk的，之后根据置信度进行筛选，之后又根据mns进行筛选)
        outputs = self.backbone(x)
        c4, c5 = outputs['c4'], outputs['c5']  # c4 = [b, 512, h/16, w/16], c5 = [b, 1024, h/32, w/32]
        # 特征融合
        p4 = self.reorg(self.route_layer(c4))  # [b, 256, h/32, w/32]
        p5 = self.convsets_1(c5)  # [b, 1024, h/32, w/32]
        p5 = torch.concat([p4, p5], dim=1)  # [b, 1280, h/32, w/32]
        # 融合后的卷积
        p5 = self.convsets_2(p5)  # [b, 1024, h/32, w/32]
        # 最后一层的预测
        prediction = self.pred(p5)  # [b, num_anchor * (num_classes + 5), hs, ws]
        # 对prediction进行变换
        B, abC, H, W = prediction.shape
        KA = self.num_anchors
        NC = self.num_classes
        prediction = prediction.permute(0, 2, 3, 1)  # [b, hs, ws, num_anchor * (num_classes + 5)]
        prediction = prediction.reshape(B, -1, abC)  # [b, hs * ws, num_anchor * (num_classes + 5)]
        # 将置信度、类别预测、位置预测给分开来
        conf_pred = prediction[..., :KA]  # [b, hs * ws, num_anchors]
        conf_pred = conf_pred.reshape(B, -1).unsqueeze(-1)  # [b, hs * ws * num_anchors, 1]

        cls_pred = prediction[..., KA: KA + KA * NC]  # [b, hs * ws, num_class * num_anchors]
        cls_pred = cls_pred.reshape(B, -1, NC)  # [b, hs * ws * num_anchors, num_class]

        txtytwth_pred = prediction[..., KA + KA * NC:]  # [b, hs * ws, 4 * num_anchors]
        txtytwth_pred = txtytwth_pred.reshape(B, -1, 4)  # [b, hs * ws * num_anchors, 4]
        # 由于每次的输入都是一个样本因此 b=1, 可以对其进行降维处理
        conf_pred = conf_pred[0] # [hs * ws * num_anchors, 1]
        cls_pred = cls_pred[0] # [hs * ws * num_anchors, num_class]
        txtytwth_pred = txtytwth_pred[0] # [hs * ws * num_anchors, 4]
        # 对其进行后处理
        bboxes, scores, labels = self.postprocess(conf_pred, cls_pred, txtytwth_pred)
        return bboxes, scores, labels

    def postprocess(self, conf_pred, cls_pred, txtytwth_pred):
        anchors = self.anchor_boxes # [hs * ws * num_anchors, 4]
        # 求解每个类别的分数
        score = (torch.sigmoid(conf_pred) * torch.softmax(cls_pred,dim=-1)).flatten() # [hs * ws * num_anchors * num_class, ]
        # 从中选取前topk个分数最大的框
        num_topk = min(score.shape[0], 100)
        predict_prob, topk_idx = score.sort(descending=True)
        predict_prob = predict_prob[:num_topk]
        topk_idx = topk_idx[:num_topk]
        # 利用置信度筛选出大于置信度阈值的序列
        mask = (predict_prob > self.conf_thresh)
        topk_score = predict_prob[mask]  # 此时已经是从小到达进行排序的
        topk_idx = topk_idx[mask]
        # 确定的anchor以及所在的class和其对应的txtytwth_pred(anchor为选中的预测框对应的anchor)
        anchor_idx = topk_idx // self.num_classes
        labels = topk_idx % self.num_classes
        anchors = anchors[anchor_idx]
        txtytwth_pred = txtytwth_pred[anchor_idx]
        # 根据anchor和txtytwth_pred获取选取的标签框的左上角和右下角的坐标(绝对坐标)
        bbox = self.decode_boxes(anchors, txtytwth_pred)
        # 将bbox和topk_score和labels转换到cpu的numpy格式进行非极大值抑制
        topk_score = topk_score.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        bbox = bbox.detach().cpu().numpy()
        # 进行mns非极大值抑制
        keep = np.zeros(shape=(len(labels),),dtype=np.int)
        for cls in range(self.num_classes):
            cls_choice = np.where(labels == cls)[0] # 选取当前类别的bbox和topk_score
            if len(cls_choice) == 0:
                continue
            c_bbox = bbox[cls_choice] # [n,4]
            c_score = topk_score[cls_choice] # [n,]
            c_keep = self.nms(c_bbox, c_score)
            keep[cls_choice[c_keep]] = 1
        keep_idx = np.where(keep == 1)[0]
        choice_score = topk_score[keep_idx]
        choice_bbox = bbox[keep_idx]
        choice_labels = labels[keep_idx]
        choice_bbox /= self.input_size
        choice_bbox = np.clip(choice_bbox, a_min=0., a_max=1.)
        return choice_bbox, choice_score, choice_labels

    def nms(self,c_bbox, c_score):
        # 当前选中的bbox和其他边框的bbox计算iou，取出iou大于阈值的边框
        order = np.argsort(c_score)[::-1] # 从大到小给置信度排序
        keep = []
        area = (c_bbox[:,2] - c_bbox[:,0]) * (c_bbox[:,3] - c_bbox[:,1])
        while len(order) > 0:
            choice_box = c_bbox[order[0]]
            keep.append(order[0])
            # 计算当前的box和其他的box之间的iou
            xx1 = np.maximum(choice_box[0], c_bbox[order[1:]][:, 0])
            yy1 = np.maximum(choice_box[1], c_bbox[order[1:]][:, 1])
            xx2 = np.minimum(choice_box[2], c_bbox[order[1:]][:, 2])
            yy2 = np.minimum(choice_box[3], c_bbox[order[1:]][:, 3])
            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h
            union = area[order[0]] + area[order[1:]] - inter
            iou = inter / union
            keep_idx = np.where(iou < self.nms_thresh)[0]
            order = order[1 + keep_idx]
        return keep

    def forward(self,x,target=None):
        if self.trainable:
            outputs = self.backbone(x)
            c4, c5 = outputs['c4'], outputs['c5'] # c4 = [b, 512, h/16, w/16], c5 = [b, 1024, h/32, w/32]
            # 特征融合
            p4 = self.reorg(self.route_layer(c4)) # [b, 256, h/32, w/32]
            p5 = self.convsets_1(c5) # [b, 1024, h/32, w/32]
            p5 = torch.concat([p4,p5],dim=1) # [b, 1280, h/32, w/32]
            # 融合后的卷积
            p5 = self.convsets_2(p5) # [b, 1024, h/32, w/32]
            # 最后一层的预测
            prediction = self.pred(p5) # [b, num_anchor * (num_classes + 5), hs, ws]
            # 对prediction进行变换
            B, abC, H, W = prediction.shape
            KA = self.num_anchors
            NC = self.num_classes
            prediction = prediction.permute(0,2,3,1) # [b, hs, ws, num_anchor * (num_classes + 5)]
            prediction = prediction.reshape(B, -1, abC) # [b, hs * ws, num_anchor * (num_classes + 5)]
            # 将置信度、类别预测、位置预测给分开来
            conf_pred = prediction[...,:KA] # [b, hs * ws, num_anchors]
            conf_pred = conf_pred.reshape(B,-1).unsqueeze(-1) # [b, hs * ws * num_anchors, 1]

            cls_pred = prediction[...,KA: KA + KA * NC] # [b, hs * ws, num_class * num_anchors]
            cls_pred = cls_pred.reshape(B, -1, NC) # [b, hs * ws * num_anchors, num_class]

            txtytwth_pred = prediction[...,KA + KA * NC:] # [b, hs * ws, 4 * num_anchors]
            txtytwth_pred = txtytwth_pred.reshape(B, -1, 4) # [b, hs * ws * num_anchors, 4]
            # 根据预测的坐标信息和anchor的信息求解出预测框的左上角和右下角的坐标
            x1y1x2y2_pred = self.decode_boxes(self.anchor_boxes, txtytwth_pred) / self.input_size # [b, hs * ws * num_anchors, 4]
            # 计算标签框和预测框之间的iou值
            x1y1x2y2_tg = target[...,7:] # [b, hs * ws * num_anchors, 4]
            x1y1x2y2_pred = x1y1x2y2_pred.reshape(-1, 4) # [b * hs * ws * num_anchors, 4]
            x1y1x2y2_tg = x1y1x2y2_tg.reshape(-1, 4) # [b * hs * ws * num_anchors, 4]
            conf_tg = iou_score(x1y1x2y2_pred,x1y1x2y2_tg) # [b * hs * ws * num_anchors]
            conf_tg = conf_tg.reshape(B, -1, 1) # [b, hs * ws * num_anchors, 1]
            with torch.no_grad():
                conf_tg_copy = conf_tg.clone()
            # targets的重新组合
            targets = torch.cat([conf_tg_copy, target[...,:7]],dim=-1) # [b, hs * ws * num_anchors, 8]
            # 计算模型的损失
            conf_loss, cls_loss, bbox_loss, total_loss = compute_loss(pred_conf=conf_pred, pred_cls=cls_pred,
                                                                      pred_txtytwth=txtytwth_pred, target=targets)
            return conf_loss, cls_loss, bbox_loss, total_loss
        else:
            bboxes, scores, labels = self.inference(x)
            return bboxes, scores, labels

def compute_loss(pred_conf, pred_cls,pred_txtytwth, target):
    batch_size = pred_conf.shape[0]
    # 对target进行拆剪
    conf_gt = target[...,0].float() # [b, hs * ws * num_anchors]
    # 其中值为0的代表与anchor的iou小于阈值，值为1的代表与anchor的iou最大，值为-1的代表与anchor的iou大于阈值但是不是最大的
    obj_gt = target[...,1].float() # [b, hs * ws * num_anchors]
    cls_gt = target[...,2].long() # [b, hs * ws * num_anchors]
    txty_gt = target[...,3: 5] # [b, hs * ws * num_anchors, 2]
    twth_gt = target[...,5: 7] # [b, hs * ws * num_anchors, 2]
    weight = target[...,7].float() # [b, hs * ws * num_anchors]
    mask = (obj_gt > 0)
    # pred的内容进行拆分
    pred_conf = pred_conf[...,0] # [b, hs * ws * num_anchors]
    pred_cls = pred_cls.permute(0,2,1) # [b, num_class, hs * ws * num_anchors]
    pred_txty = pred_txtytwth[...,:2] # [b, hs * ws * num_anchors, 2]
    pred_twth = pred_txtytwth[...,2:] # [b, hs * ws * num_anchors, 2]
    # 置信度和类别和坐标的损失函数
    conf_loss_func = MSEWithLogitsLoss()
    cls_loss_func = nn.CrossEntropyLoss(reduction='none')
    txty_loss_func = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_func = nn.MSELoss(reduction='none')
    # 分别求解各自的损失
    conf_loss = conf_loss_func(conf_gt, obj_gt, pred_conf)
    conf_loss = conf_loss.sum() / batch_size

    cls_loss = cls_loss_func(pred_cls, cls_gt) * mask
    cls_loss = cls_loss.sum() / batch_size

    txty_loss = txty_loss_func(pred_txty,txty_gt).sum(-1) * mask * weight
    txty_loss = txty_loss.sum() / batch_size

    twth_loss = twth_loss_func(pred_twth,twth_gt).sum(-1) * mask * weight
    twth_loss = twth_loss.sum() / batch_size

    bbox_loss = txty_loss + twth_loss
    total_loss = conf_loss + cls_loss + bbox_loss
    return conf_loss, cls_loss, bbox_loss, total_loss

class MSEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self,tg_conf, tg_obj, pred_conf):
        pred_conf = torch.clamp(torch.sigmoid(pred_conf),min=1e-4, max=1 - 1e-4)
        mask_1 = (tg_obj == 0).float() # 针对于与anchor的iou小于阈值
        mask_2 = (tg_obj == 1).float() # 针对于与anchor的iou是最大值的情况
        loss2 = mask_2 * 5.0 * (tg_conf - pred_conf) ** 2
        loss1 = mask_1 * 1.0 * pred_conf ** 2
        loss = loss2 + loss1
        return loss

if __name__ == '__main__':
    # # 训练集上的验证
    # import matchor
    # torch.manual_seed(1)
    # device = torch.device('cuda')
    # x = torch.randn(size=(2, 3, 224, 224), device=device)
    # net = model = YOLOv2(
    #     device=device,
    #     input_size=224,
    #     num_classes=20,
    #     trainable=True,
    #     topk=100,
    #     anchor_size=[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    # )
    # input_size = 224
    # stride = 32
    # label_lists = [[[0.4, 0.3, 0.6, 0.7, 2], [0.2, 0.1, 0.9, 0.5, 4]], [[0.5, 0.6, 0.8, 0.8, 3]]]
    # ignore_thresh = 0.6
    # anchor_size = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    # gt_tensor = matchor.gt_creator(input_size, stride, label_lists, anchor_size, ignore_thresh)
    # gt_tensor = torch.tensor(gt_tensor, device=device)
    # net.to(device)
    # conf_loss, cls_loss, bbox_loss, total_loss = net(x,gt_tensor)
    # print(conf_loss)
    # print(cls_loss)
    # print(bbox_loss)
    # print(total_loss)

    # 测试集上的验证（验证为正确的）
    torch.manual_seed(1)
    device = torch.device('cuda')
    x = torch.randn(size=(2, 3, 224, 224), device=device)
    net = model = YOLOv2(
        device=device,
        input_size=224,
        num_classes=20,
        trainable=False,
        topk=100,
        anchor_size=[[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
    ).to(device)
    bboxes, scores, labels = net(x)
    print(bboxes,bboxes.shape,bboxes.dtype)
    print(scores,scores.shape,scores.dtype)
    print(labels,labels.shape)



