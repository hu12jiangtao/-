# 此时对于检测模型来说将骨干网络由resnet替换成了darknet(大量使用1*1卷积来降低模型的计算量，resnet的计算量太大了)
import pickle

import numpy as np
import torch
from torch import nn
import darknet

class Conv(nn.Module):
    def __init__(self, in_channel,out_channel,k=1,p=0,s=1,d=1,act=True):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size=k,padding=p,stride=s,dilation=d,bias=False),
                                  nn.BatchNorm2d(out_channel),nn.LeakyReLU(0.1,inplace=True) if act else nn.Identity())

    def forward(self,x):
        return self.conv(x)

class reorg_layer(nn.Module):
    def __init__(self, stride):
        super(reorg_layer, self).__init__()
        self.stride = stride

    def forward(self,x):
        b, c, h, w = x.shape
        h_num = h // self.stride
        w_num = w // self.stride
        x = x.reshape(b, c, h_num, self.stride, w_num, self.stride).transpose(3,4) # [b,c,h_num,w_num,stride,stride]
        x = x.reshape(b, c, h_num * w_num, self.stride * self.stride).transpose(2,3) # [b, c, stride * stride, h_num * w_num]
        # 此时必须要有这句话，原因是此时的顺序应当是每一层的part1 + 每一层的part2 +..+, 而不是 第一层的part1 + part2 + 第二层的part1 + part2
        x = x.reshape(b, c, -1, h_num, w_num).transpose(1,2) # [b, stride * stride, c, h_num, w_num]
        x = x.reshape(b, -1, h_num, w_num) # [b, stride * stride * c, h_num, w_num]
        return x


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

        # 骨干网络
        self.backbone, self.feat_dims = darknet.build_darknet19(pretrained=True)
        self.head_dim = 1024

        # 经过两个检测头
        self.convsets_1 = nn.Sequential(Conv(self.feat_dims[-1], self.head_dim, 3, 1),
                                        Conv(self.head_dim, self.head_dim, 3, 1))
        # 进行特征的融合(浅层特征信息和深层特征信息的融合)
        self.route_layer = Conv(self.feat_dims[-2], self.reorg_dim, k=1)
        self.reorg = reorg_layer(2)
        # 之后又加入一个卷积来融合特征
        self.convsets_2 = Conv(4 * self.reorg_dim + self.head_dim, self.head_dim, k=3, p=1)
        # 对于所有所有的anchor的预测框信息进行预测
        self.pred = nn.Conv2d(self.head_dim, self.num_anchors * (num_classes + 4 + 1), 1)

        if self.trainable:
            self.init_bias()

    def init_bias(self):
        # init bias
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.pred.bias[..., 1 * self.num_anchors:(1 + self.num_classes) * self.num_anchors],
                          bias_value)

    def create_grid(self, input_size): # 目标是获取每个grid_cell中的所有的anchor对应的grid_cell的左上角的坐标以及anchor的宽高
        # 首先获得每个grid_cell的相对坐标
        w, h = input_size, input_size
        step_w, step_h = w // self.stride, h // self.stride
        h_axis = torch.arange(step_h).unsqueeze(-1).repeat(1, step_w)
        w_axis = torch.arange(step_w).unsqueeze(0).repeat(step_h, 1)
        grid_axis = torch.stack([w_axis,h_axis],dim=-1) # [step_h, step_w, 2]
        grid_axis = grid_axis.reshape(-1, 2) # [step_h * step_w, 2]
        # 求解每个anchor的坐标信息
        grid_xy = grid_axis.unsqueeze(1).repeat(1,self.num_anchors,1) # [step_h * step_w, num_anchor, 2]
        anchor_wh = self.anchor_size.unsqueeze(0).repeat(step_w * step_h, 1, 1) # [step_h * step_w, num_anchor, 2]
        anchor_box = torch.cat([grid_xy, anchor_wh], dim=-1) # [step_h * step_w, num_anchor, 4]
        anchor_box = anchor_box.reshape(-1, 4).to(self.device) # [step_h * step_w * num_anchor, 4]
        return anchor_box

    def set_grid(self,input_size):
        self.input_size = input_size
        self.anchor_boxes = self.create_grid(self.input_size)

    def decode_boxes(self, anchors, txtytwth_pred): # 用于求解预测框的坐标
        # anchor.shape=[b, H*W*num_anchor, 4],其中存储着anchor所在的grid_cell的左上角坐标，以及anchor的相对长宽(相对于stride)
        # txtytwth_pred=[b, H*W*num_anchor, 4],存放利用模型得到的预测框的信息(前两个为预测框的中心点的偏移量，后两个存储当前预测框和anchor宽高的倍率)
        # sigmoid(txtytwth_pred[...,:2])才是真正的中心点偏移量(sigmoid的作用是将中心点的偏移量固定在0，1之间)
        # 若无sigmoid则偏移量可能大于1，此时预测框的中心点就不在当前的grid-cell中了(与事实不符，增大了训练难度)

        # 给出每个预测框的中心点的相对坐标
        xy_pred = anchors[...,:2] + torch.sigmoid(txtytwth_pred[...,:2])
        # 给出每个预测框的宽高的相对坐标
        wh_pred = anchors[...,2:] * torch.exp(txtytwth_pred[...,2:])
        # 将两者变为绝对坐标
        xywh_pred = torch.cat([xy_pred,wh_pred],dim=-1) * self.stride # [b, H*W*num_anchor, 4]
        # 给出预测框的 中心点+高宽 的格式 换算为 左上角坐标+右上角坐标
        x1y1x2y2_pred = torch.zeros_like(xywh_pred)
        x1y1x2y2_pred[...,:2] = xywh_pred[...,:2] - xywh_pred[...,2:] * 0.5
        x1y1x2y2_pred[...,2:] = xywh_pred[...,:2] + xywh_pred[...,2:] * 0.5
        return x1y1x2y2_pred # 返回的是绝对的中心点坐标加上宽高

    def inference(self,x): # 此时没有标签，与需要对预测的内容进行非极大值的抑制
        # 首先和训练的过程一样得到预测的内容
        feat = self.backbone(x)
        c4, c5 = feat['c4'], feat['c5']
        p4 = self.route_layer(c4)
        p4 = self.reorg(p4)
        p5 = self.convsets_1(c5)
        p5 = torch.cat([p4,p5],dim=1)
        p5 = self.convsets_2(p5)
        prediction = self.pred(p5) # [b, num_anchors * (1 + 20 + 4), H, W]
        # 对其进行分类
        B, abC, H, W = prediction.shape
        KA = self.num_anchors
        NC = self.num_classes
        prediction = prediction.permute(0,2,3,1).reshape(B, -1, abC) # [b, H * W, num_anchors * (1 + 20 + 4)]
        # 将置信度、每个类别的条件概率，中心点和宽高的值分开
        conf_pred = prediction[...,:KA].reshape(B, -1, 1) # [b, H * W * num_anchors, 1]
        cls_pred = prediction[...,KA: KA + KA * NC].reshape(B, -1, NC) # [b, H * W * num_anchors, num_classes]
        txtytwth_pred = prediction[...,KA + KA * NC:].reshape(B, -1, 4) # [b, H * W * num_anchors, 4]
        # 由于输入的内容为一个批量
        conf_pred = conf_pred[0] # [H * W * num_anchors, 1]
        cls_pred = cls_pred[0] # [H * W * num_anchors, num_classes]
        txtytwth_pred = txtytwth_pred[0] # [H * W * num_anchors, 4]
        # 将其进行后处理
        bboxes,scores,labels = self.postprocess(conf_pred, cls_pred, txtytwth_pred)
        return bboxes,scores,labels

    def postprocess(self, conf_pred, cls_pred, reg_pred):
        # conf_pred=[H * W * num_anchor, 1], cls_pred=[H * W * num_anchors, num_classes], reg_pred=[H * W * num_anchors, 4]
        anchors = self.anchor_boxes # [step_h * step_w * num_anchor, 4]
        # 求解每个类别的概率
        scores = (torch.sigmoid(conf_pred) * torch.softmax(cls_pred,dim=-1)).flatten() # [H * W * num_anchor * num_classes, ]
        # 取出所有的grid_cell的所有anchor的所有类别的最大的的前面的num_topk的分数以及其所在的位置
        num_topk = min(self.topk, reg_pred.shape[0])
        predicted_prob, topk_idxs = scores.sort(descending=True)
        predicted_prob = predicted_prob[:num_topk]
        topk_idxs = topk_idxs[:num_topk]
        # 和设定的分数阈值进行比较，保留大于分数阈值的序列
        keep_idx = topk_idxs > self.conf_thresh
        topk_score = predicted_prob[keep_idx]
        topk_idxs = topk_idxs[keep_idx]
        # 获取当前保留的分数所在的anchor
        anchor_idx = torch.div(topk_idxs, self.num_classes,rounding_mode='floor')
        # 获取当前的保留分数为当前anchor中的哪一个标签类别
        labels = topk_idxs % self.num_classes
        # 获取当前的anchor的坐标
        reg_pred = reg_pred[anchor_idx]  # 获取预测框当前选取的anchor的坐标位置信息
        anchors = anchors[anchor_idx] # 获取对应的anchor的位置信息(所在的grid_cell的坐标，anchor的宽高)
        # 求解预测框的实际的左上角和右下角的坐标(绝对坐标而不是相对的坐标)
        bboxes = self.decode_boxes(anchors, reg_pred) # [num_topk, 4]
        # 转换为numpy的格式
        topk_score = topk_score.detach().cpu().numpy() # [num_topk, ]
        labels = labels.detach().cpu().numpy() # [num_topk, ]
        bboxes = bboxes.detach().cpu().numpy() # [num_topk, 4]
        # 这些预测框进行非极大值抑制
        keep = np.zeros(shape=len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            # 取出当前类别的所有的标签
            index = np.where(i == labels)[0]
            if len(index) == 0:
                continue
            cur_score = topk_score[index]
            cur_bbox = bboxes[index]
            c_keep = self.nms(cur_bbox, cur_score)
            keep[index[c_keep]] = 1
        keep = np.where(keep > 0)[0]
        labels = labels[keep]
        scores = topk_score[keep]
        bboxes = bboxes[keep]
        # 预测框进行归一化处理
        bboxes = bboxes / self.input_size
        bboxes = np.clip(bboxes, 0., 1.)
        return bboxes,scores,labels

    def nms(self, c_bboxes, c_scores):
        # 按照c_scores的大小对c_bboxes进行排序, 之后当前的bbox和其他的bbox进行iou计算
        order = np.argsort(c_scores)[::-1]
        # 计算每个预测框的面积
        area = (c_bboxes[:, 2] - c_bboxes[:, 0]) * (c_bboxes[:, 3] - c_bboxes[:, 1])
        # 筛选出需要保留的的索引
        keep = []
        while len(order) > 0:
            cur_box = c_bboxes[order[0]]
            keep.append(order[0])
            # 和其他所有的bbox计算iou
            x1 = np.maximum(cur_box[0], c_bboxes[order[1:]][:,0])
            y1 = np.maximum(cur_box[1], c_bboxes[order[1:]][:,1])
            x2 = np.minimum(cur_box[2], c_bboxes[order[1:]][:,2])
            y2 = np.minimum(cur_box[3], c_bboxes[order[1:]][:,3])
            w = np.maximum(1e-10, x2 - x1)
            h = np.maximum(1e-10, y2 - y1)
            inter = w * h
            union = area[order[0]] + area[order[1:]] - inter
            iou = inter / union
            drop = np.where(iou <= self.nms_thresh)[0]
            order = order[drop + 1]
        return keep

    def forward(self, x, target=None): # x = [b, 32, h, w]
        if self.trainable is True:
            feat = self.backbone(x)
            c4, c5 = feat['c4'], feat['c5'] # c4 = [b, 512, h/16, w/16], c5 = [b, 1024, h/32, w/32]
            # 加入细粒度特征
            p4 = self.route_layer(c4) # [b, 64, h/16, w/16]
            p4 = self.reorg(p4) # [b, 64*4, h/32, w/32]
            # p4相同
            p5 = self.convsets_1(c5) # [b, 1024, h/32, w/32]
            # 进行特征融合
            p5 = torch.cat([p4,p5],dim=1)
            # 利用卷积融合信息
            p5 = self.convsets_2(p5)
            # 预测层
            # prediction的排列顺序应为[a1,...,a5, a1_num_class(20类),...,a5_num_class, a1_axis(预测框),...,a5_axis]
            prediction = self.pred(p5) # [b, num_anchors * (20 + 1 + 4), H, W]  # 到达prediction已经验证其为正确的
            # 求解模型的损失函数
            B, abC, H, W = prediction.shape
            KA = self.num_anchors
            NC = self.num_classes
            # 对prediction进行变换处理
            prediction = prediction.permute(0, 2, 3, 1)  # [b, H, W, num_anchors * (20 + 1 + 4)]
            prediction = prediction.reshape(B, H * W, abC) # [b, H * W, num_anchors * (20 + 1 + 4)]
            # 分离出置信度，条件概率，坐标位置
            # 得到每个anchor对应的预测框的置信度
            conf_pred = prediction[...,:KA].reshape(B, -1).unsqueeze(-1) # [b, H * W * num_anchors, 1] 已经经过验证
            # 得到每个anchor对应的预测框的条件概率
            cls_pred = prediction[...,KA: KA + KA * NC] # [b, H * W, num_anchor * num_classes]
            cls_pred = cls_pred.reshape(B, -1, NC) # [b, H * W * num_anchors, num_classes] ,已经经过验证
            # 得到每个anchor对应的预测框的坐标位置
            txtytwth_pred = prediction[...,KA + KA * NC:] # [b, H * W, num_anchor * 4]
            txtytwth_pred = txtytwth_pred.reshape(B, -1, 4) # [b, H * W * num_anchor, 4]
            # 求解预测得到每个anchor对应的预测框（转换为相对坐标），且其中存储着左上角和右下角的坐标
            x1y1x2y2_pred = self.decode_boxes(self.anchor_boxes, txtytwth_pred) / self.input_size # [b, H * W * num_anchor, 4]
            x1y1x2y2_pred = x1y1x2y2_pred.reshape(-1, 4) # [b * H * W * num_anchor, 4] ,已经经过验证 针对于整张图的相对尺寸的相对坐标
            x1y1x2y2_gt = target[...,7:].reshape(-1, 4) # [b * H * W * num_anchor, 4] # 针对于整张图的相对尺寸的相对坐标
            # 计算预测框和标签框之间的iou值
            iou_pred = iou_score(x1y1x2y2_pred,x1y1x2y2_gt).reshape(B, -1, 1) # [B, H*W*num_anchor, 1] 经过验证其是对的
            with torch.no_grad():
                conf_gt = iou_pred.clone() # [B, H*W*num_anchor, 1]
            targets = torch.cat([conf_gt, target[:, :, :7]], dim=2)  # [B, H*W*num_anchor, 1 + 7]

            conf_loss,cls_loss,bbox_loss,total_loss = compute_loss(pred_conf=conf_pred,pred_cls=cls_pred,
                                                                   pred_txtytwth=txtytwth_pred,target=targets)
            return conf_loss,cls_loss,bbox_loss,total_loss
        else:
            bboxes, scores, labels = self.inference(x)
            return bboxes, scores, labels

def compute_loss(pred_conf, pred_cls, pred_txtytwth, target):
    # 此时的损失仍然由四部分构成
    # 1.置信度损失(此时只考虑1.anchor和对应的target小于规定的阈值 和 2.anchor和对应的target的iou值最大的情况）
    #   在2.的情况下的置信度为预测框和标签框的iou和预测的置信度的平方
    # 2.类别损失(只考虑anchor和target的iou最大的情况下的类别损失)
    # 3.宽高损失(只考虑anchor和target的iou最大的情况的预测框): log(预测框的宽高 / 对应的anchor的宽高)[该项由模型直接预测得到] 和 log(标签框的宽高 / 对应的anchor的宽高)
    # 4.中心点的损失(只考虑anchor和target的iou最大的情况的预测框): 预测框的偏移量(值为预测的txty进行sigmoid) 和 标签框的偏移量 的BCELoss
    batch_size = pred_conf.shape[0]
    # 标签的分类
    tg_conf = target[...,0].float() # [b, H*W*num_anchor]
    tg_obj = target[...,1].float() # [b, H*W*num_anchor]
    tg_cls = target[...,2].long() # [b, H*W*num_anchor]
    tg_txty = target[...,3:5].float() # [b, H*W*num_anchor, 2]
    tg_twth = target[...,5:7].float() # [b, H*W*num_anchor, 2]
    tg_weight = target[...,7] # [b, H*W*num_anchor]
    tg_mask = (tg_weight > 0)
    # 预测内容的分类
    pred_conf = pred_conf[...,0] # [b, H * W * num_anchors]
    pred_cls = pred_cls.permute(0,2,1) # [b, num_classes, H * W * num_anchors]
    pred_txty = pred_txtytwth[...,:2] # [b, H * W * num_anchors, 2]
    pred_twth  =pred_txtytwth[...,2:] # [b, H * W * num_anchors, 2]
    # 用于预测的损失函数
    conf_loss_function = MSEWithLogitsLoss()
    cls_loss_function = nn.CrossEntropyLoss(reduction='none')
    twth_loss_function = nn.MSELoss(reduction='none')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    # 计算此时的损失
    # 第一个损失:置信度的损失
    with open('1.pkl', 'wb') as f:
        pickle.dump([tg_conf, tg_obj, pred_conf],f)
    conf_loss = conf_loss_function(tg_conf, tg_obj, pred_conf)
    conf_loss = conf_loss.sum() / batch_size
    # 第二个损失:类别的损失(只有tg_mask为True的情况)
    cls_loss = cls_loss_function(pred_cls,tg_cls) * tg_mask
    cls_loss = cls_loss.sum() / batch_size
    # 第三个损失:预测框和标签框的宽高损失
    twth_loss = twth_loss_function(pred_twth, tg_twth).sum(-1) * tg_mask * tg_weight
    twth_loss = twth_loss.sum() / batch_size
    # 第四个损失:预测框和标签框的中心点损失
    txty_loss = txty_loss_function(pred_txty,tg_txty).sum(-1) * tg_mask * tg_weight
    txty_loss = txty_loss.sum() / batch_size

    bbox_loss = twth_loss + txty_loss
    total_loss = conf_loss + cls_loss + bbox_loss
    return conf_loss, cls_loss, bbox_loss, total_loss

class MSEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()
    def forward(self, tg_conf, tg_obj, pred_conf):
        pred_conf = torch.clamp(torch.sigmoid(pred_conf), min=1e-4, max=1-1e-4)
        # tg_conf = pred_conf = tg_obj = [b, H * W * num_anchors]
        h1 = (tg_obj == 0).float() # anchor和标签框的iou小于阈值的情况
        h2 = (tg_obj == 1).float() # anchor和标签框的iou最大的情况时
        loss1 = 5.0 * h2 * (pred_conf - tg_conf) ** 2
        loss2 = 1.0 * h1 * pred_conf ** 2
        loss = loss1 + loss2
        return loss

def iou_score(bboxes_a, bboxes_b):
    # bboxes_a = bboxes_b = [b * H * W * num_anchor, 4]
    # 求解左上角和右下角的坐标
    tl = torch.max(bboxes_a[:,:2], bboxes_b[:,:2]) # [b * H * W * num_anchor, 2]
    br = torch.min(bboxes_a[:,2:], bboxes_b[:,2:]) # [b * H * W * num_anchor, 2]
    area_a = torch.prod(bboxes_a[:,2:] - bboxes_a[:,:2],dim=1) # [b * H * W * num_anchor, ]
    area_b = torch.prod(bboxes_b[:,2:] - bboxes_b[:,:2],dim=1) # [b * H * W * num_anchor, ]
    flag = (tl < br).type(tl.dtype).prod(dim=-1)
    inter = flag * torch.prod(br - tl,dim=1)
    return inter / (area_a + area_b - inter)  # [b * H * W * num_anchor, ]


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
    #
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


