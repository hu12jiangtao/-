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
        return torch.cat([x,x1,x2,x3],dim=1) # [batch,4*c,h,w]

class Conv(nn.Module):
    def __init__(self,c1, c2, k, s=1, p=0, d=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(c1,c2,k,stride=s,padding=p,dilation=d,groups=g),
                                  nn.BatchNorm2d(c2),nn.LeakyReLU(0.1,inplace=True) if act else nn.Identity())

    def forward(self,x):
        return self.conv(x)


class MSEWithLogitsLoss(nn.Module):
    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()

    def forward(self, logits, target):
        # target.shape=logits.shape=[batch, H*W]
        # grid_cell中不存在标签框中心以及存在标签框中心的置信度损失给予不同的权重
        confidence = torch.clamp(torch.sigmoid(logits),min=1e-4,max=1-1e-4) # 真正的置信度
        save = (target == 1.).float()
        mask = (target == 0.).float()
        save_loss = 5. * save * (target - confidence) ** 2
        mask_loss = mask * confidence ** 2
        return save_loss + mask_loss




def compute_loss(pred_conf,pred_cls,pred_txtytwth,targets):
    # pred_conf.shape=[b, h*w, 1], pred_cls.shape=[b, h*w, 20], pred_txtytwth=[b, h*w, 4]
    # targets.shape=[b, h*w, 7]
    batch_size = pred_conf.shape[0]
    # 首先给出每种损失所用的损失函数
    pred_conf_loss = MSEWithLogitsLoss() # 置信度的损失
    pred_cls_loss = nn.CrossEntropyLoss(reduction='none') # 类别的交叉熵函数
    pred_txty_loss = nn.BCEWithLogitsLoss(reduction='none') # 预测框的相对坐标的损失函数
    pred_twth_loss = nn.MSELoss(reduction='none') # 预测框的宽高的损失函数
    # 对输入的target进行处理
    target_conf = targets[...,0] # [b, h*w]
    target_label = targets[...,1].long() # [b, h*w]
    target_txty = targets[..., 2:4] # [b, h*w, 2]
    target_twth = targets[..., 4:6] # [b, h*w, 2]
    target_weight = targets[...,-1] # [b, h*w]
    # 对预测的内容进行处理
    pred_conf = pred_conf[...,0] # [batch, h*w] 经过sigmoid后才是真正的置信度
    pred_cls = pred_cls.permute(0,2,1) # [b, 20, h*w]
    pred_txty = pred_txtytwth[..., :2] # [b, h*w, 2] # 经过sigmoid后才是真正的相对于当前grid_cell的坐标
    pred_twth = pred_txtytwth[..., 2:]  # [b, h*w, 2]
    # 开始计算损失
    # 置信度的损失函数
    conf_loss = pred_conf_loss(pred_conf, target_conf)
    conf_loss = conf_loss.sum() / batch_size
    # 标签的损失函数(此时类别的概率为置信度 * 条件概率)
    cls_loss = pred_cls_loss(pred_cls, target_label) * target_conf # 只记录置信度为1的时候的bbox和标签框的损失函数
    cls_loss = cls_loss.sum() / batch_size
    # 中心坐标的损失函数
    txty_loss = pred_txty_loss(pred_txty,target_txty).sum(-1) * target_weight * target_conf # [batch,h*w]
    txty_loss = txty_loss.sum() / batch_size
    # bbox的宽高和target之间的损失
    twth_loss = pred_twth_loss(pred_twth,target_twth).sum(-1) * target_weight * target_conf
    twth_loss = twth_loss.sum() / batch_size
    bbox_loss = twth_loss + txty_loss
    loss = conf_loss + cls_loss + bbox_loss
    return conf_loss, cls_loss, bbox_loss, loss


class myYOLO(nn.Module):
    def __init__(self, device, input_size=None, num_classes=20, trainable=False, conf_thresh=0.01, nms_thresh=0.5):
        super(myYOLO, self).__init__()
        self.trainable = trainable
        self.device = device
        self.stride = 32
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # 求解得到每一个grid_cell的坐标
        self.grid_cell = self.create_grid(input_size)
        # 首先是进入backbone阶段
        pretrain_net = torchvision.models.resnet18(pretrained=True)
        feature_dim = pretrain_net.fc.in_features
        self.backbone = nn.Sequential(*list(pretrain_net.children())[:-2])
        # neck阶段(SPP,Conv)
        self.neck = nn.Sequential(SPP(), Conv(feature_dim * 4, feature_dim, k=1))
        # dec阶段(全卷积)
        self.convsets = nn.Sequential(Conv(feature_dim, feature_dim // 2, k=1),
                                      Conv(feature_dim // 2, feature_dim, k=3, p=1),
                                      Conv(feature_dim, feature_dim // 2, k=1),
                                      Conv(feature_dim // 2, feature_dim, k=3, p=1))
        # pred阶段
        self.pred = nn.Conv2d(feature_dim, 1 + num_classes + 4, kernel_size=1)
        # 进行初始化
        if self.trainable:
            self.init_bias()



    def init_bias(self):
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.pred.bias[..., :1], bias_value)
        nn.init.constant_(self.pred.bias[..., 1:1+self.num_classes], bias_value)

    def create_grid(self, input_size): # 给每个grid_cell分配坐标，例如第一行的第二个grid_cell的坐标应为(1,0)
        ws = input_size // self.stride
        hs = input_size // self.stride
        axis_w = torch.arange(ws)
        axis_w = axis_w.reshape(1,-1).repeat((hs,1)) # [hs, ws]
        axis_h = torch.arange(hs)
        axis_h = axis_h.reshape(-1,1).repeat((1,ws)) # [hs, ws]
        axis = torch.stack([axis_w,axis_h],dim=-1) # [hs, ws, 2]
        return axis.reshape(-1,2).to(self.device) # [hs * ws, 2]



    def set_grid(self, input_size):
        self.grid_cell = self.create_grid(input_size)


    def inference(self, x):
        # 与训练的过程相同首先得到所有的预测框
        feature = self.backbone(x)
        feature = self.neck(feature)
        feature = self.convsets(feature)
        pred = self.pred(feature) # [b, 1 + 20 + 4, H, W]
        # 进行变换
        pred = pred.permute(0,2,3,1)
        pred = pred.reshape(pred.shape[0], -1, pred.shape[-1]) # [b, h * w, 25]
        # 将pred变成想要的内容
        pred_conf = pred[..., :1] # [b, h*w, 1]
        pred_cls = pred[...,1: 1 + self.num_classes] # [b, h*w, 20]
        pred_txtythtw = pred[...,1 + self.num_classes:] # [b, h*w, 4]
        # 由于输入的样本数量为1，因此可以将其b这个维度给去掉
        pred_conf = pred_conf[0] # [h*w, 1]
        pred_cls = pred_cls[0] # [h*w, 20]
        pred_txtythtw = pred_txtythtw[0] # [h*w, 4]
        # 求解边框的实际的宽高
        out_bbox = self.decode_boxes(pred_txtythtw) / self.input_size # [h*w, 4]
        out_bbox = torch.clamp(out_bbox, min=0., max=1.)
        # 每个类别的概率
        score = torch.sigmoid(pred_conf) * torch.softmax(pred_cls,dim=-1) # [h*w, 20]
        # 通过极大值抑制、置信度阈值等方法对预测的边框进行筛选
        out_bbox = out_bbox.detach().cpu().numpy() # [h*w, 4]
        score = score.detach().cpu().numpy() # [h*w, 20]
        save_bbox, save_score, save_label = self.postprocess(out_bbox, score)
        return save_bbox,save_score,save_label

    def postprocess(self,bbox, score):
        # 得到每个预测框中的最大类别概率
        label = np.argmax(score,axis=-1) # [h*w, ]
        score = score[np.arange(label.shape[0]), label] # [h*w, ]
        index = np.where(score >= self.conf_thresh)[0]
        bbox = bbox[index] # [n, 4]
        score = score[index]
        label = label[index] # [n,],假设此时保存下来的预测框的个数为n
        keep = np.zeros(shape=(len(bbox)),dtype=int) # [n,]
        for i in range(self.num_classes):
            choice = np.where(label == i)[0] # 选择出当前的图片的当前类别的bbox锚框
            if len(choice) == 0:
                continue
            else:
                c_bbox = bbox[choice]
                c_score = score[choice]
                c_keep = self.nms(c_bbox, c_score) # 最终当前保存下来的的序列
            keep[choice[c_keep]] = 1

        save_index = np.where(keep == 1)[0]
        save_bbox = bbox[save_index,:] # [m, 4],m为经过非极大值抑制后这张图上保留的检测目标的位置
        save_score = score[save_index]
        save_label = label[save_index]
        return save_bbox, save_score, save_label

    def nms(self, c_bbox, c_scores): # c_bbox=[n, 4]
        x1 = c_bbox[:, 0]
        y1 = c_bbox[:, 1]
        x2 = c_bbox[:, 2]
        y2 = c_bbox[:, 3]
        area = (x2 - x1) * (y2 - y1)
        # 首先需要根据c_scores进行排序
        index = np.argsort(c_scores)[::-1]
        keep = []
        while len(index) > 0:
            i = index[0] # 最大置信度的预测框的序列
            keep.append(index[0])
            # 当前的预测框和其他的预测框之间的iou值
            xx1 = np.maximum(x1[i], x1[index[1:]]) # [n, ]
            yy1 = np.maximum(y1[i], y1[index[1:]])
            xx2 = np.minimum(x2[i], x2[index[1:]])
            yy2 = np.minimum(y2[i], y2[index[1:]])
            w = np.maximum(1e-10, xx2 - xx1) # [n, ]
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h # [n, ]
            union = area[i] + area[index[1:]] - inter # [n, ]
            iou = inter / union
            # 与设定的IOU阈值进行比较(删除大于阈值的预测框)
            save_index = np.where(iou <= self.nms_thresh)[0]
            index = index[save_index + 1]
        return keep

    def decode_boxes(self, pred): #
        # pred = [h*w, 4],
        # 获取中心点的绝对坐标
        out_bbox = torch.zeros_like(pred)
        # 存储真实的中心坐标
        pred[...,:2] = (self.grid_cell + torch.sigmoid(pred[...,:2])) * self.stride
        # 获得真实的bbox的宽高
        pred[...,2:] = torch.exp(pred[...,2:])
        # 获得bbox的坐标
        out_bbox[...,:2] = pred[...,:2] - 0.5 * pred[...,2:] # 左上角的坐标
        out_bbox[...,2:] = pred[...,:2] + 0.5 * pred[...,2:] # 右下角的坐标
        return out_bbox

    def forward(self, x, targets=None):
        if self.trainable is True:
            # targets = [batch, h*w, 7]
            feature = self.backbone(x) # [batch,c,h,w]
            feature = self.neck(feature)
            feature = self.convsets(feature)
            pred = self.pred(feature) # [batch, 1+20+4, h, w]
            # 对预测的内容进行处理，处理成和target相同的格式
            pred = pred.permute(0,2,3,1) # [batch, h, w, c]
            pred = pred.reshape(pred.shape[0], -1, pred.shape[-1]) # [batch,h*w,c]
            # 将pred中的内容分开
            pred_conf = pred[...,:1] #
            pred_cls = pred[...,1: 1 + self.num_classes] # [batch,h*w,20]
            pred_txtytwth = pred[..., 1 + self.num_classes:] # [batch,h*w,4]
            # 计算损失值
            conf_loss, cls_loss, bbox_loss, total_loss = compute_loss(pred_conf, pred_cls, pred_txtytwth, targets)
            return conf_loss, cls_loss, bbox_loss, total_loss
        else:
            save_bbox, save_score, save_label = self.inference(x)
            return save_bbox, save_score, save_label



if __name__ == '__main__':
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


    # # 推理部分的验证
    # torch.manual_seed(1)
    # device = torch.device('cuda')
    # model = myYOLO(device, 416, 20, trainable=False).to(device)  # 输入的大小为416
    # model.eval()
    # x = torch.randn(size=(1, 3, 416, 416), device=device)
    # bbox, score, labels = model(x)
    # print(bbox)
    # print(score)
    # print(labels)



