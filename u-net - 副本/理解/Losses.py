# 用于计算训练模型的损失函数
import torch
from torch import nn
from torch.nn import functional as F

# 计算Dice_Loss(1 - Dice系数)
# 1.首先计算每个类别的预测概率，之后根据数据集中的label创建Dice_label
# 根据公式计算Dice系数
def build_target(label,ignore_index,num_classes):
    dice_target = label.clone()
    dice_index = torch.eq(dice_target,ignore_index)
    dice_target[dice_index] = 0 # [batch,h,w]
    dice_target = F.one_hot(dice_target,num_classes).float() # [batch,h,w,num_classes]
    dice_target[dice_index] = 255
    return dice_target.permute(0,3,1,2)

def dice_loss(model_out,label,ignore_index,num_classes): # 取出了照片中细胞外的像素造成的损失
    # 首先计算每个类别的预测概率
    x = F.softmax(model_out,dim=1)
    # 创建dice_label
    dice_target = build_target(label,ignore_index,num_classes) # 此时输出的dice_label形状应当和相同model_out相同
    # 计算每一个类别的每一个batch的平均dice系数
    fn = multiclass_dice_coeff
    dice_loss = 1 - fn(x,dice_target,ignore_index)
    return dice_loss

def multiclass_dice_coeff(x,target,ignore_index, epsilon=1e-6): # 用于计算每个类别的dice系数
    classes = x.shape[1]
    dice = 0
    for i in range(classes):
        dice += dice_coeff(x[:,i,:,:], target[:,i,:,:], ignore_index, epsilon)
    return dice / classes

def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    # 对于特定的一个类别的每个样本的dice系数计算,x.shape=target.shape=[batch,h,w]
    d = 0
    batch_size = x.shape[0]
    for i in range(batch_size):
        x_now = x[i].reshape(-1)
        t_now = target[i].reshape(-1)
        mask = torch.ne(t_now,ignore_index) # 原本ignore_index元素的位置值变为False，其余的地方为True
        t_now = t_now[mask]
        x_now = x_now[mask]
        inter = torch.dot(t_now,x_now.type(t_now.dtype)) # 分子部分
        set_sum = torch.sum(x_now) + torch.sum(t_now) # 坟墓部分
        if set_sum == 0: # 此时说明这个batch的这个类别的预测概率和标签值都为0，这样此时的dice系数应当为1
            set_sum = 2 * inter
        d += (2 * inter + epsilon) / (set_sum + epsilon)
    return d / batch_size

def Focal_Loss(outputs, labels, device, gamma=2):  # 用于平衡样本不均衡(图中细胞占比大，背景占比小)，同时着重关注较难分类的对象
    # outputs.shape=[batch,num_classes,h,w], label.shape[batch,h,w], weight = [num_classes,]
    batch, num_classes, h, w = outputs.shape
    weight = torch.tensor([1,2],dtype=torch.float32,device=device) # 每个标签的权重，标签0代表的是背景，标签1代表的是神经(由于图片中神经少，因此神经的权重较大)
    # 忽略标签为num_classes造成的损失，设置每个类别的权重（其中weight中的第0个元素，对应第0个标签）
    # 此时nn.CrossEntropyLoss(reduction='none', ignore_index=255)+之后mean求平均得到的l1
    # 和 nn.CrossEntropyLoss(ignore_index=255)得到的l2不相同 ,l2计算的时候求解的平均是除 总的个数-ignore_index序列的个数
    criterion = nn.CrossEntropyLoss(ignore_index=255,weight=weight)
    y_hat = outputs.permute(0,2,3,1).reshape(-1,num_classes) # [b * h * w, nu,_classes]
    y = labels.reshape(-1) # [b * h * w]
    loss = criterion(y_hat,y)
    # 此时利用Focal_Loss对模型的准确率并没有特别大的变化(此时的Focal_Loss求均值时包含了ignore_index的个数，因此损失值相交而言较小)，最后结果为:
    # acc_global:0.954 acc_cls:[0.979, 0.776] iou:[0.948, 0.68] mean_iou:0.814 cumulative_dice:0.809
    # log_pred = -criterion(y_hat,y) # [b * h * w, ]
    # pred = torch.exp(log_pred)
    # # 此时为focal_loss,当pt接近1时 logpt接近0,(1 - pt)**gamma接近0，此时这个是易分类样本，此时的损失值值较小，这个易分类的样本所占权重较小
    # # 当pt接近0时logpt接近负无穷,(1 - pt)**gamma接近1，说明这是一个比较难分类的，此时的损失值值较大，这个难分类的样本所占权重较大
    # loss = - (1 - pred) ** gamma * log_pred
    # loss = loss.mean()
    return loss

def sum_loss(outputs,labels,ignore_index,num_classes,device):
    loss = Focal_Loss(outputs, labels,device) + dice_loss(outputs,labels,ignore_index,num_classes)
    return loss

if __name__ == '__main__':
    device = torch.device('cpu')
    torch.manual_seed(1)
    ignore_index = 255
    num_classes = 2
    label = torch.tensor([[[255,1,255],[1,1,0],[255,0,255]]])
    model_out = torch.randn(size=(1,2,3,3))
    l = dice_loss(model_out,label,ignore_index,num_classes)
    print(l)

    l2 = Focal_Loss(model_out, label, device, gamma=2)
    print(l2)


