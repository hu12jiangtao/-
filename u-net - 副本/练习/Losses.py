# Loss分为了两个部分:1.普通的交叉熵损失 2.Dice损失(1-Dice系数)
from torch import nn
import torch
from torch.nn import functional as F

# 1.普通的交叉熵函数
def Focal_Loss(outputs, labels, device):
    # 此时的输入outputs.shape=[batch,num_class,h,w],label=[batch,h,w]
    outputs = outputs.permute(0,2,3,1) # [batch * h * w, num_class]
    outputs = outputs.reshape(-1,outputs.shape[-1])
    labels = labels.reshape(-1) # [batch * h * w,]
    weights = torch.tensor([1,2],dtype=torch.float32,device=device)
    loss = nn.CrossEntropyLoss(ignore_index=255,weight=weights)
    l = loss(outputs,labels)
    return l

# 2.Dice损失函数:1.创建Dice_target 2. 获得每个类的所有batch的Dice系数均值 3. 获得每个类的每个batch的Dice系数均值
def build_target(label,ignore_index,num_classes):
    # label.shape=[batch,h,w],输出的结果应当是[batch,num_class,h,w]
    label = label.clone()
    mask_flag = (label == ignore_index) # 记录原先像素值为255的地方
    label[mask_flag] = 0 # 将原先像素值为255的地方的像素值转换为0
    dice_target = F.one_hot(label,num_classes).float() # [batch,h,w,num_class]
    dice_target[mask_flag] = ignore_index #
    return dice_target.permute(0,3,1,2) # [batch,num_class,h,w]

def dice_loss(model_out,label,ignore_index,num_classes):
    pred = F.softmax(model_out,dim=1)
    dice_target = build_target(label,ignore_index,num_classes) # [batch,num_class,h,w]
    f = multiclass_dice_coeff
    return 1 - f(pred,dice_target,ignore_index)

def multiclass_dice_coeff(x,target,ignore_index, epsilon=1e-6): # 2. 获得每个类的所有batch的Dice系数均值
    # x.shape=[batch,num_class,h,w],target.shape=[batch,num_class,h,w]
    dice_value = 0
    for i in range(x.shape[1]):
        class_x = x[:,i,:,:]
        class_target = target[:,i,:,:]
        dice_value += dice_coeff(class_x, class_target, ignore_index, epsilon)
    return dice_value / x.shape[1]

def dice_coeff(x, target, ignore_index, epsilon=1e-6): # 3. 获得每个类的每个batch的Dice系数均值
    # x.shape=target.shape=[batch,h,w]
    dice_value = 0
    batch = x.shape[0]
    for i in range(batch):
        now_x = x[i].reshape(-1) # [h * w,]
        now_target = target[i].reshape(-1) # [h * w,]
        # 需要除去标签值为255的部分
        mask_flag = (now_target != ignore_index)
        now_x = now_x[mask_flag]
        now_target = now_target[mask_flag]
        # 计算每个类的每个样本的dice系数
        numerator = 2 * torch.dot(now_x.type(now_target.dtype),now_target) # 分子
        denominator = now_x.sum() + now_target.sum() # 分母部分
        if denominator == 0:
            denominator = numerator
        dice_value += (numerator + epsilon) / (denominator + epsilon)
    return dice_value / batch

def sum_loss(outputs,labels,ignore_index,num_classes,device):
    return dice_loss(outputs,labels,ignore_index,num_classes) + Focal_Loss(outputs, labels, device)


if __name__ == '__main__':
    device = torch.device('cpu')
    torch.manual_seed(1)
    ignore_index = 255
    num_classes = 2
    label = torch.tensor([[[255,1,255],[1,1,0],[255,0,255]]])
    model_out = torch.randn(size=(1,2,3,3))
    l = dice_loss(model_out,label,ignore_index,num_classes)
    print(l)

    l2 = Focal_Loss(model_out, label, device)
    print(l2)



