# yolo v1 算法的改进版本
# 1.将原先的vgg作为backbone替换成了resnet作为backbone
# 2.经过backbone的特征向量在原先的yolo v1中是利用全连接层进行变换后得到的，此时利用spp和一维卷积获取预测内容
# 3.原先的yolo v1中每个grid_cell中产生B个预测框，改进后只有一个预测框
# 4.针对于标签框的大小基于了不同的权重(此时的关于框的宽高的的MSELoss由之前的(sqrt(x1)-sqrt(x2))^2 + (sqrt(x1)-sqrt(x2))^2)变为了
# (log(x1)-log(x2))^2 + (log(x1)-log(x2))^2)
# 针对于标签框的中心点的相对坐标造成的损失由之前的MSELoss变为了BCELoss

# 5.针对于的标签框的中心坐标和预测的中心坐标之间利用了BCELoss(为什么？)


import os

import torch
from thop import profile
import transform
import VOC07
from torch.utils import data
import models
from copy import deepcopy
import macher
import time
import eval2

# 此时已经验证所有的子文件都是正确的，迭代10次得到的结果为map=44.16%(在1.py中利用原来正确的main函数)
# 该文件出现了问题(最终发现当前文件对于验证集仍然使用了数据增强)，进行修改后得到 10次跌打map=31.1 10次跌打map=39.44

class Config(object):
    def __init__(self):
        self.val_size = 416
        self.train_size = 416
        self.root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
        self.batch_size = 16
        self.accu = 4
        self.device = torch.device('cuda')
        self.lr = 0.001
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.max_epoch = 150
        self.lr_epoch = [90,120]
        self.start_epoch = 0
        self.no_warm_up = False
        self.wp_epoch = 1
        self.eval_epoch = 5



def build_dataset(root, train_size):
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    train_transform = transform.Augmentation(train_size,pixel_mean,pixel_std)
    train_dataset = VOC07.VOCDetection(root, train_size, transform=train_transform)
    num_classes = 20
    return train_dataset,num_classes

def FLOPs_and_Params(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x, ))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))


def detection_collate(batch):
    # 此时输入的应当是一个列表，列表的元素为一个样本中的内容
    targets = []
    images = []
    for sample in batch:
        images.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(images,dim=0),targets

if __name__ == '__main__':
    config = Config()
    # 制作数据集
    train_dataset,num_classes = build_dataset(config.root, config.train_size) # 此时对于不同的样本上的检测目标的个数是不相同的
    # 此时train_loader中的其中一个元素为[[num_obj1,5], [num_obj2,5], ....,[num_obj3,5]]
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                   collate_fn=detection_collate)
    # 创建一个验证集数据，并且求解其MAP值
    pixel_mean = (0.406, 0.456, 0.485)  # BGR
    pixel_std = (0.225, 0.224, 0.229)   # BGR
    val_transform = transform.BaseTransform(config.val_size, pixel_mean, pixel_std)  # 此时应当是BaseTransform而不是Augmentation
    evaluator = eval2.VOCAPIEvaluator(config.root,config.val_size,device=config.device,transform=val_transform)
    # 创建一个模型
    model = models.myYOLO(config.device, config.train_size, num_classes,trainable=True)
    model.to(config.device).train()
    # 计算模型的浮点运算数量
    # 计算模型的FLOPs和参数量
    model_copy = deepcopy(model)
    model_copy.trainable = False
    model_copy.eval()
    # FLOPs_and_Params计算整个模型的浮点运算量的情况下必须要有前向传播的过程以及模型的架构
    # 优于之前设置了model_copy.trainable = False、 model_copy.eval()，因此此时的前向传播需要走trainable = False的路径
    FLOPs_and_Params(model=model_copy,
                        img_size=config.val_size,
                        device=config.device)
    del model_copy
    # 构建优化器
    base_lr = config.lr
    tmp_lr = base_lr
    optimizer = torch.optim.SGD(model.parameters(),lr=config.lr,
                          momentum=config.momentum,weight_decay=config.weight_decay)
    max_epoch = config.max_epoch                 # 最大训练轮次
    lr_epoch = config.lr_epoch
    epoch_size = len(train_loader)  # 每一训练轮次的迭代次数
    # 开始训练
    best_map = -1
    t0 = time.time()
    for epoch in range(config.start_epoch, max_epoch):
        # 使用warm_up策略来控制早期的学习率
        if epoch in config.lr_epoch:
            # 进行学习率的更新
            tmp_lr = 0.1 * tmp_lr
            optimizer.param_groups[0]['lr'] = tmp_lr

        for iter_i, (x, targets) in enumerate(train_loader): # 此时使用warm_up策略
            ni = iter_i + epoch * epoch_size
            if not config.no_warm_up: # 使用warm_up策略
                if epoch < config.wp_epoch: # 学习率每次迭代都会进行更新
                    n_w = config.wp_epoch * epoch_size
                    tmp_lr = base_lr * pow(ni * 1. / n_w, 4)
                    optimizer.param_groups[0]['lr'] = tmp_lr
                elif epoch == config.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    optimizer.param_groups[0]['lr'] = tmp_lr

            # 制作标签
            targets = [label.tolist() for label in targets]
            targets = macher.gt_creator(input_size=config.train_size,stride=model.stride,
                                        label_lists=targets) # [batch, H * W, 7]
            # 计算出训练的损失
            targets = targets.to(config.device)
            x = x.type(torch.float32)
            x = x.to(config.device)
            conf_loss, cls_loss, bbox_loss, total_loss = model(x, targets) # 求解出损失
            total_loss /= config.accu
            total_loss.backward()
            if ni % config.accu == 0:
                optimizer.step()
                optimizer.zero_grad()
            # 用于记录当前的损失
            if iter_i % 20 == 0:
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, max_epoch, iter_i, epoch_size, tmp_lr,
                         conf_loss.item(),
                         cls_loss.item(),
                         bbox_loss.item(),
                         total_loss.item(),
                         config.train_size,
                         t1 - t0)) # 进行一次的参数更新所消耗的时间
                t0 = time.time()

        # 对其进行验证(每一轮迭代对其进行一次验证，并存储MAP提升的模型)
        if epoch % config.eval_epoch == 0 or (epoch + 1) == max_epoch:
            model.eval()
            model.trainable = False
            model.set_grid(config.val_size)
            evaluator.evaluate(model)
            # 为之后的训练做准备
            model.trainable = True
            model.set_grid(config.train_size)
            model.train()
            cur_map = evaluator.map
            if best_map < cur_map:
                best_map = cur_map
                print('Saving state, epoch:', epoch + 1)
                weight_name = 'epoch_{}_{:.1f}.pth'.format(epoch + 1, best_map * 100)
                checkpoint_path = os.path.join('checkpoint',weight_name)
                torch.save(model.state_dict(), checkpoint_path)












