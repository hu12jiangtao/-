# 经过20轮迭代后的map的值变为了57.3%

import torch
from torch.utils import data
import VOC07
from thop import profile
import transforms
import eval
import models
from copy import deepcopy
import matchor
import numpy as np
from torch.nn import functional as F
import time
import os

class Config(object):
    def __init__(self):
        self.root = 'D:\\python\\pytorch作业\\计算机视觉\\data\\VOCdevkit'
        self.pixel_mean = (0.406, 0.456, 0.485)
        self.pixel_std = (0.225, 0.224, 0.229)
        self.train_size = 640
        self.val_size = 416
        self.device = torch.device('cuda')
        self.batch_size = 4
        self.accumulate = 16
        self.anchor_size = [[1.19, 1.98], [2.79, 4.59], [4.53, 8.92], [8.06, 5.29], [10.32, 10.65]]
        self.lr = 0.001
        self.lr_epoch = [100, 150]
        self.num_epochs = 200
        self.warm_epoch = 1
        self.multi_scale = True
        self.stride = 32 # 每个grid_cell的宽高
        self.eval_epoch = 5



def load_dataset(root, train_size, val_size, mean, std, device):
    train_transform = transforms.Augmentation(train_size, mean, std)
    val_transform = transforms.BaseTransform(val_size, mean, std)
    # 导入训练集的数据集
    train_dataset = VOC07.VOCDetection(
                                root=root,
                                img_size=train_size,
                                image_sets=[('2007', 'trainval')],
                                transform=train_transform
                                )
    # 导入进行验证的数据集
    val_dataset = eval.VOCAPIEvaluator(data_root=root,
                                       img_size=val_size,
                                       device=device,
                                       transform=val_transform)
    return train_dataset, val_dataset

def collect_fn(batch):
    # 图片上为一个列表[[3,H,W],...,[3,H,W]], 标签应为[[n_objs1,5],...,[n_obj_k,5]]
    images = []
    targets = []
    for i in batch:
        images.append(i[0])
        targets.append(torch.tensor(i[1],dtype=torch.float32))
    return torch.stack(images,dim=0), targets

def FLOPs_and_Params(model,img_size,device):
    x = torch.randn(size=(1, 3, img_size, img_size),device=device)
    model.to(device)
    print('==============================')
    flops, params = profile(model, inputs=(x,))
    print('==============================')
    print('FLOPs : {:.2f} B'.format(flops / 1e9))
    print('Params : {:.2f} M'.format(params / 1e6))


if __name__ == '__main__':
    config = Config()
    # 导入训练集数据以及测试集数据
    train_size = config.train_size
    val_size = config.val_size
    train_dataset, val_dataset = load_dataset(config.root, train_size, val_size,
                                              mean=config.pixel_mean, std=config.pixel_std, device=config.device)
    train_loader = data.DataLoader(train_dataset,batch_size=config.batch_size,shuffle=True,collate_fn=collect_fn)
    # 导入模型
    model = models.YOLOv2(
                          device=config.device,
                          input_size=config.train_size,
                          num_classes=20,
                          trainable=True,
                          topk=100,
                          anchor_size=config.anchor_size
                          ).to(config.device)
    model.train()
    # 求解计算复杂度以及参数的个数
    model_copy = deepcopy(model)
    model_copy.eval()
    model_copy.trainable = False
    FLOPs_and_Params(model=model_copy,
                     img_size=val_size,
                     device=config.device)
    del model_copy
    # 求解优化方式
    opt = torch.optim.SGD(model.parameters(),lr=config.batch_size, momentum=0.9, weight_decay=5e-4)
    epoch_size = len(train_loader)
    base_lr = config.lr
    tmp_lr = config.lr
    # 开始进行训练(warm-up,利用多尺度的融合)
    best_map = -1.
    t0 = time.time()
    for epoch in range(config.num_epochs):
        if epoch in config.lr_epoch: # 进行梯度的衰减
            tmp_lr = 0.1 * tmp_lr
            opt.param_groups[0]['lr'] = tmp_lr
        for iter_i, (img, tg) in enumerate(train_loader):
            ni = epoch_size * epoch + iter_i
            if epoch < config.warm_epoch:
                nw = config.warm_epoch * epoch_size
                tmp_lr = base_lr * pow((ni)*1. / (nw), 4)
                opt.param_groups[0]['lr'] = tmp_lr
            elif epoch == config.warm_epoch and iter_i == 0:
                tmp_lr = base_lr
                opt.param_groups[0]['lr'] = tmp_lr
            # 利用多尺度的方式来提升准确率
            if config.multi_scale and iter_i > 0 and iter_i % 10 == 0:  # 每十次迭代改变一次输入图片的宽高
                train_size = np.random.randint(10, 13) * config.stride
                model.set_grid(train_size)
            if config.multi_scale:
                img = F.interpolate(img, train_size, mode='bilinear', align_corners=False)
            # 对targets进行处理
            tg = [label.tolist() for label in tg]
            tg = matchor.gt_creator(train_size, config.stride, tg, config.anchor_size, ignore_thresh=0.5)
            tg = torch.from_numpy(tg)
            img = img.type(torch.float32)
            # 进行训练和迭代
            img, tg = img.to(config.device), tg.to(config.device)
            conf_loss, cls_loss, bbox_loss, total_loss = model(img, tg)
            total_loss /= config.accumulate
            total_loss.backward()
            if ni % config.accumulate == 0: # 16次迭代的梯度进行累加后进行梯度的更新
                opt.step()
                opt.zero_grad()

            if iter_i % 10 == 0: # 打印出此次迭代的损失
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                      '[Loss: obj %.2f || cls %.2f || bbox %.2f || total %.2f || size %d || time: %.2f]'
                      % (epoch + 1, config.num_epochs, iter_i, epoch_size, tmp_lr,
                         conf_loss.item(),
                         cls_loss.item(),
                         bbox_loss.item(),
                         total_loss.item(),
                         train_size, t1 - t0),
                      flush=True)
                t0 = time.time()
        # 进行验证
        if epoch % config.eval_epoch == 0 or (epoch + 1) == config.num_epochs:
            model.eval()
            model.trainable = False
            model.set_grid(val_size)
            val_dataset.evaluate(model)
            cur_map = val_dataset.map
            # 验证后将其回复为训练的状态
            model.train()
            model.trainable = True
            model.set_grid(train_size)
            if cur_map > best_map:
                best_map = cur_map
                print('Saving state, epoch:', epoch + 1)
                weight_name = 'epoch_{}_{:.1f}.pth'.format(epoch + 1, best_map*100)
                checkpoint_path = os.path.join('checkpoints', weight_name)
                torch.save(model.state_dict(), checkpoint_path)








