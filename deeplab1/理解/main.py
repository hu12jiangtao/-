import torchvision
import torch
from torch import nn
from model import deeplabv3
import config
import transformers
import my_datasets
from torch.utils import data
import train_and_eval

if __name__ == '__main__':
    # 导入模型
    net = deeplabv3.DeepLabV3(config.num_classes).to(config.device)
    # 导入数据集
    train_dataset = my_datasets.VOCSegmentation(config.voc_root, year="2007",
                                                transforms=transformers.get_transform('train'))
    test_dataset = my_datasets.VOCSegmentation(config.voc_root, year='2007',
                                               transforms=transformers.get_transform('test'),txt_name='val.txt')
    train_loader = data.DataLoader(train_dataset,batch_size=config.train_batch_size,shuffle=True,
                                   collate_fn=my_datasets.collect_fn)
    test_loader = data.DataLoader(test_dataset,batch_size=config.test_batch_size,shuffle=False,
                                  collate_fn=my_datasets.collect_fn)
    # 对学习略衰减器进行设计
    opt = torch.optim.Adam(net.parameters(),lr=config.lr,betas=(config.momentum,0.999),weight_decay=config.weight_decay)
    lr_scheduler = train_and_eval.create_lr_scheduler(opt,len(train_loader),config.epochs)
    # 开始进行训练(此时的损失函数就是交叉熵的损失函数)
    for epoch in range(config.epochs):
        train_loss,lr = train_and_eval.train_one_epoch(net, opt, train_loader, config.device, lr_scheduler)
        global_acc, class_acc, iou, mean_iou = train_and_eval.evaluate(net,test_loader,config.device,config.num_classes)
        print(f'epoch:{epoch + 1} train_loss:{train_loss:1.4f} lr:{lr:1.5f} F'
              f'global_acc:{global_acc:1.4f} '
              f'class_acc:{[round(i, 3) for i in class_acc]} '
              f'iou:{[round(i, 3) for i in iou]} '
              f'mean_iou:{mean_iou:1.4f}')






