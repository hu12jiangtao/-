# 此时两个模型只存在唯一一个不同的点，一个在训练过程中利用了label_smoothing,另一个没有利用label_smoothing
# 此时选用的数据集为IFAR10，应用的模型为resnet18
import os
import torch
from torch import nn
import torchvision
import function
import resnet

class Config(object):
    def __init__(self):
        self.weight_decay = 1e-4
        self.lr = 0.1
        self.momentum = 0.9
        self.train_batch_size = 256
        self.test_batch_size = 100
        self.data_dir = 'D:\\python\\pytorch作业\\知识蒸馏\\my_self_cifir10_2\\dataset'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 10
        self.lr = 0.1
        self.num_config = 120
        self.model_save_dir = 'params/no_smoothing.pkl'
        self.loss_mode = 0


if __name__ == '__main__':
    torch.manual_seed(1)
    config = Config()
    torch.backends.cudnn.benchmark = True
    # 导入数据集(train:[256,3,32,32],test:[100,3,32,32])
    train_loader, test_loader = function.load_cifir_data(config.data_dir,config.train_batch_size,config.test_batch_size)
    # 导入模型
    net = resnet.ResNet18()
    net = net.to(config.device)
    # 进行训练
    if not os.path.exists(config.model_save_dir):
        loss = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(net.parameters(),lr=config.lr,momentum=config.momentum,weight_decay=1e-4)
        schedular = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30,60,90])  # 在第30、60、90处学习率下降10倍
        for epoch in range(config.num_config):
            function.train_epoch(epoch,net,train_loader,test_loader,loss,opt,config)
            schedular.step()
            torch.save(net.state_dict(),config.model_save_dir)
    else:
        net.load_state_dict(torch.load(config.model_save_dir))

    train_acc = function.evaluate(net, train_loader, config.device) # 0.994
    test_acc = function.evaluate(net, test_loader, config.device) # 0.9171
    print(train_acc,test_acc)




