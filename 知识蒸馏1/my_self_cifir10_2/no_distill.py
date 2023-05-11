# 教师网络已经训练好了训练集的准确率在0.999，测试集的准确率在0.9548
# 学生网络此处使用的是Alexnet网络,教师网络利用的是resnet18，两者参数相差5.23
from torch import nn
import torch
import function
import torchvision
import os
import model
class Config(object):
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_batch_size = 128
        self.test_batch_size = 32
        self.data_dir = 'dataset'
        # 'D:\\python\\pytorch作业\\知识蒸馏\\my_self_cifir10\\dataset'为自己电脑上,'dataset'为云上训练的
        self.momentum = 0.9
        self.lr = 0.01
        self.num_epoch = 10
        self.model_save_dir = f'param/AlexNet1.pkl'


if __name__ == '__main__':
    # 创建数据集
    torch.manual_seed(1)
    config = Config()
    torch.backends.cudnn.benchmark = True
    # 导入数据集(train:[256,3,32,32],test:[100,3,32,32])
    train_loader, test_loader = function.load_cifir_data(config.data_dir,config.train_batch_size,config.test_batch_size)
    # 模型
    net = model.Conv_net()
    net.to(config.device)
    net.load_state_dict(torch.load('param/conv.pkl')) # 20个epoch + 加训10个epoch
    param_num = sum([i.numel() for i in net.parameters()])
    print(f'student param_num:{param_num}')
    # 进行训练
    if not os.path.exists(config.model_save_dir):
        print('start training')
        loss = nn.CrossEntropyLoss()
        opt = torch.optim.SGD(net.parameters(), lr=config.lr, momentum=config.momentum)
        for epoch in range(config.num_epoch):
            function.train_epoch(epoch, net, train_loader, test_loader, loss, opt, config)
        torch.save(net.state_dict(), config.model_save_dir)
    else:
        net.load_state_dict(torch.load(config.model_save_dir))

    print('test_acc:',function.evaluate(net,test_loader,config.device))  # 0.7415
    print('train_acc:', function.evaluate(net, train_loader, config.device))  # 0.82054

