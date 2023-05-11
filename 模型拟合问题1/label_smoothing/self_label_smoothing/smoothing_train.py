# 此时通过创建一个类来定义label_smoothing的损失函数
# 输入的应当是自己定义的权重alpha以及训练集的target和image
# 其余的超参数和no_smoothing中的应该保持相同
import os
import torch
from torch import nn
import torchvision
import function
from torch.nn import functional as F
import resnet

# 下面是利用了label smoothing,虽然在验证中利用了交叉熵损失函数、训练中利用了自定义的label smoothing函数
# 但是随着训练时的label smoothing损失的下降,在验证集中的交叉熵损失一样下降
# 该结论从逻辑回归的代码文件中得到
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()

    def forward(self,y_hat,labels,smoothing):
        final_out = torch.log_softmax(y_hat,dim=-1)
        smoothing_loss = torch.mean(final_out,dim=-1) * smoothing  # [batch,]
        nll_out = torch.gather(final_out,index=labels.reshape(-1,1),dim=-1)
        nll_out = nll_out.squeeze(1)
        nll_loss = (1 - smoothing) * nll_out
        sum_loss = nll_loss + smoothing_loss
        return - torch.mean(sum_loss)

'''
# 另一种方法为使用label的one-hot形式，这样就避免了gather函数的使用
def Label_Smoothing2(pred,labels,smoothing):
    # Label_Smoothing2相当于Label_Smoothing1在二维时的情况(Label_Smoothing1)为所有维度的通式
    # 首先将labels进行one-hot处理
    labels = F.one_hot(labels,63) # [5,63]
    print(smoothing * torch.mean(pred,dim=-1))
    print(smoothing / pred.shape[1] * torch.sum(pred,dim=-1))
    loss = (1 - smoothing) * torch.sum(pred * labels,dim=-1) + \
           smoothing * torch.mean(pred,dim=-1)
    loss = -torch.mean(loss,dim=-1)
    return loss
'''


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
        self.model_save_dir = 'params/smoothing.pkl'
        self.loss_mode = 1  # 代表的是平滑化的损失函数
        self.smoothing = 0.1 # epsilon的值


if __name__ == '__main__':
    config = Config()
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1)
    # 导入数据集(train:[256,3,32,32],test:[100,3,32,32])
    train_loader, test_loader = function.load_cifir_data(config.data_dir,config.train_batch_size,config.test_batch_size)
    # 导入模型(利用的是resnet函数中的模型，和torchvision中存在着不同)
    '''
    net = torchvision.models.resnet18(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, config.num_classes)
    net.apply(function.init_params)
    net.to(config.device)
    '''
    net = resnet.ResNet18()
    net = net.to(config.device)
    # 进行训练
    if not os.path.exists(config.model_save_dir):
        loss = LabelSmoothingCrossEntropy()
        opt = torch.optim.SGD(net.parameters(),lr=config.lr,momentum=config.momentum,weight_decay=1e-4)
        schedular = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30,60,90])  # 在第30、60、90处学习率下降10倍
        for epoch in range(config.num_config):
            function.train_epoch(epoch,net,train_loader,test_loader,loss,opt,config)
            schedular.step()
            torch.save(net.state_dict(),config.model_save_dir)
    else:
        net.load_state_dict(torch.load(config.model_save_dir))

    train_acc = function.evaluate(net, train_loader, config.device) # 0.994
    test_acc = function.evaluate(net, test_loader, config.device) # 0.9209
    print(train_acc,test_acc)
