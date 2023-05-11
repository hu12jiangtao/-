import torchvision
from torch import nn
import torch
import function
from torch.nn import functional as F
import os
import model

print(1)
class Config(object):
    def __init__(self):
        self.train_batch_size = 128
        self.test_batch_size = 32
        self.seed = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_epoch = 30
        self.momentum = 0.9
        self.lr = 0.01  #
        self.distill_save_path = 'param/distill.pkl'
        self.teacher_save_path = 'param/resnet18.pkl'
        self.data_dir = 'dataset'
        self.T = 10
        self.alpha = 0.9


def distill_loss(teacher_y_hat,student_y_hat,label,alpha,T):
    loss1 = nn.CrossEntropyLoss()(student_y_hat,label)
    loss2 = nn.KLDivLoss()(F.log_softmax(student_y_hat / T, dim=-1),F.softmax(teacher_y_hat / T, dim=-1)) * T * T * 2.0
    return (1 - alpha) * loss1 + alpha * loss2

def distill_epoch_train(epoch,student_net,teacher_net,train_loader,test_loader,loss,opt,config): # 用测试集验证时只用学生网络
    student_net.train()
    metric = function.add_machine(3)
    for index,(X,y) in enumerate(train_loader):
        X,y = X.to(config.device),y.to(config.device)
        teacher_y_hat = teacher_net(X).detach()
        student_y_hat = student_net(X)
        l = loss(teacher_y_hat,student_y_hat,y,config.alpha,config.T)
        opt.zero_grad()
        l.backward()
        opt.step()
        metric.add(l * y.numel(), function.accuracy(y,student_y_hat), y.numel())
    test_acc = function.evaluate(student_net, test_loader, config.device)
    print(f'[epoch:{epoch + 1} \t train_loss:{metric[0] / metric[2]:1.3f} \t train_acc:{metric[1] / metric[2]:1.3f} \t test_acc:{test_acc}')


if __name__ == '__main__':
    config = Config()
    torch.manual_seed(config.seed) # 固定网络的初始权重
    torch.manual_seed(config.seed)
    # 加载数据集
    train_loader, test_loader = function.load_cifir_data(config.data_dir,config.train_batch_size,config.test_batch_size)
    # 加载教师模型
    teacher_net = torchvision.models.resnet18(pretrained=True)
    teacher_net.fc = nn.Linear(512, 10)
    teacher_net.to(config.device)
    teacher_net.load_state_dict(torch.load(config.teacher_save_path))
    # 加载学生网络
    student_net = model.Conv_net()
    student_net.to(config.device)
    # 进行训练
    if not os.path.exists(config.distill_save_path):
        loss = distill_loss
        opt = torch.optim.SGD(student_net.parameters(),momentum=config.momentum,lr=config.lr)
        for epoch in range(config.num_epoch):
            distill_epoch_train(epoch,student_net,teacher_net,train_loader,test_loader,loss,opt,config)
            torch.save(student_net.state_dict(),config.distill_save_path)
    else:
        student_net.load_state_dict(torch.load(config.distill_save_path))

    finally_train_acc = function.evaluate(student_net, train_loader, config.device)
    finally_test_acc = function.evaluate(student_net,test_loader,config.device)
    print(finally_train_acc)  # 0.876
    print(finally_test_acc)  # 0.8012
