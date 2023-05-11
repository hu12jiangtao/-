import torch
from torch import nn

def create_lr_scheduler(optimizer,num_step,epochs,warmup=True,warmup_epochs=1,warmup_factor=1e-3):
    # 使用deeplabv2中的ploy学习率衰减函数
    # warmup代表在使用ploy学习率衰减函数的前面是否使用其他的学习率衰减策略
    assert num_step > 0 and epochs > 0
    if warmup is not True:
        warmup_epochs = 0
    def f(x): # x为样本迭代次数，而不是批量迭代次数
        if warmup is True and x < warmup_epochs * num_step:
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9
    return torch.optim.lr_scheduler.LambdaLR(optimizer,f)

def train_one_epoch(opt, lr_scheduler):
    lr_scheduler.step()
    return opt.param_groups[0]['lr']



net = nn.Linear(4,3)
opt = torch.optim.Adam(net.parameters(),lr=0.001)
lr_scheduler = create_lr_scheduler(opt,num_step=5,epochs=20,warmup=True,warmup_epochs=1,warmup_factor=1e-3)
for i in range(20):
    lr = train_one_epoch(opt, lr_scheduler)
    print(lr)