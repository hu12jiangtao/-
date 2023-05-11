# 使用warm_up策略的目标是防止开始的时候迭代过猛在成最后的准确率的下降
from torch import nn
import torch
from torch.utils import data

warm_up_epoch = 5
total_epoch = 100
epoch_change = [40,80]
warm_up_flag = True
net = nn.Linear(4,3)
opt = torch.optim.SGD(net.parameters(),lr=0.1,momentum=0.9,weight_decay=5e-4)
base_lr = 0.1
tmp_lr = 0.1
x = torch.randn(size=(100,4))
dataset = data.TensorDataset(x)
loader = data.DataLoader(dataset,batch_size=10,shuffle=True)
each_epoch_iter = len(loader)

for epoch in range(total_epoch):
    if epoch in epoch_change:
        tmp_lr = tmp_lr * 0.1
        opt.param_groups[0]['lr'] = tmp_lr

    for idx, x in enumerate(loader):
        if warm_up_flag:
            now_step = each_epoch_iter * epoch + idx
            total_warm_step = each_epoch_iter * warm_up_epoch
            if epoch < warm_up_epoch:
                tmp_lr = pow(now_step / total_warm_step, 4)
                opt.param_groups[0]['lr'] = tmp_lr
            elif epoch == warm_up_epoch and idx == 0:
                tmp_lr = base_lr
                opt.param_groups[0]['lr'] = tmp_lr

    print(opt.param_groups[0]['lr'])

