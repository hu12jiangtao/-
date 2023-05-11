import torchvision.transforms as T
import torch

def _data_transforms_cifar10():
    mean = [0.49139968, 0.48215827, 0.44653124]
    std = [0.24703233, 0.24348505, 0.26158768]
    train_transform = T.Compose([T.RandomCrop(32,padding=4),T.RandomHorizontalFlip(),
                                 T.ToTensor(),T.Normalize(mean=mean,std=std)])
    test_transform = T.Compose([T.ToTensor(),T.Normalize(mean=mean,std=std)])
    return train_transform,test_transform

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.avg = 0
        self.cnt = 0

    def update(self,val,n=1):
        self.cnt += n
        self.sum += val * n
        self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    # output = [batch, 10], target = [batch, ]
    res = []
    batch_size = output.shape[0]
    max_len = max(topk)
    _, index = torch.topk(output, max_len, dim=1) # [batch, 5]
    index = index.permute(1,0) # [5, batch]
    cmp = torch.eq(index, target.reshape(1,-1).expand_as(index)) # [5, batch]
    for i in topk:
        current_cmp = cmp[:i].reshape(-1).float().sum(0)
        acc = 100 * current_cmp / batch_size
        res.append(acc)
    return res


