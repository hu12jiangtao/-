import os
import time
import torch
from torch.utils import data
from mingpt.utils import CfgNode as CN
from torch import nn
from collections import defaultdict

class Trainer(object):
    @staticmethod
    def get_default_config():
        C = CN()
        # dataloder parameters
        C.device = 'auto'
        # optimizer parameters
        C.max_iters = 10000
        C.batch_size = 64
        C.learning_rate = 3e-4
        C.betas = (0.9, 0.95)
        C.weight_decay = 0.1 # only applied on matmul weights
        C.grad_norm_clip = 1.0
        C.work_dir = './out'
        return C

    def __init__(self,config,model,train_dataset):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset

        if config.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = config.device
        self.model.to(self.device)
        print("running on device", self.device)

        self.iter_num = 0 # 用于记录iter迭代的个数
        self.iter_time = 0.0 # 用于记录每个iter迭代开始的时间
        self.iter_dt = 0.0 # 用于记录每个iter迭代的时间
        self.optimizer = None
        self.callbacks = defaultdict(list)

    def grad_clipping(self,net,limit_num=1):
        if isinstance(net,nn.Module):
            params = [param for param in net.parameters() if param.requires_grad is True]
            norm = torch.sqrt(sum([torch.sum(p.grad ** 2) for p in params]))
            if norm > limit_num:
                for param in params:
                    param.grad[:] *= limit_num / norm

    def run(self):
        model, config = self.model, self.config
        self.optimizer = model.configure_optimizers(config) # 此时输入config为trainer的config
        # 用于训练的数据集
        train_loader = data.DataLoader(self.train_dataset,self.config.batch_size,shuffle=True)
        model.train()
        self.iter_num = 0
        self.iter_time = time.time()
        flag = 1
        while True:
            for x,y in train_loader:
                x,y = x.to(self.device),y.to(self.device)
                logits, loss = model(x,y)
                self.optimizer.zero_grad()
                loss.backward()
                # 进行梯度的枝剪
                self.grad_clipping(model, limit_num=1)
                self.optimizer.step()
                # 进行验证和打印
                if self.iter_num % 100 == 0:
                    print(f"iter_dt {self.iter_dt:.2f}s; iter {self.iter_num}: train loss {loss.item():.5f}")
                if self.iter_num % 500 == 0:
                    model.eval()
                    with torch.no_grad():
                        context = 'O God, O God!'
                        x = torch.tensor([self.train_dataset.token_to_idx[char] for char in context]
                                         ,dtype=torch.long).reshape(1,-1).to(self.device)
                        y = model.generate(x, 50, temperature=1.0, do_sample=True, top_k=10)[0]
                        # 取出这个生成的文本,y.shape = [给出的序列长度 + 需要预测的序列长度,],之后需要将序列转换为字符同时打印出来
                        completion = ''.join([self.train_dataset.idx_to_token[int(i)] for i in y])
                        print(completion)
                    print('saving model')
                    ckpt_path = os.path.join(config.work_dir,'model_10000.pt')
                    torch.save(model.state_dict(),ckpt_path)
                    model.train()
                # 为之后的迭代进行准备
                self.iter_num += 1
                tnow = time.time()
                self.iter_dt = tnow - self.iter_time
                self.iter_time = tnow
                if config.max_iters is not None and self.iter_num >= config.max_iters:
                    flag = 0
                    break
            if flag == 0:
                break




if __name__ == '__main__':
    a = time.time()
    time.sleep(1)
    b = time.time()
    print(b - a)