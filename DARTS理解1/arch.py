import torch

def concat(params): # 将所有的
    return torch.cat([param.reshape(-1) for param in params],dim=0)

class Arch(object):
    def __init__(self,model,config):
        # 该函数是用于更新架构参数的，因此输入的内容应当是super_net中的_arch_parameters
        self.momentum = config.momentum # 训练集上用于更新操作参数的momentum
        self.wd = config.wd # 训练集上用于更新操作参数的weight_decay
        self.model = model # 整一个network的模型(其中包含了 操作参数 和 架构参数)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=config.arch_lr,
                                          betas=(0.5,0.999),weight_decay=config.arch_wd) # 用于架构参数的更新

    def step(self, x_train, target_train, x_valid, target_valid, eta, optimizer):
        self.optimizer.zero_grad()
        self.backward_step_unrolled(x_train, target_train, x_valid, target_valid, eta, optimizer) # 对架构参数进行更新
        self.optimizer.step()

    def backward_step_unrolled(self,x_train, target_train, x_valid, target_valid, eta, optimizer):
        # 此时的eta, optimizer分别代表着用于操作参数更新的学习率和优化方式
        # 第一步为求解最优的操作参数(类似于meta_learning，进行一次的训练集的迭代),此时得到操作参数w'
        unrolled_model = self.comp_unrolled_model(x_train, target_train, eta, optimizer)
        # 第二部计算当前操作参数 和 架构参数下 验证集上的损失
        unrolled_loss = unrolled_model.loss(x_valid,target_valid)
        # 第三步计算unrolled_loss关于架构参数a和w'的梯度
        unrolled_loss.backward()
        d_alpha = [v.grad for v in unrolled_model.arch_parameters()] # a的梯度 ▼a L_val(w',a)
        vector = [v.grad for v in unrolled_model.parameters()] # w'的梯度, ▼w' L_val(w',a)
        # 第四步 求解出[▼α L_train(w+,a+) - ▼α L_train(w-,a-)] / 2ε
        implicit_grads = self.hessian_vector_product(vector, x_train, target_train)
        # 第五步 得到验证集上的梯度 ▼a L_val(w',a) - eta * {[▼α L_train(w+,a+) - ▼α L_train(w-,a-)] / 2ε}
        for g, ig in zip(d_alpha, implicit_grads):
            g.data.sub_(eta, ig)
        # 第六步 将用于更新的梯度赋值给网络中去
        for v, g in zip(self.model.arch_parameters(),d_alpha):
            if v.grad is None:
                v.grad = g
            else:
                v.grad.data.copy_(g.data)

    def comp_unrolled_model(self, x, target, eta, optimizer): # 进行一次的训练集的迭代，并且更新操作参数
        # x, target是训练集的输入和标签, eta, optimizer为操作参数的学习率和优化方式
        loss = self.model.loss(x,target)
        theta = concat(self.model.parameters()).detach() # 将当前的模型参数拉成一条直线并且concat起来
        try:
            moment = concat(optimizer.state[v]['momentum_buffer'] for v in self.model.parameters())
            moment.mul_(self.momentum)
        except:
            moment = torch.zeros_like(theta)
        # 利用反向传播求解梯度
        d_theta = concat(torch.autograd.grad(loss,self.model.parameters())).data # 求解所有训练参数并将其展开为一个一维向量(操作参数和架构参数)
        # 进行super_net的参数更新
        # theta = theta - eta * (moment + weight decay * d_theta + d_theta)
        theta = theta.sub_(eta, moment + d_theta + self.wd * theta)
        # 将更新后的参数按照原来模型参数的形状排列并且更新
        unrolled_model = self.construct_model_from_theta(theta)
        return unrolled_model

    def construct_model_from_theta(self,theta):
        model_new = self.model.new()
        # 创建了一个新的network，此时的存储架构参数的列表_arch_parameters保持，其余参数重置为初始值
        # 原因是此时的self._arch_parameters是一个列表，不包含在self.model.state_dict()
        # 此时model_dict和self.model.state_dict()无关系，model_dict改变self.model.state_dict()并不会改变
        model_dice = self.model.state_dict() # 所有的操作参数和架构参数
        params, offset = {}, 0
        for name,param in self.model.named_parameters():
            param_length = param.numel()
            params[name] = theta[offset : offset + param_length].reshape(param.shape)
            offset += param_length
        assert offset == len(theta)
        model_dice.update(params)
        model_new.load_state_dict(model_dice) # 此时将network中的所有的 操作参数 和 权重参数 进行更新
        return model_new.cuda()

    def hessian_vector_product(self, vector, x, target, r=1e-2):
        # 求解[▼α L_train(w+,a+) - ▼α L_train(w-,a-)] / 2ε
        epsilon = r / concat(vector).norm() # ε = 0.01 / || ▼w' L_val(w',a) ||
        # 求解w+，其公式为: w + ε * ▼w' L_val(w',a)
        for p, v in zip(self.model.parameters(),vector):
            p.data.add_(epsilon, v) # 对当前的model的操作参数和架构参数进行了修改,此时必须把常熟放在前面，矩阵放在后面，不然会报错
        # 求解▼α L_train(w+,a+)
        loss = self.model.loss(x, target)
        grads_p = torch.autograd.grad(loss, self.model.arch_parameters())
        # 求解w-，其公式为: w - ε * ▼w' L_val(w',a)
        for p, v in zip(self.model.parameters(),vector):
            p.data.sub_(2 * epsilon, v)
        loss = self.model.loss(x, target)
        # 求解▼α L_train(w-,a-)
        grads_n = torch.autograd.grad(loss, self.model.arch_parameters())
        # 求解[▼α L_train(w+,a+) - ▼α L_train(w-,a-)] / 2ε
        h = [(x - y) / (2 * epsilon) for x,y in zip(grads_p, grads_n)]
        # 此时需要保证原来的网络参数并没有发生变化
        for p,v in zip(self.model.parameters(),vector):
            p.data.add_(epsilon, v)
        return h









