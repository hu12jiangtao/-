# 池化层的作用是使得神经网络对位置信息的敏感程度变低
# 池化层的输出通道数，应当等于输入通道数
# 池化层在使用的过程中一般不会重叠，就是你的pool_size=stride_size

import torch
from torch import nn
#池化层的代码和卷积层的相类似（首先定义单输入，单输出的情况）
def pool2d(X,pool_size,mode='max',padding=(0,0),stride=(1,1)): # 单样本单通道的输入
    x_h,x_w=X.shape
    p_h,p_w=pool_size
    padding_h,padding_w=padding
    stride_h,stride_w=stride
    Y=torch.zeros(((x_h-p_h+2*padding_h)//stride_h+1,(x_w-p_w+2*padding_w)//stride_w+1),dtype=torch.float32)
    zero_pad=nn.ZeroPad2d(padding=(padding_w,padding_w,padding_h,padding_h))  #padding的顺序为左右上下
    X=zero_pad(X)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode =='max':
                Y[i,j]=X[i*stride_h:i*stride_h+p_h,j*stride_w:j*stride_w+p_w].max()
            elif mode == 'mean':
                Y[i,j]=X[i*stride_h:i*stride_h+p_h,j*stride_w:j*stride_w+p_w].mean()
    return Y   #输出应当是一个二维的

X=torch.arange(25).reshape(5,5)
pool_size=(3,3)
Y=pool2d(X,pool_size,mode='max',stride=(2,2),padding=(1,1))
print(Y)
print('*'*50)
#扩展到多输入上面
def pool_2d_out(X,pool_size,mode='max',padding=(0,0),stride=(1,1)): # 单样本多通道的输入
    return torch.stack([pool2d(x,pool_size,mode=mode,padding=padding,stride=stride)for x in X],0)  #输出应当是一个三维的

X=torch.arange(50).reshape(2,5,5)
pool_size=(3,3)
Y=pool_2d_out(X,pool_size,mode='max',padding=(1,1),stride=(2,2))
print(Y)

#利用模板来实现池化层
X=torch.arange(50,dtype=torch.float32).reshape(1,2,5,5)
pool_2d=nn.MaxPool2d((3,3),padding=(1,1),stride=(2,2))   #此时X的输入为一个四维的值，就是加上了样本的数量
Y=pool_2d(X)
print(Y)
print('*'*50)

def pool_2d_sum(X,pool_size,mode,padding,stride):  #此时X为多样本多通道的输入，输出应当也是四维的(等价于模板中的nn.MaxPool2d)
    return torch.stack([pool_2d_out(x,pool_size,mode=mode,padding=padding,stride=stride) for x in X],0)

X=torch.arange(100).reshape(2,2,5,5)
pool_size=(3,3)
Y=pool_2d_sum(X,pool_size,mode='max',stride=(2,2),padding=(1,1))
print(Y)

