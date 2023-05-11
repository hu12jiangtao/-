# 实现互相关操作
# 对于经典的神经网络例如Vgg，LeNet一般不会修改卷积的参数，一般只会修改通道数，同时通道数需要控制好，太多会overfitting，一般在1024以下
import torch
from torch import nn

def corr_2d(X,K): #对于单通道的卷积输入输出，输入，输出都是二维的,此时加入了padding和stride
    x_h,x_w=X.shape
    k_h,k_w=K.shape
    Y=torch.zeros(x_h-k_h+1,x_w-k_w+1)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[i:k_h+i,j:j+k_w]*K).sum()
    return Y
#实例:训练得到检测图像颜色的边缘检测器（此时为一个样本单通道的灰白图片）
#已知这个过滤器为[[1,-1]]
X=torch.ones((6,8))
X[:,2:6]=0
K=torch.tensor([[1,-1]])
print(X)
Y=corr_2d(X,K)
print(Y)

#利用神经网络进行训练得到这个kenel
#kernel_size也被称为感受野，一般的卷积神经网络都是感受野小一点，网络深一点比较好（卷积层多一点），效果比感受野大一点，网络浅一点模型好
conv2d=nn.Conv2d(1,1,kernel_size=(1,2),bias=False)  # 第一个为输入通道为1，第二个1代表kenel的个数
X=torch.reshape(X,(1,1,6,8))
Y=torch.reshape(Y,(1,1,6,7))
for i in range(50):
    Y_hat=conv2d(X)
    loss=0.5*(Y_hat-Y)**2
    conv2d.zero_grad()  #模板中net.zero_grad(),等价与trainer.zero_grad()
    loss.sum().backward()
    conv2d.weight.data-=0.03*conv2d.weight.grad
    print(loss.sum())
    print(conv2d.weight.data)  #训练得到的核的训练参数
    print('*'*50)