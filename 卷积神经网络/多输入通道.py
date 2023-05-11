# 多个输入通道和多输出通道，此时输入为 (n_(c-1)*x_h,x_w)，参数应当为(n_c,n_(c-1),k_h,k_w)，输出为(n_c*y_h*y_w)，偏置为（n_c,n_(c-1)），利用广播

# 对于每一个卷积来说都是在提取图片的某一个特征，对于多输出相当于提取了输入内容的多个特征，输入通道相当于对前一个输出的几个特征进行加权组合得到得到一个新的特征
# 变相说明了深度越深识别的内容越复杂，例如对于识别猫的图片，最初的卷积识别出的是一些基本特征（斜线等），
# 随着网络的加深（最后几次）输出可以识别猫的眼睛，胡须等，经过下面的卷积后将这些猫的眼睛，胡须加权变得可以识别猫的整张脸

# 1*1的卷积层相当于与将多通道进行融合，即使全连接层（只进行通道融合，并不识别空间模式（只关注这一个像素点，并不考虑这个像素点和其他点的关系））
# 例如：输入为3*3*3的矩阵，经过2个1*1的卷积核，得到的结果就是2*3*3，
# 相当于输入为(x_h*x_w(样本数),n_c-1（特征个数）)，经过(n_c-1*n_c)的权重后两个神经元上得到（x_h*x_w，n_c）的全连接

#卷积对位置信息十分敏感，输出的一个元素就是只和输入这个元素周围像素的值，输出的（1，1），核为（3，3），只和原输入的X[1:4,1:4]有关

from torch import nn
import torch

'''
#对于内置函数sum为对一个列表或者元组内的元素进行求和
a=torch.ones((3,3),dtype=torch.float32)
b=torch.ones((3,3),dtype=torch.float32)
print(sum((a,b)))
'''

'''
a=torch.ones((3,3),dtype=torch.float32)
b=torch.zeros((3,3),dtype=torch.float32)
print(torch.stack([a,b],dim=0))  #会新增一个维度
print(torch.cat([a,b],dim=0))  #不会新增一个维度
'''

def corr2d(X,K,padding,stride): #对于单通道的卷积输入输出，输入，输出都是二维的,此时加入了padding和stride
    x_h,x_w=X.shape
    k_h,k_w=K.shape
    padding_h,padding_w=padding
    stride_h,stride_w=stride
    Y=torch.zeros(((x_h-k_h+2*padding_h)//stride_h+1,(x_w-k_w+2*padding_w)//stride_w+1))
    zero_pad=torch.nn.ZeroPad2d(padding=(padding_w,padding_w,padding_h,padding_h))
    X=zero_pad(X)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]=(X[stride_h*i:k_h+stride_h*i,stride_w*j:stride_w*j+k_w]*K).sum()
    return Y  #最后求解得到一个面

X=torch.arange(16).reshape(4,4)
K=torch.tensor([[1,2],[3,4]])
Y=corr2d(X,K,padding=(1,1),stride=(2,2))
print(Y)
print('*'*50)

def corr2d_multi_in(X,K,padding,stride):  #此时的X为(n_c-1,x_h,x_w),kernel为(n_c-1,k_h,k_w),加入padding,stride后结果已经验证正确（多输入单输出）
    #此时注意zip时压缩的是shape最前面的值，例如：a=（3，3，3），b=（3，2，2）的两个矩阵进行zip(a,b),那么其中的一个元素为（3，3），（2，2）构成的元组
    return sum(corr2d(x,k,padding,stride) for x,k in zip(X,K))   # corr2d(x,k)返回的是一个面,此时的sum(对内置元素进行求和)应该是对三个矩阵进行求和

X=torch.arange(18).reshape(2,3,3)
K=torch.arange(8).reshape(2,2,2)
Y=corr2d_multi_in(X,K,padding=(1,1),stride=(2,2))
print(Y)

def corr2d_multi_out(X,K,padding,stride):  # 对应于多输入多输出,此时的K应当是四维的
    return torch.stack([corr2d_multi_in(X,k,padding,stride) for k in K],dim=0)  # for k in K相当于对shape的最外围进行固定 ，stack为增添一个维度
print('*'*50)
X=torch.arange(27,dtype=torch.float32).reshape(3,3,3)
K=torch.arange(8,dtype=torch.float32).reshape(1,2,2,2)
Y=corr2d_multi_out(X,K,padding=(0,0),stride=(1,1))
print(Y)

'''
#对1*1卷积等于进行一个全连接做的判定
X=torch.normal(0,1,(3,3,3))
w=torch.normal(0,1,(2,3,1,1))
def corr2d_multi_out_1x1(X,K):
    n_c_forward,n_h,n_w=X.shape
    n_c=K.shape[0]
    X=torch.reshape(X,(n_c_forward,n_h*n_w)).T  #此时一行为一个通道的所有元素
    K=torch.reshape(K,(n_c,n_c_forward)).T  #此时对于(2,3,1,1)的四维向量转换为（2，3），对于一层的元素的K[0,3,1,1]存放的是一层的参数
    Y=torch.matmul(X,K)  #是一个9*2的，9代表的是一个通道的元素
    Y=Y.T #转秩后一行是一个通道的所有元素
    return torch.reshape(Y,(n_c,n_h,n_w))  #首先填第一个通道的9个元素，由Y的第一行填充

out1=corr2d_multi_out_1x1(X,w)
out2=corr2d_multi_out(X,w,padding=(0,0),stride=(1,1))
print(out1)
print(out2)
assert float(torch.abs(out1-out1).sum()<1e-4)  #进行一个确认操作
'''

def corr2d_muliti_sum(X,K,padding=(0,0),stride=(1,1)): #此时的X有四个维度，最开始的维度为样本的个数，同时此时的输出应当也是四维的，w为四维(扩展到多个样本上)
    return torch.stack([corr2d_multi_out(x,K,padding,stride) for x in X],0)
print('*'*50)
X=torch.arange(54).reshape(2,3,3,3)
K=torch.arange(8).reshape(1,2,2,2)
Y=corr2d_muliti_sum(X,K,padding=(0,0),stride=(1,1))  #相当于模板中的实现的卷积操作
print(Y)