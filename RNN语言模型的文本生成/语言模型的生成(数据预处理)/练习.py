import torch
import random
# 创建一个每个batch相关的data_iter
def create_data_iter_seq(corpus,batch_size,num_steps):
    offset = random.randint(0,num_steps)
    num_tokens = (len(corpus) - offset - 1) // batch_size * batch_size
    X_s = torch.tensor(corpus[offset:offset + num_tokens]).reshape(batch_size,-1)
    Y_s = torch.tensor(corpus[offset + 1: offset + num_tokens + 1]).reshape(batch_size,-1)
    num = X_s.shape[1] // num_steps
    for i in range(0, num_steps * num, num_steps):
        X = X_s[:, i: i + num_steps]
        y = Y_s[:, i: i + num_steps]
        yield X,y


random.seed(1)
my_seq=list(range(35))
for X, Y in create_data_iter_seq(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)


def seq_data_iter_random(corpus,batch_size,num_steps):
    corpus = corpus[random.randint(0,num_steps-1):]
    num_subseq = (len(corpus) - 1) // num_steps
    initial_index = list(range(0,num_subseq * num_steps, num_steps))
    random.shuffle(initial_index)
    num_batch = num_subseq // batch_size
    for i in range(0, num_batch * batch_size, batch_size):
        batch_initial_index = initial_index[i:i + batch_size]
        X = [corpus[i : i + num_steps] for i in batch_initial_index]
        y = [corpus[i + 1 : i + num_steps + 1] for i in batch_initial_index]
        yield torch.tensor(X),torch.tensor(y)

random.seed(1)
my_seq = [0,4,3,5,9,8,6,2,10,7,15,13,18,17,19,16,14,12,11,1]
for X, Y in seq_data_iter_random(my_seq, batch_size=2, num_steps=5):
    print('X: ', X, '\nY:', Y)
