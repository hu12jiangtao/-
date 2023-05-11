import torch

from mingpt.utils import CfgNode as CN
from torch.utils import data
from mingpt import model
from mingpt import trainer
from mingpt import utils

def get_config():
    C = CN()
    C.system = CN()
    C.system.seed = 3047
    C.system.work_dir = './out/chargpt'

    C.data = CharDataset.get_default_config()

    C.model = model.GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    C.trainer = trainer.Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4
    return C

class CharDataset(data.Dataset):
    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128 # 语言模型中输入序列的长度
        return C

    def __init__(self,config,text):
        super(CharDataset, self).__init__()
        self.config = config
        vocab = sorted(list(set(text))) # 并不是按照在文章中出现的次数进行排序，而是根据阿斯克码值进行排序
        self.text_size = len(text) # 文本中一种有多少个字符
        self.vocab_size = len(vocab)
        print('data has %d characters, %d unique.' % (self.text_size, self.vocab_size))
        self.text = text
        self.token_to_idx = {char:idx for idx,char in enumerate(vocab)}
        self.idx_to_token = {idx:char for idx,char in enumerate(vocab)}

    def get_block_size(self):
        return self.config.block_size

    def __getitem__(self, item): # 语言模型的输入x=text[i:i+block_size],y=text[i+1:i+block_size+1]
        sentence = self.text[item : item + self.get_block_size() + 1]
        sentence_ids = [self.token_to_idx[s] for s in sentence]
        x = torch.tensor(sentence_ids[:-1],dtype=torch.long)
        y = torch.tensor(sentence_ids[1:],dtype=torch.long)
        return x,y

    def __len__(self):
        return self.text_size - self.get_block_size() - 1



if __name__ == '__main__':
    # 确认所用到的参数
    config = get_config()
    # 确认随机种子
    utils.set_seed(config.system.seed)
    # 数据集的预处理
    text = open('input.txt','r',encoding='utf-8').read()
    train_dataset = CharDataset(config.data, text) # config.data的值为CN的类对象
    config.model.vocab_size = train_dataset.vocab_size
    config.model.block_size = train_dataset.get_block_size()
    # 导入GPT的模型
    model = model.GPT(config.model)
    # 迭代的方式
    trainer = trainer.Trainer(config.trainer, model, train_dataset)
    # 进行训练
    trainer.run()



