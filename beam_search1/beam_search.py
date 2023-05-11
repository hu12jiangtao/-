# 此处的beam_search用于translate的翻译
import torch
from heapq import heappop,heappush


class BeamSearchNode(object):
    def __init__(self, h, prev_node, wid, logp, length):
        self.h = h # 用来存放每个序列的decode输入的序列,存放 shape = [num_layer,1,dec_hidden_num]
        self.prev_node = prev_node # 用来存放当前节点的上一个节点
        self.wid = wid # 用来存放decode预测的序列 # shape = [1,1]
        self.logp = logp # 通过decode得到的概率值 # 值为常数
        self.length = length # 为当前node的序列长度+1

    def eval(self):
        return - self.logp / (float(self.length - 1 + 1e-6) ** 0.75) # 预测的概率越大，这个评估函数的分数就越小

def beam_search_decoding(decoder,enc_outs, enc_last_h, beam_width, n_best, sos_token, eos_token, max_dec_steps,device):
    # 此函数中为单个样本输入，判断出最优的n_best个结果作为这个样本可能的预测结果
    # 在该项目中enc_hidden_num = dec_hidden_num
    # 对于翻译中的encode来说 输入为X=[batch,seq] 输出的y=[seq,batch,enc_hidden_num],state=[num_layer=2,batch,enc_hidden_num]
    # decode模块,输入X=[batch,seq],state=(enc_out=[batch,seq,enc_hidden_num],enc_state=[2,batch,state])
    # 输出为 out=[batch, seq, vocab_size], state= (enc_out=[batch,seq,enc_hidden_num],dec_state=[2,batch,state])
    assert beam_width >= n_best
    n_best_list = []
    bs = enc_outs.shape[1]
    enc_outs = enc_outs.permute(1,0,2) # [batch, seq,enc_hidden_num]
    for i in range(bs):
        nodes = []
        end_nodes = [] # 用来存放5个最优的序列
        n_best_seq_list = []
        # 创建一个初始的节点，对于最开始的序列的hidden应当取自encode的hidden状态(hidden=[batch,dec_hidden_num],wid对应的应当是<sos>序列)
        decode_input = torch.tensor([sos_token],dtype=torch.long,device=device).reshape(1,1) # [1,1]
        decode_hidden = enc_last_h[:,i,:].contiguous().unsqueeze(1) # [2,1,dec_hidden_num]  不加入contiguous()会报错
        enc_out = enc_outs[i].unsqueeze(0) # [1,seq,num_hidden]
        node = BeamSearchNode(decode_hidden, prev_node = None, wid=decode_input, logp=0, length=1)
        heappush(nodes,(node.eval(),id(node),node))
        # 开始进入循环，跳出循环的两个条件:1.到达规定的步数 2.找出n_best个结果
        n_dec_step = 0
        while True:
            if n_dec_step > max_dec_steps:
                break
            score, _, n = heappop(nodes)
            if n.prev_node is not None and n.wid.item() == eos_token:
                end_nodes.append((score, id(n), n))
                if len(end_nodes) >= n_best:
                    break
                else:
                    continue
            # 进入到当前的序列模型中
            decode_input = n.wid # [1,1]
            decode_hidden = n.h # [2,1,dec_hidden_num]
            dec_state = (enc_out,decode_hidden)
            pred, (enc_out,decode_hidden) = decoder(decode_input,dec_state) # decode_hidden = [2,1,dec_hidden_num]
            # pred = [1,1,vocab_size], decode_hidden = (enc_y.permute(1,0,2)=[batch,seq,enc_hidden_num],[2,1,dec_hidden_num])
            topk_log_value, topk_index = torch.topk(pred.squeeze(0),beam_width) # [1,beam_width]
            for j in range(beam_width):
                decode_p = topk_index[0][j].reshape(1,1) # [1,1]
                logp = n.logp + topk_log_value[0][j].item() # 得到一个常数
                node = BeamSearchNode(decode_hidden, prev_node = n, wid = decode_p, logp = logp, length = n.length+1)
                heappush(nodes,(node.eval(),id(node),node))
            n_dec_step += beam_width
        if len(end_nodes) == 0: # 达到了指定步数，但是没有eos结尾的句子的情况
            end_nodes = [heappop(nodes) for _ in range(n_best)]
        for node in sorted(end_nodes,key=lambda x:x[0]): # 遍历其中的所有的节点
            score, _, n = node
            sequence = [n.wid.item()]
            while n.prev_node is not None:
                n = n.prev_node
                sequence.append(n.wid.item())
            sequence = sequence[::-1]
            n_best_seq_list.append(sequence)
        n_best_list.append(n_best_seq_list) # 存放所有样本的最优的n_best_seq_list
    return n_best_list