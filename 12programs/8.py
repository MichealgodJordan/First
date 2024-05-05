'''
Author: MichealgodJordan swag162534@outlook.com
Date: 2024-04-01 22:35:34
LastEditors: MichealgodJordan swag162534@outlook.com
LastEditTime: 2024-04-01 23:56:52
FilePath: \neuralComputing\12programs\8.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms, datasets
import d2lzh_pytorch as d2l
import math
import time
import torch.nn as nn
import zipfile
import random
#define the method to import the data
def load_data_jay_lyrics():
    #加载数据集
    with zipfile.ZipFile('./dataset/jaychou_lyrics.txt.zip') as zin:
        with zin.open('./dataset/jaychou_lyrics.txt.zip') as f:
            corpus_chars = f.read().decode('utf-8')
    corpus_chars = corpus_chars.replace('\n',' ').replace('\r',' ')
    corpus_chars = corpus_chars[0:10000]
    #输入中文的索引
    idx_to_char = list(set(corpus_chars))
    char_to_idx = dict([(char,i) for char,i in enumerate(idx_to_char)])
    vocab_size = len(char_to_idx)
    corpus_indices = [char_to_idx[char] for char in corpus_chars]
    return corpus_indices,char_to_idx,idx_to_char,vocab_size

(corpus_indices,char_to_idx,idx_to_char,vocab_size) = load_data_jay_lyrics()

num_hiddens = 256
vocab_size = 10000
rnn_layer = nn.RNN(input_size=vocab_size,hidden_size=num_hiddens)
gru_layer = nn.GRU(input_size=vocab_size,hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None
X = torch.rand(num_steps,batch_size,vocab_size)
Y , state_new = rnn_layer(X,state)
print(Y.shape,len(state_new),state_new[0].shape)

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel,self).__init__()
        #定义为双向
        self.rnn = rnn_layer.hidden_size*(2 if rnn_layer.bidirectional else 1)
        
        self.hidden_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        
        self.state = None
        
    def forward(self, inputs, state):
        #obtain the one_hot 向量表示
#X is a list
        X = d2l.to_onehot(inputs,self.vocab_size)
        #堆叠并返回每次的隐藏层状态
        Y, self.state = self.rnn(torch.stack(X),state)
        #全连接层想将Y的shape变成(num_steps*batch_size, num_hiddens)
        #全连接 层的输出shape为(num_steps*batch_size, vocab_size)
        output = self.dense(Y.view(-1,Y.shape[-1]))
        
        return output, self.state
    
    def test_rnn(prefix,num_chars, model, vocab_size, device, idx_to_char, char_to_idx):
        #prefix 文本开头
        state = None
        #记录prefix加上输出
        output = [char_to_idx[prefix[0]]]
        #依次生成
        for t in range(num_chars + len(prefix)-1):
            X = torch.tensor([output[-1]],device=device).view(1,1)
            
            if state is not None:
                #如果是第一步 避免重复计算梯度
                if isinstance(state,tuple):
                    state = (state[0].to(device),state[1].to(device))
                else:
                    state = state.to(device)
            (Y,state) = model(X,state)
            #如果长度小于继续
            if t<len(prefix)-1:
                output.append(char_to_idx[prefix[t+1]])
                #否则格式化输出
            else:
                output.append(int(Y.argmax(dim=1).item()))
        return ' '.join([idx_to_char[i] for i in output])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RNNModel(gru_layer,vocab_size).to(device)
RNNModel.test_rnn('分开',10,model,vocab_size,device,idx_to_char,char_to_idx)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
model.to(device)
state = None
epochs= 250
batch_size = 32
clipping_theta = 1e-2
pre_period, pred_len, prefixes = 50,50,['我爱','我不爱']

for epoch in range(epochs):
    l_sum,n,start = 0.0,0.0,time.time()
    data_iter = d2l.data_iter_consecutive(corpus_indices,batch_size, num_steps, device)
    for X,y in data_iter:
        if state is not None:
            #use datch function to extract the hidden from computing pictures
            #防止梯度爆炸
            if isinstance(state,tuple):
                state = (state[0].detach(), state[1].detach())
            else:
                state = state.detach()
                
        (output,state) = model(X,state)
        y = torch.transpose(Y,0,1).contiguous().view(-1)
        l = loss(output,y.long())
        optimizer.zero_grad()
        l.backward()
        
        #梯度裁剪
        d2l.grad_clipping(model.parameters(),clipping_theta,device)
        optimizer.step()
        l_sum += l.item()*y.shape[0]
        n += y.shape[0]
        
        #避免长度过长
    try:
        #计算困惑度 = loss/n
        perplexity = math.exp(l_sum/n)
    except OverflowError:
        perplexity = float('int')
        #控制每50轮输出一次
    if(epoch+1) % pre_period == 0:
        print('epoch%d,perplexity %f, time %.2f sec'
              %(epoch+1,perplexity,time.time()-start))
        for prefix in prefixes:
            #将检索转为文本
            print(' -',RNNModel.test_rnn(prefix,pred_len,model,vocab_size,device,
                                idx_to_char,char_to_idx))
