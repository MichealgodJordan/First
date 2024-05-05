'''
Author: MichealgodJordan swag162534@outlook.com
Date: 2024-04-01 22:35:38
LastEditors: MichealgodJordan swag162534@outlook.com
LastEditTime: 2024-05-02 16:35:15
FilePath: \neuralComputing\12programs\9.py
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
    # with zipfile.ZipFile('F:/neuralComputing/src/Jaychou.zip') as zin:
    #     with zin.open('F:/neuralComputing/src/Jaychou.zip') as f:
    with open('F:/neuralComputing/src/Jaychou.txt') as f:
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


num_inputs, num_hiddens, num_outputs = vocab_size,256,vocab_size

#instance the inital parameters
def get_params():
    def _one(shape):
        ts = torch.tensor(np.random.normal(0,0.01,size = shape),device = device,dtype = torch.float32)
        return torch.nn.Parameter(ts,requires_grad = True)
    
    def _three():
        return (_one((num_inputs,num_hiddens)),
                _one((num_hiddens,num_hiddens)),
                torch.nn.Parameter(torch.zeros(num_hiddens,device = device, dtype = torch.float32),
                                   requires_grad= True))

    #输入门遗忘门输出门候选记忆细胞参数
    W_xi,W_hi,b_i = _three()
    W_xf,W_hf,b_f = _three()
    W_xo,W_ho,b_o = _three()
    W_xc,W_hc,b_c = _three()
    
    #output layer parameters
    W_hp = _one((num_hiddens,num_outputs))
    b_q = torch.nn.Parameter(torch.zeros(num_outputs,device =device, dtype = torch.float32, requires_grad = True))
    return nn.ParameterList([W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hp,b_q])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_lstm_state(batch_size,num_hiddens,device):
    return(torch.zeros((batch_size,num_hiddens),device = device),
torch.zeros((batch_size,num_hiddens),device = device))
    
    
#define the model
def lstm(inputs,state,params):
    [W_xi,W_hi,b_i,W_xf,W_hf,b_f,W_xo,W_ho,b_o,W_xc,W_hc,b_c,W_hp,b_q] = params
    (H,C) = state
    outputs = []
    for X in inputs:
        I = torch.sigmoid(torch.matmul(X,W_xi) + torch.matmul(H,W_hi)+ b_i)
        F = torch.sigmoid(torch.matmul(X,W_xf) + torch.matmul(H,W_hf)+ b_f)
        O = torch.sigmoid(torch.matmul(X,W_xo) + torch.matmul(H,W_ho)+ b_o)
        
        H = O*C.tanh()
        Y = torch.matmul(H,W_hp)+ b_q
        outputs.append(Y)
        
    return outputs,(H,C)

num_epochs , num_steps,batch_size,lr, clipping_theta = 160,35,32,1e2 ,1e-2

d2l.train_and_predict_rnn(lstm,get_params,init_lstm_state,num_hiddens,
                          vocab_size,device,corpus_indices,idx_to_char,char_to_idx,
                          False,num_epochs,num_steps,lr,clipping_theta,batch_size,
                          pred_period=40, pred_len=50,prefixes=['分开','不分开'])