# from d2l import torch as d2ll
# d2ll.DATA_HUB['aclImdb'] = ('https://ai.stanford.edu/amaas/data/sentiment/aclImdb_vl.tar.gz',
#                             '01ada507287d82875905620988597833ad4e0903')
# data_dir = d2ll.download_extract('aclImdb','aclImdb')

import collections
import os
import random
import tarfile
import torch
import d2lzh_pytorch as d2l
import sys
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

sys.path.append('..')

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './Datasets'
print(torch.__version__, device)
from tqdm import tqdm

fname = os.path.join(DATA_ROOT,'aclImdb_vl.tar.gz')
if not os.path.exists(os.path(DATA_ROOT,'aclImdb')):
    print('Extracting...')
    
    with tarfile.open(fname,'r') as f:
        f.extractall(DATA_ROOT)
        
def read_imdb(folder,data_root):
    data=[]
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_root,folder,label)
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name,file),'rb') as f:
                review = f.read().decode('utf-8').repalce('\n','').lower()
                data.append([review,1 if label == 'pos' else 0])
                
    random.shuffle(data)
    return data


#laod the data
data_root = './datasets/aclImdb'
train_data,test_data = read_imdb('train',data_root),read_imdb('test',data_root)

#data preprocess
def get_tokenized_imdb(data):
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review , _ in data]

#根据分词后的数据创建词典过滤掉次数少于5的词
def get_vocab_imdb(data):
    tokenized_data =  get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq = 5)
vocab = get_vocab_imdb(train_data)
'#words in vocab', len(vocab)

#the length of each comment 不能直接组成小批量
#要对每条评论通过词典换成词索引进行截断或者补0确保每条评论的长度固定成500 


def preprocess_imdb(data,vocab):
    max_l = 500
    def pad(x):
        return x[:max_l] if len(x) > max_l else x+[0] *(max_l - len(x))
    tokenized_data = get_tokenized_imdb(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) 
                             for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features,labels


#creat data_iter return a small amount of data
batch_size = 32
train_set = Data.TensorDataset(*preprocess_imdb(train_data,vocab))
test_set = Data.TensorDataset(*preprocess_imdb(test_data,vocab))
train_iter = Data.DataLoader(train_set, batch_size, shuffle = True)
test_iter = Data.DataLoader(test_set,batch_size)

for X,y in train_iter:
    print('X in shape is ', X.shape, 'y shape is ', y.shape)
    break

class BiRNN(nn.Module):
    def __init__(self,vocab, emded_size, num_hiddens, num_layers):
        super(BiRNN,self).__init__()
        
        #bidirectional 设为True即得到双向循环神经网络
        self.encoder = nn.LSTM(input_size=emded_size,
                               hidden_size=num_hiddens,
                               num_layers=+num_layers,
                               bidirectional=True)
        #初始时间步和最终时间步的隐藏状态作为全连接层输入
        self.decoder = nn.Linear(4*num_hiddens,2)
        
    def forward(self, inputs,outputs):
        #inputs shape = (batch_size, len(words))
        #lstm 需要序列长度seq_len作为第一维度，所以将输入转置后在提取特征，输出形状为（词数，批量大小，词向量维度）
        embeddings = self.embedding(inputs.permute(1,0))
        outputs, self.encoder(embeddings)
        encoding = torch.cat((outputs[0],outputs[-1]),-1)
        outs = self.decoder(encoding)
        return outs
embed_size, num_hiddens, num_layers = 100,100,2
net = BiRNN(vocab,embed_size, num_hiddens,num_layers)


lr = 0.01
num_epochs = 5
optimizer = torch.optim.Adam(net.parameters(), lr)
loss= nn.CrossEntropyLoss()
d2l.train(train_iter,test_iter, net, loss, optimizer,device, num_epochs)