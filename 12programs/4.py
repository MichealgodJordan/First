'''
Author: MichealgodJordan swag162534@outlook.com
Date: 2024-04-01 21:36:53
LastEditors: MichealgodJordan swag162534@outlook.com
LastEditTime: 2024-04-01 22:32:29
FilePath: \neuralComputing\12programs\4.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time

from torch.utils.data import DataLoader

from torchvision import transforms , datasets

def nin_block(in_channels,out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels,out_channels, kernel_size,stride,padding),
                        nn.ReLU(),
                        #后两层对数据再接收再处理
                        nn.Conv2d(out_channels, out_channels, kernel_size =1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels,out_channels,kernel_size = 1),
                        nn.ReLU()
              )
    return blk
class GobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GobalAvgPool2d,self).__init__()
        
    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:])

net = nn.Sequential(
    
    nin_block(3,96,kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=2, stride = 2),
    #缩小kernel_size
    nin_block(96,256,kernel_size= 5, stride = 1, padding = 2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nin_block(256,384,kernel_size= 3, stride = 1, padding = 1),
    nn.MaxPool2d(kernel_size=2, stride=2),
    #根据标签数量来设置最后一个nin__block
    nin_block(384,10,kernel_size= 5, stride = 1, padding = 1),
    GobalAvgPool2d(),
    #把四维输出转换成2维 shape=(batch_size,10(标签数量))
    nn.Flatten()
)

x = torch.rand(1,3,224,224)
print(net(x).shape)

for name, blk in net.named_children():
    x = blk(x)
    print(name,'output size' , x.shape)
    
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #标准化
    transforms.Normalize(mean=[0.485,0.456,0.406], std = [0.229,0.224,0.225])
])

train_sets = datasets.CIFAR10(root = 'cifar_10',#选择数据根目录
                              train = True,#选择训练集
                              download=True,#下载图片
                              transform=data_transform)
test_sets = datasets.CIFAR10(root = 'cifar_10',#选择数据根目录
                              train = False,#选择测试集
                              download=True,#下载图片
                              transform=data_transform)
batch_size = 64

train_loader = DataLoader(dataset=train_sets, 
                          batch_size=batch_size,
                          shuffle=True )#打乱数据
                    
test_loader = DataLoader(dataset=train_sets, 
                          batch_size=batch_size,
                          shuffle=True)

def evaluate_accuracy(data_iter,net):
    acc_sum , n = 0.0,0
    net.eval()
    for X,y in data_iter:
        X = X.cuda()
        y = y.cuda()
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        
        n += y.shape[0]
    return acc_sum / n

#train_iter训练集
def train(net, train_iter, test_iter,batch_size, optimizer, device, num_epochs):
    
    net = net.to(device)
    print("training on", device)
    loss = torch.nn.CrossEntropyLoss()
    batch_acount = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum,n,start = 0.0,0.0,0,time.time()
        net.train()
        for X,y in train_iter:
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X)
            l = loss(y_hat,y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            
            #验证测试集
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_acount += 1
            
        test_acc = evaluate_accuracy(test_iter,net)
        print('epochs %d, loss %.4f, train acc %.3f, test f1 %3f, test recall %.3f, precision %3f,time cost %.1f sec'
              %(epoch+1,train_l_sum/batch_acount, train_acc_sum/n,test_acc,time.time()-start))

train_iter,test_iter = train_loader,test_loader
lr =0.0001
num_epochs = 5
optimizer = torch.optim.Adam(net.parameters(),lr)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train(net, train_iter,test_iter,batch_size,optimizer,device,num_epochs)