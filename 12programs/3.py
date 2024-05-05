import numpy as np
import torch
import operator
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms , datasets
import d2lzh_pytorch as d2l
import time


#清除缓存占用
torch.cuda.empty_cache()
from torchvision import models
net = models.resnet18(pretrained=True)
net.eval()
for param in net.parameters():
    param.requires_grad = True
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
    #将图片resize成256*256
    transforms.Resize(256),
    #随机裁剪224*224
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #标准化
    transforms.Normalize(mean=[0.285,0.456,0.406],std = [0.229,0.224,0.225])
])


train_dataset = datasets.ImageFolder(root='F:/neuralComputing/12programs/dataset/stock_1500.csv',transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle = True, num_workers = 2)

validate_dataset = datasets.ImageFolder(root='F:/neuralComputing/12programs/dataset/stock_1500.csv',transform=data_transform)
validate_loader = DataLoader(validate_dataset,batch_size = 64 , shuffle = True, num_workers = 2)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,96,11,4),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(3,2),
            
            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            
            nn.Conv2d(374,384,3,1,1),
            nn.ReLU(),
            
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(3,2)
        )
        
        self.fc = nn.Sequential(
        nn.Linear(256*5*5, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.Linear(4096,4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.Linear(4096,2),
    )
        


batch_size = 64
num_epochs = 5

optimizer = torch.optim.Adam(net.parameters(),lr=operator.le-4,weight_decay=operator.le-3)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.is_available())

from sklearn import metrics
def evaluate_f1_score(data_iter, net):
    predict = np.array([])
    truelab = np.array([])
    net.to('cpu')
    net.eval()
    for X, y in data_iter:
        #print(net(X).argmax(dim=1))
        predict = np.hstack((net(X).argmax(dim=1).detach().numpy()))
        #acc_num += (net(x).argmax(dim=1) == y).flot().sum().item()
        
        truelab = np.hstack(((y.detach().numpy())))
        print(predict)
        print(truelab)
        
    truelab = np.array(truelab).astype('int64')
    predict = np.array(predict).astype('int64')
    f1 = metrics.f1_score(truelab,predict,pos_label=1, average='weighted')
    recall = metrics.recall_score(truelab,predict,pos_label=1, average = 'weighted')
    precision = metrics.precision_score(truelab, predict,pos_label = 1, average = 'weighted')
    return f1, recall ,precision

def my_train(net, train_iter, test_iter, batch_size, optimizer,device,num_epochs):
    net = net.to(device)
    print('training on'. device)
    loss = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_l_sum, train_acc_num, n, batch,count, start = 0.0, 0.0, 0,0, time.time()
        net = net.to(device)
        net.train()
        
        for X, y in train_iter:
            net= net.to(device)
            X = X.to(device)
            y = y.to(device)
            
            y_hat = net(X)
            l = loss(y_hat,y)
            
            optimizer.zero_grad()
            
            l.backward()
            optimizer.step()
            
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            
            n+= y.shape[0]
            batch_count +=1
        #f1(recall,precision)
        test_f1,test_recall, test_precision = evaluate_f1_score(test_iter,net)
        
        print('epochs %d, loss %4f, train acc %3f, test f1 %3f, test recall %3f, precision %3f,time cost %.1f sec'
              %(epoch+1,train_l_sum/n, train_acc_num/n,test_recall,test_precision,time.time()-start))




my_train(net,train_loader,validate_dataset, batch_size, optimizer, device, num_epochs)