'''
Author: MichealgodJordan swag162534@outlook.com
Date: 2024-04-01 15:52:41
LastEditors: MichealgodJordan swag162534@outlook.com
LastEditTime: 2024-05-02 16:26:21
FilePath: \neuralComputing\12programs\2.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import numpy as np
import pandas as pd

#数据集分割
from sklearn.model_selection import train_test_split
#标签编码器
from sklearn.preprocessing import LabelEncoder
#逻辑回归
from sklearn.linear_model import LogisticRegression
#K近邻分类器
from sklearn.neighbors import KNeighborsClassifier
#决策树分类器
from sklearn.tree import DecisionTreeClassifier
#评估指标矩阵
from sklearn import metrics

from sklearn import datasets

import matplotlib.pyplot as plt
plt.style.use('seaborn')
import seaborn as sns
sns.set_style("whitegrid")


data= pd.read_csv('F:/neuralComputing/12programs/dataset/titanic.csv')
print(data)


X = data[['Pclass','SibSp','Parch','Fare']]
Y = data['Survived']

#对标签编码
encoder = LabelEncoder()
Y = encoder.fit_transform(Y)

train_X, test_X , train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state = 101)

#K近邻分类器
#define the model
KNmodel = KNeighborsClassifier(algorithm='auto',
                               leaf_size = 6,
                               metric= 'minkowski',
                               n_jobs= None,
                               n_neighbors=9,
                               p=1,
                               weights= 'uniform')

KNmodel.fit(train_X,train_y)
pre_KN = KNmodel.predict(test_X)

acc_KN = metrics.accuracy_score(pre_KN, test_y)
print('The acc of KNeighborsClassifier is : {0}'.format(acc_KN))

#逻辑回归分类器
LRmodel = LogisticRegression()

LRmodel.fit(train_X, train_y)

pre_LR = LRmodel.predict(test_X)
acc_LR = metrics.accuracy_score(pre_LR, test_y)
print('The acc of LogisticRegression is :{0}'.format(acc_LR))

#决策树分类器
DTmodel = DecisionTreeClassifier()

DTmodel.fit(train_X, train_y)

pre_DT = DTmodel.predict(test_X)

acc_DT = metrics.accuracy_score(pre_DT,test_y)
print('The acc of LogisticRegression is {0}'.format(acc_DT))

