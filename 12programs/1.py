'''
Author: MichealgodJordan swag162534@outlook.com
Date: 2024-04-01 15:21:04
LastEditors: MichealgodJordan swag162534@outlook.com
LastEditTime: 2024-04-29 16:41:05
FilePath: \neuralComputing\12programs\1.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('F:/neuralComputing/practice/深度学习12个项目实践/1时间预测回归预测/1F_SZ300.csv')

data.values[:,0:7].shape


x_data = data.values[:,0:7]
y_data = data.values[:,7:]

#切分训练集测试集
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state = 0, test_size = 0.20)
#数据标归一化处理
min_max_scaler =  preprocessing.MinMaxScaler()
#使用线性回归模型
lr = LinearRegression()
#传入数据集
lr.fit(X_train, y_train)

#预测
predit = lr.predict(X_test)

#使用R2_score评估模型
score = r2_score(y_test,predit)
print(score)