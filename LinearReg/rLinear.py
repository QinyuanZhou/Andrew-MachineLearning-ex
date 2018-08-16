import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = 'ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population','Profit'])
# print(data.head()) #输出前5个数据
# print(data.describe()) #输出最大最小平均值等

# data.plot(kind='scatter',x='Population',y='Profit',figsize=(10,6))
# plt.show()#显示数据图

# 定义代价函数
def computeCost(X, y, theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

data.insert(0,'ones',1) #在第0列插入全为1的数
cols = data.shape[1] #记录数据的列数，shape[0]记录行数
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]
# print(X.head())
# print(y.head()) #输出前5个数据

# 转换成矩阵
X = np.matrix(X.values) #97行 2列
y = np.matrix(y.values) #97行 1列
theta = np.matrix(np.array([0,0]))#1行2列
# print(computeCost(X,y,theta)) #刚开始是32.07273

# 批量梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1]) #ravel()是降为成一维，并且是一个引用,这里返回列数2
    cost = np.zeros(iters)
    for i in range(iters):
        error = (X*theta.T)-y
        for j in range(parameters):
            term = np.multiply(error, X[:,j]) #第j个term等于thetta[j]
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term))
        theta = temp
        cost[i] = computeCost(X, y, theta)
        print('Epoch:', i, cost[i], '\n')
    return theta, cost

alpha = 0.01
iters = 1000
g, cost = gradientDescent(X, y, theta, alpha, iters) #cost是一个1000的列表
x = np.linspace(data.Population.min(), data.Population.max(), 100)#在指定间隔内返回均匀间隔的数字
f = g[0, 0] + g[0, 1] * x

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

ax2 = fig.add_subplot(1,2,2)
ax2.plot(np.arange(iters), cost, 'r')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Cost')
ax2.set_title('Error vs. Training Epoch')
plt.show()