import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'ex1data2.txt'
data2 = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])
data2 = (data2 - data2.mean()) / data2.std()
# print(data2.head())

# 定义代价函数
def computeCost(X, y, theta):
    inner = np.power(((X*theta.T)-y),2)
    return np.sum(inner)/(2*len(X))

data2.insert(0, 'Ones', 1)
cols = data2.shape[1]
X2 = data2.iloc[:, 0:cols-1]
y2 = data2.iloc[:, cols-1:cols]

X2 = np.matrix(X2.values)
y2 = np.matrix(y2.values)
theta2 = np.matrix(np.array([0,0,0]))

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
g2, cost2 = gradientDescent(X2, y2, theta2, alpha, iters)
print(computeCost(X2, y2, g2))

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iteration')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()