# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 00:26:01 2019

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def hypothesis(X,theta,b):
    z=np.dot(theta,X)+b
    h=1.0/(1.0+np.exp(-z))
    return h
def find_gradient(theta,X,Y,b):
    grad_theta=[0,0]
    grad_const=0
    for i in range(len(X)):
        grad_theta+=(-hypothesis(X[i],theta,b)+Y[i])*X[i]
        grad_const+=(-hypothesis(X[i],theta,b)+Y[i])
    #grad_theta=grad_theta/(len(X))
    #grad_const=grad_const/len(X)
    return grad_theta,grad_const    
def gradient_descent(X,Y,learning_rate,n_iterations=100):
    theta=np.random.random((2,))
    b=0
    for i in range(n_iterations):
        grad_theta,grad_const=find_gradient(theta,X,Y,b)
        theta=theta+learning_rate*grad_theta
        b=b+learning_rate*grad_const
    return theta,b
mean_01 = np.array([1,0.5])
cov_01 = np.array([[1,0.1],[0.1,1.2]])

mean_02 = np.array([4,5])
cov_02 = np.array([[1.21,0.1],[0.1,1.3]])


# Normal Distribution

dist_01 = np.random.multivariate_normal(mean_01,cov_01,500)
dist_02 = np.random.multivariate_normal(mean_02,cov_02,500)

print(dist_01.shape)
print(dist_02.shape)
data = np.zeros((1000,3))
print(data.shape)
split = int(0.8*data.shape[0])
data[:500,:2] = dist_01
data[500:,:2] = dist_02
data[500:,-1] = 1.0
X_train = data[:split,:-1]
X_test = data[split:,:-1]
Y_train = data[:split,-1]
Y_test  = data[split:,-1]
W = 2*np.random.random((X_train.shape[1],))
theta,b=gradient_descent(X_train,Y_train,0.1,1000)
plt.figure(0)
plt.scatter(dist_01[:,0],dist_01[:,1],label='Class 0')
plt.scatter(dist_02[:,0],dist_02[:,1],color='r',marker='^',label='Class 1')
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('x1')
plt.ylabel('x2')

x = np.linspace(-4,8,10)
y = -(theta[0]*x + b)/theta[1]
plt.plot(x,y,color='k')

plt.legend()
plt.show()


