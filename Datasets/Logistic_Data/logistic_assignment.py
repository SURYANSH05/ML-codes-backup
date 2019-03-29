# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 00:33:41 2019

@author: HP
"""
def hypothesis(x,theta,b):
    z=np.dot(theta,x)+b
    h=1.0/(1.0+np.exp(-z))
    return h
def find_gradient(X,Y,theta,b,batch_size):
    grad_theta=np.zeros(2)
    grad_const=0
    indices=np.arange(X.shape[0])
    np.random.shuffle(indices)
    indices=indices[:batch_size]
    for i in range(len(indices)):
        t=(Y[indices[i]]-hypothesis(X[indices[i]],theta,b))
        grad_theta+=(Y[indices[i]]-hypothesis(X[indices[i]],theta,b))*X[indices[i]]
        grad_const+=(Y[indices[i]]-hypothesis(X[indices[i]],theta,b))
    return grad_theta,grad_const    
def gradient_descent(X,Y,learning_rate,n_iterations):
    theta=np.zeros(2)
    b=0
    for i in range(X.shape[0]):
        grad_theta,grad_const=find_gradient(X,Y,theta,b,5)
        theta=theta+learning_rate*grad_theta
        b=b+learning_rate*grad_const
    return theta,b    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
X=pd.read_csv('logisticX.csv')
Y=pd.read_csv('logisticY.csv')
X=X.values
Y=Y.values
plt.figure()
for i in range(len(X)):
    if(Y[i]==1):
        plt.scatter(X[i][0],X[i][1],color='yellow')
    if(Y[i]==0):
        plt.scatter(X[i][0],X[i][1],color='red')
theta,b=gradient_descent(X,Y,0.01,100)
indices=np.arange(10)
# corresponding y values
y=-((theta[0]*indices)+b)/theta[1]
plt.plot(indices,y)
