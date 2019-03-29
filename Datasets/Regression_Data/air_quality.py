# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 19:36:00 2019

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#function to calculate the hypothesis with the given parameters
def hypothesis(theta,X):
    return theta[0]+theta[1]*X[0]+theta[2]*X[1]+theta[3]*X[2]+theta[4]*X[3]+theta[5]*X[4]
#function to calculate the total error
def error_function(X,Y,theta):
    sum=0
    # check value of length X
    for i in range(len(X)):
        sum+=(hypothesis(theta,X[i,:])-Y[i])**2
    return sum
#function to calculate the partial derivative
def gradient(theta,X,Y):
    grad=np.zeros(len(theta))
    #grad[0]=hypothesis(theta,X[0])-Y[0]
    for j in range(len(theta)):
        for i in range(len(X)):
            if(j==0):
                grad[j]+=hypothesis(theta,X[i])-Y[i]
                continue
            grad[j]+=(hypothesis(theta,X[i])-Y[i])*X[i][j-1]
    return grad    
# function to find the best theta
def gradient_descent(X,Y,learning_rate):
    theta=np.zeros(6)
    threshold=2
    theta[0]=3
    theta[1]=2
    error=[]
    itr=1
    while(error_function(X,Y,theta)>threshold & itr<100):
        error.append(error_function(X,Y,theta))
        # update theta[i] using gradient descent
        grad=gradient(theta,X,Y)
        for i in range(len(theta)):
            theta[i]=theta[i]-learning_rate*grad[i]
        itr+=1    
    return theta,error
dataset=pd.read_csv("Train.csv").values
X=dataset[:,0:5]
Z=(X-X.mean())/X.std()
Y=dataset[:,5]
theta,error=gradient_descent(X,Y,1)
plt.figure()
plt.plot(error)
plt.show()
 