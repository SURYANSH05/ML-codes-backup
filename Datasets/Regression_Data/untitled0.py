# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:44:43 2019

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_regression
# Generate Dataset
X,Y = make_regression(n_samples=400,n_features=1,n_informative=1,noise=1.8,random_state=11)


Y = Y.reshape((-1,1))
print(X.shape)
print(Y.shape)
# Normalize
X = (X-X.mean())/X.std()

ones = np.ones((X.shape[0],1))
X_ = np.hstack((X,ones))
print(X_.shape)
print(X_[:5,:])
def predict(X,theta):
    return np.dot(X,theta)

def getThetaClosedForm(X,Y):
    
    Y = np.mat(Y)
    firstPart = np.dot(X.T,X)
    secondPart = np.dot(X.T,Y)
    
    theta = np.linalg.pinv(firstPart)*secondPart
    return theta
theta = getThetaClosedForm(X_,Y)
print(theta)
plt.figure()
plt.scatter(X,Y)
plt.plot(X,predict(X_,theta),color='red',label="prediction")
plt.title("Normalized Data")
plt.legend()
plt.show()