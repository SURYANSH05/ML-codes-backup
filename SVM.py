# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 20:11:35 2019

@author: HP
"""
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
class SVM:
    def __init__(self,C,W,b):
        self.c=C
        self.W=W
        self.b=b
        self.W=self.W.astype(float)
    def HingeLoss(self,X,Y,W,b):
        W_t=np.zeros((2,1))
        W_t[0][0]=self.W[0]
        W_t[1][0]=self.W[1]
        loss=np.dot(W,W_t)
        for i in range(len(X)):
            ti=(np.dot(X[i],W_t)+b)*Y[i]
            loss+=max(0,(1-ti))
        return loss
    def find_gradient_w(self,X_random,X,Y,s,e,c):
        ans=self.W
        ans=ans.astype(float)
        W_t=np.zeros((2,1))
        W_t[0][0]=self.W[0]
        W_t[1][0]=self.W[1]
        for i in range(s,e):
            ti=(np.dot(X[X_random[i]],W_t)+self.b)*Y[X_random[i]]
            if((1-ti)>0):
                ans-=self.c*X[X_random[i]]*Y[X_random[i]]
        return ans
    def find_gradient_b(self,X_random,X,Y,s,e,c):
        ans=0
        W_t=np.zeros((2,1))
        W_t[0][0]=self.W[0]
        W_t[1][0]=self.W[1]
        for i in range(s,e):
            ti=(np.dot(X[X_random[i]],W_t)+self.b)*Y[X_random[i]]
            if((1-ti)>0):
                ans-=c*Y[X_random[i]]
        return ans
    #Function that returns the optimal parameters    
    def fit(self,X,Y,batch_size,max_iterations,learning_rate):
        losses=[]
        for j in range(max_iterations):
            X_random=np.arange(len(X))
            np.random.shuffle(X_random)
            for i in range(0,(len(X_random)-batch_size),batch_size):
                grad_w=self.find_gradient_w(X_random,X,Y,i,i+batch_size,self.c)
                grad_b=self.find_gradient_b(X_random,X,Y,i,i+batch_size,self.c)
                self.W-=learning_rate*grad_w
                self.b-=learning_rate*grad_b
                curr_loss=self.HingeLoss(X,Y,self.W,self.b)
            losses.append(curr_loss)
        return losses,self.W,self.b
X,Y=make_classification(n_samples=400,n_features=2,n_informative=2,n_classes=2,random_state=3,n_redundant=0,n_clusters_per_class=1)
Y[Y==0]=-1
classifier=SVM(1,np.array([0,0]),0)
losses,W,b=classifier.fit(X,Y,100,300,0.01)
plt.plot(losses)