# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:56:17 2019

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from pathlib import Path
import os
def predict(x,curr_parameters):
    prediction=np.dot(x,curr_parameters.W.T)+curr_parameters.b
    return prediction
class parameters:
    def __init__(self,W,b):
        self.W=W
        self.b=b        
def join_arrays(a,b):
    #a=np.ndarray.tolist(a)
    for i in range(len(b)):

        a.append(b[i])
    a=np.asarray(a)
    return a
class SVM:
    def __init__(self,C,W,b):
        self.c=C
        self.W=W
        self.b=b
        self.W=self.W.astype(float)
    def HingeLoss(self,X,Y,W,b):
       #W_t=np.zeros((2,1))
        W_t=self.W.T
        #W_t[1][0]=self.W[1]
        loss=np.dot(W,W_t)
        for i in range(len(X)):
            ti=(np.sum(np.dot(X[i],W_t))+b)*Y[i]
            loss+=max(0,(1-ti))
        return loss
    def find_gradient_w(self,X_random,X,Y,s,e,c):
        ans=self.W
        ans=ans.astype(float)
        #W_t=np.zeros((2,1))
        W_t=self.W.T
        #W_t=self.W[1]
        for i in range(s,e):
            ti=(np.sum(np.dot(X[X_random[i]],W_t))+self.b)*Y[X_random[i]]
            if((1-ti)>0):
                ans-=self.c*X[X_random[i]]*Y[X_random[i]]
        return ans
    def find_gradient_b(self,X_random,X,Y,s,e,c):
        ans=0
        W_t=self.W.T
       # W_t[0][0]=self.W[0]
        #W_t[1][0]=self.W[1]
        for i in range(s,e):
            ti=(np.sum(np.dot(X[X_random[i]],W_t))+self.b)*Y[X_random[i]]
            if((1-ti)>0):
                ans-=c*Y[X_random[i]]
        return ans
    #Function that returns the optimal parameters    
    def fit(self,X,Y,batch_size,max_iterations,learning_rate):
        losses=[]
        curr_loss=self.HingeLoss(X,Y,self.W,self.b)
        for j in range(max_iterations):
            X_random=np.arange(len(X))
            np.random.shuffle(X_random)
            for i in range(0,(len(X_random)-batch_size),batch_size):
                grad_w=self.find_gradient_w(X_random,X,Y,i,i+batch_size,self.c)
                grad_b=self.find_gradient_b(X_random,X,Y,i,i+batch_size,self.c)
                self.W-=learning_rate*grad_w
                self.b-=learning_rate*grad_b
            losses.append(curr_loss)
        return losses,self.W,self.b
# code to import the data    
p=Path("datasets/Images")
dirs=p.glob("*")
dataset_dict={}
for folder in dirs:
    path_array=str(folder).split('\\')
    key=str(path_array[-1])
    for image_name in folder.glob("*.jpg"):
        img=image.load_img(image_name,target_size=(32,32))
        img_array=image.img_to_array(img)
        if (key in dataset_dict.keys()):
            dataset_dict[key].append(img_array)
        else:
            dataset_dict[key]=[]
            dataset_dict[key].append(img_array)
keys=dataset_dict.keys()
keys=list(keys)
# dictionary that contains the count of favourability of the given parameter
parameters_dict={}
t=1
# matrix that keeps record of classifying parameters
record_parameters=np.zeros((len(keys),len(keys)))
record_parameters=np.ndarray.tolist(record_parameters)
for i in range(len(keys)):
    for j in range(i+1,len(keys)):
        if(record_parameters[i][j]==-1):
            continue
        X1=dataset_dict[keys[i]]
        Y1=np.zeros((len(X1),))
        Y1[:]=1
        Y1=np.ndarray.tolist(Y1)
        X2=dataset_dict[keys[j]]
        Y2=np.zeros((len(X2),))
        Y2[:]=-1
        Y2=np.ndarray.tolist(Y2)
        X=join_arrays(X1,X2)
        Y=join_arrays(Y1,Y2)
        # why??
        X=X.reshape(X.shape[0],-1)
        classifier=SVM(1,np.zeros((1,X.shape[1])),0)
        losses,W,b=classifier.fit(X,Y,100,50,0.1)
        curr_parameters=parameters(W,b)
        #print(t)
        #t+=1
        record_parameters[i][j]=curr_parameters
        record_parameters[j][i]=-1
x=dataset_dict['horses'][4]
x=x.reshape((1,3072))
majority_count=np.zeros((len(keys),))
for i in range(len(record_parameters)):
    for j in range(len(record_parameters)):
        if(record_parameters[i][j]!=-1 and  record_parameters[i][j]!=0):
            prediction=predict(x,record_parameters[i][j])
            if(prediction>0):
                majority_count[i]+=1
            else:
                majority_count[j]+=1
#majority_count=sorted(majority_count)
print(np.argmax(majority_count))    