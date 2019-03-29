# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 23:38:21 2019

@author: HP
"""
def prior(Y_set,typee):
    count=0
    for i in range(len(Y_set)):
        if(Y_set[i]==typee):
           count+=1
    return (count/len(Y_set))
# probability of x given type y
def likelihood(X_train,Y_train,x,typee):
    X_filtered=[]
       # getting all the rows of the rquired type
    for i in range(len(Y_train)):
        if(Y_train[i]==typee):
            X_filtered.append(X_train[i])
    likelihood=1
    for i in range(len(x)):
       numerator=0
       for j in range(len(X_filtered)):
           if(X_filtered[j][i]==x[i]):
               numerator+=1
       likelihood*=(numerator/len(X_filtered))
    return likelihood
# type array should contain all the type of entry ex [0,1,2]
def naive_bayes(X_train,Y_train,x,types):
    probability=[]
    for i in range(len(types)):
        curr_prob=prior(Y_train,types[i])*likelihood(X_train,Y_train,x,types[i])
        probability.append((curr_prob,types[i]))
    probability=sorted(probability)
    return probability[-1][1]
def get_accuracy(X_train,X_test,Y_train,Y_test,types):
    # create the types array
    numerator=0
    for i in range(len(X_test)):
        prediction=naive_bayes(X_train,Y_train,X_test[i],types)
        if(prediction==Y_test[i]):
            numerator+=1
    return (numerator/len(X_test))        
import numpy as np
import pandas as pd
dataframe=pd.read_csv('mushrooms.csv')
data=dataframe.values
X=data[:,1:]
Y=data=data[:,0]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)
x=X[7]
types=np.unique(Y)
prediction=naive_bayes(X,Y,x,types)
print(get_accuracy(X_train,X_test,Y_train,Y_test,types))
 
