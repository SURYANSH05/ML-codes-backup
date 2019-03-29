# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 12:44:23 2019

@author: HP
"""

import numpy as np
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
#from nltk.stem.snowball import PorterStemmer
from nltk.stem.snowball import PorterStemmer
# function to separate words from a sentence
def sep_words(array):
    output=[]
    for i in range(len(array)):
        j=0
        for m in range(len(array[i])):
            a=""
            if(j>=len(array[i])):
                break
            while array[i][j]!=' ' and j!=len(array[i])-1:
                a+=array[i][j]
                j+=1
            output.append(a)
            j+=1
    return output        
def clean_data(data):
    tokenizer=RegexpTokenizer("[\w']+")
    tokenized=tokenizer.tokenize(data)
    stopword=set(stopwords.words('english'))
    final_array=[]
    for i in range(len(tokenized)):
        if (tokenized[i] in stopword)==False:
            final_array.append(tokenized[i])
    Ps=PorterStemmer()
    output=[]
    final_array=sep_words(final_array)
    for i in final_array:
        output.append(Ps.stem(i))
    return output
# probability of x given y
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
       likelihood*=(numerator+1/len(X_filtered)+1)
    return likelihood
def prior(Y_set,typee):
    count=0
    for i in range(len(Y_set)):
        if(Y_set[i]==typee):
           count+=1
    return (count/len(Y_set))
def naive_bayes(X_train,Y_train,x,types):
    probability=[]
    for i in range(len(types)):
        curr_prob=prior(Y_train,types[i])*likelihood(X_train,Y_train,x,types[i])
        probability.append((curr_prob,types[i]))
    probability=sorted(probability)
    return probability[-1][1]
def predict_NB(X_train,Y_train,x):
    ans=0
    for i in range(len(Y_train)):
        temp_ans=likelihood(x,Y_train[i],X_train,Y_train)*prior(Y_train,Y_train[i])
        if temp_ans<ans:
            ans=temp_ans
    return ans
#Function to convert the string array data into frequency type
def convert_data(clean_data,unique_set):
    column_length=len(unique_set)
    row_length=len(clean_data)
    output=np.zeros((row_length,column_length))
    for i in range(len(clean_data)):
        t=0
        for j in ((unique_set)):
            output[i][t]=clean_data[i].count(j)
            t+=1
    return output
# import some data
data = ["This was an awesome movie",
     "Great movie! I liked it a lot",
     "Happy Ending! awesome acting by the hero",
     "loved it! truly great",
     "bad not upto the mark",
     "could have better",
     "Surely a Disappointing movie"]
data_Y=[1,1,1,1,0,0,0]
#Cleaning the data
for i in range(len(data)):
    data[i]=clean_data(data[i])
cleaned_data=np.asarray(data)
unique_data=np.concatenate(data,axis=0)
#unique_data=np.unique(unique_data)   
#a = np.array(unique_data, dtype='<U4')
unique_set=set(unique_data) 
test_data="movie sucked"
test_data=clean_data(test_data)
test_data=[test_data]
test_data=convert_data(test_data,unique_set)
machine_data=(convert_data(cleaned_data,unique_set))
print(naive_bayes(machine_data,data_Y,test_data[0],[1,0]))

