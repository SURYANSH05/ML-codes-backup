# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 13:59:07 2019

@author: HP
"""
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
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
# function to tokenize,stopword,stem
def make_dataset(data):
    tokenize=sent_tokenize(data)
    #print(tokenize)
    stopword=set(stopwords.words('english'))
    #Array that contains words after removing stopwords
    final_array=[]
    for i in range(len(tokenize)):
        if(tokenize[i] in stopword):
            continue
        final_array.append(tokenize[i])
    #print(stopword)
    #for i in tokenize
    Ps=PorterStemmer()
    output=[]
    final_array=sep_words(final_array)
    for i in final_array:
        output.append(Ps.stem(i))
    return output
data="Running from my self no more"
#o=sep_words([data])
print(make_dataset(data))    
