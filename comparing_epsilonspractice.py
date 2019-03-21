# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:20:58 2019

@author: HP
"""
import numpy as np
import matplotlib.pyplot as plt
a=np.arange(5)
class Bandit:
    def __init__(self,m):
        self.m=m
        self.N=0
        self.mean=0
    def update(self,x):
        self.N=self.N+1
        self.mean=(1 - 1.0/self.N)*self.mean + (1.0/self.N)*x
    def pull(self):
        return np.random.randn()+self.m
def run_experiment(m1,m2,epsilon,n):
    bandit1=Bandit(m1)
    bandit2=Bandit(m2)
    data=np.empty(n)
    for i in range(0,n):
        r=np.random.random()
        #explore
        if(epsilon<r):
            choice=np.random.choice(2)
            if(choice==1):
                x=bandit1.pull()
                bandit1.update(x)
            else:
                x=bandit2.pull()
                bandit2.update(x)
        #exploit
        else:
            if(bandit1.mean>bandit2.mean):
                x=bandit1.pull()
                bandit1.update(x)
            else:
                x=bandit2.pull()
                bandit2.update(x)
        data[i]=x    
    cumulative_average=np.cumsum(data)/(np.arange(n)+1)
    return cumulative_average
if __name__=="__main__":
    a=run_experiment(1,2,0.5,100)
    plt.plot(a)
        
            
            
            
            