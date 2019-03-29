# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:49:12 2019

@author: HP
"""

import numpy as np 
import cv2 as cv
from matplotlib import pyplot as plt
image=cv.imread("elephant.jpg")
# creating an array containing rgb values of pixels
image=image.reshape(-1,3)
from sklearn.cluster import KMeans
km=KMeans(n_clusters=6)
km.fit(image)
centers=km.cluster_centers_
centers=np.array(centers,dtype='uint8')
labels=km.labels_
new_image=np.zeros((165000,3),dtype='uint8')
for i in range(len(labels)):
    new_image[i]=centers[labels[i]]
new_image=new_image.reshape(330,500,3)
cv.imshow("segmented image",new_image)
cv.imshow("original image",image.reshape((330,500,3)))    