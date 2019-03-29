# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:32:59 2019

@author: HP
"""

import numpy as np
import cv2 as cv
import operator
import os
class data:
    def __init__(self,distance,name):
        self.distance=distance
        self.name=name
def find_distance(x1,x2):
    s=np.sum((x1-x2)**2)
    s=np.sqrt(s)
    return s
def KNN(X,Y,x,k):
    distances=[]
    for i in range(len(X)):
        check=X[i]
        curr_distance=find_distance(X[i],x)
        d=data(curr_distance,Y[i])
        distances.append(d)
    distances=sorted(distances,key=operator.attrgetter("distance"))    
    distances=distances[:k]
    count_dict={}
    maxx=0
    prediction=""
    for i in range(len(distances)):
        if (distances[i].name in count_dict.keys())==False:
            count_dict[distances[i].name]=1
        count_dict[distances[i].name]+=1
        if maxx<count_dict[distances[i].name]:
            maxx=count_dict[distances[i].name]
            prediction=distances[i].name
    return prediction    
face_cascade=cv.CascadeClassifier("/Users/HP/Desktop/ML codes/haarcascade_frontalface_alt.xml")
"""file_name=input("Enter the name of the person: ")
n=1
face_data=[]
file_path='/.data/'
cap=cv.VideoCapture(0)
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    face=face_cascade.detectMultiScale(frame,1.3,5)
    #print(face)
    if face==():
        continue
    x,y,w,h=face[0]
    cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    if n%10==0:
        face_data.append(frame)
    n+=1 
    cv.imshow("frame",frame)    
    key_pressed=cv.waitKey(1)
    if key_pressed & 0xFF==ord('e'):
        break;
np.save("/Users/HP/Desktop/ML codes/"+file_name+".npy",face_data)
cap.release()
cv.destroyAllWindows()"""
record={}
Xtrain=[]
yTrain=[]
face_data=[]
# creating the input dataset from the file
for f in os.listdir("/Users/Hp/Desktop/ML codes/data/"):
    if f.endswith('.npy'):
       record[f[:-4]]= np.load("/Users/Hp/Desktop/ML codes/data/"+f)
       face_data.append(record[f[:-4]])
       #Adding equal no of y data
       for i in range(len(record[f[:-4]])):
           yTrain.append(f[:-4])
Xtrain=np.concatenate(face_data,axis=0)
X_train2=[]
# creating X_train2
for i in range(len(face_data)):
    for j in range(len(face_data[i])):
        X_train2.append(face_data[i][j])
#Xtrain=face_data       
yTrain=np.asarray(yTrain)
yTrain.reshape(-1,1)
cap=cv.VideoCapture(0)
while True:
      ret,frame=cap.read()
      if ret==False:
          continue
      face=face_cascade.detectMultiScale(frame,1.3,5)
      # getting the face ROI
      offset=10
      if face==():
          continue
      x,y,w,h=face[0]
      face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
      face_section=cv.resize(face_section,(100,100))
      prediction=KNN(Xtrain,yTrain,face_section,5)
      cv.putText(frame,prediction,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv.LINE_AA)
      #cv.putText(frame,prediction,(x,y-10),cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),cv.LINE_AA)
      cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
      cv.imshow("frame prediction",frame)
      keypressed=cv.waitKey(1)
      if keypressed & 0xFF==ord('e'):
          cap.release()
          cv.destroyAllWindows()
          break
      