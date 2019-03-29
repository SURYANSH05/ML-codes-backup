# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 23:35:22 2019

@author: HP
"""

import numpy as np
import cv2 as cv
cap=cv.VideoCapture(0)
Face_cascade=cv.CascadeClassifier("/Users/HP/Desktop/ML codes/haarcascade_frontalface_alt.xml")
face_data=[]
dataset='./data/'
skip=0
file_name=input("Enter the name of the person ")
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    faces=Face_cascade.detectMultiScale(frame,1.3,5)
    #qfaces = sorted(faces,key=lambda f:f[2]*f[3])
   # faces=sorted(faces,key=lambda f:f[2]*f[3])
    #face_section=0
    for face in faces[-1:]:
        x,y,w,h=face
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        
        # creating a face_section
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv.resize(face_section,(100,100))
        skip+=1
        if(skip%10==0):
          face_data.append(face_section)
          print(len(face_data))
    cv.imshow("frame",frame)
    #cv.imshow("face_section",face_section)
    key_pressed=cv.waitKey(1) & 0xFF
    if key_pressed==ord('q'):
        break
face_data=np.asarray(face_data)
np.save(dataset+file_name+'.npy',face_data)    
cap.release()
cv.destroyAllWindows()