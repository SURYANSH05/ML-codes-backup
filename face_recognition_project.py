# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 13:19:00 2019

@author: HP
"""

import numpy as np
import cv2 as cv
cap=cv.VideoCapture(0)
Face_cascade=cv.CascadeClassifer("/Users/HP/Desktop/ML codes/haarcascade_frontalface_alt.xml")
file_name=input("Enter the name of the person")
n=1
face_data=[]
file_path='/.data/'
while true:
    ret,frame=cap.read()
    if ret==False:
        continue
    if n%10==0:
        face_data.append(frame)
    n+=1
    cv.imshow("frame",frame)    
    key_pressed=cv.waitKey(2)
    if key_pressed & 0xFF==ord('e'):
        break;
np.save(file_path+file_name.npy)
cap.release()
cap.destroyAllWindows()        