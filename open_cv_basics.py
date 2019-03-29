# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 14:12:13 2019

@author: HP
"""

import cv2 as cv
import matplotlib.pyplot as plt
cap=cv.VideoCapture(0)
Face_cascade=cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
while True:
    ret,frame=cap.read()
    gray_frame=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    if ret==False:
        continue
    faces=Face_cascade.detectMultiScale(frame,1.3,5)
    cv.imshow("Video ",frame)
    #cv.imshow("Gray frame",gray_frame)
    for(x,y,w,h) in faces:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    cv.imshow("Video ",frame)    
    key=cv.waitKey(1) & 0xFF
    if key==ord('q'):
        break
cap.release()
cv.destroyAllWindows() 