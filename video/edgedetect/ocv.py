#! /usr/bin/python

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    edges = cv2.Canny(gray,100,200)
#
    #cv2.imshow('frame', edges)
    cv2.imshow('frame', gray)
    if cv2.waitKey(300) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
