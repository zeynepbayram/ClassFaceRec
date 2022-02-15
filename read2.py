import cv2 as cv
from pyzbar.pyzbar import decode
import numpy as np
import imutils
cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
camera = True
while camera == True:
    ret, frame = cap.read()
   
    for code in decode(frame):
        print(code.type)
        print(code.data.decode("utf-8"))
        
    cv.imshow("f", frame)
    cv.waitKey(1)