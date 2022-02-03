import cv2 as cv
import os
import numpy as np
import face_recognition
from simple_facerec import SimpleFacerec

class FaceDetect:
    def __init__(self, model = "res10_300x300_ssd_iter_140000.caffemodel", path="deploy.prototxt.txt"):
        self.model = model
        self.path = path
        self.ssd_face_model = cv.dnn.readNetFromCaffe(path, model)
    
    def get_face_ssdnet(self, frame, image_size = 224):
        (h, w) = frame.shape[:2]
        resized_image = cv.resize(frame, (300, 300))
        blob = cv.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        #up here returns a blob which is our input image after mean subtraction, normalizing, and channel swapping.
        self.ssd_face_model.setInput(blob)
        detections = self.ssd_face_model.forward()
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int32")
        cv.rectangle(frame, (startX, startY), (endX, endY), (150, 0, 255), 3)
        return frame
    def recog_face(self, frame):
        face = self.get_face_ssdnet(frame)
        

webcam = cv.VideoCapture(0)
count = 1
face_detector = FaceDetect()
while True:
    (_, im) = webcam.read()
    x = face_detector.get_face_ssdnet(im)
    cv.imshow('OpenCV', x)
    count += 1
    key = cv.waitKey(10)
    if key == 27:
        break

