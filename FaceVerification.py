import cv2 as cv
import numpy as np
from deepface import DeepFace
import pickle
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

class FaceVerification:

    def __init__(self):
        print("Building FaceNet32 model..")
        self.facenet = DeepFace.build_model("Facenet512")
        
        print("Building Caffe Face Detector..")
        self.facedetector = cv.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')
        
        print("Building verifier model..")
        pkl_filename = "C:/Users/Zeynep/Desktop/SVMfaceR/pickle_model.pkl"
        with open(pkl_filename, 'rb') as file:
            self.verifier = pickle.load(file)

    def preprocess(self, face):
        img = cv.resize(face, (160, 160))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        return img

    def face_detection(self, frame):
        (h, w) = frame.shape[:2]
        resized_image = cv.resize(frame, (300, 300))
        blob = cv.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.facedetector.setInput(blob)
        detections = self.facedetector.forward()
        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int32")
        cv.rectangle(frame, (startX, startY), (endX, endY), (150, 0, 255), 3)
        face = frame[startY:endY, startX:endX]
        return frame, face

    def face_verify(self, ref_img, frame, flag = 0):
        frame, face = self.face_detection(frame)
        img1_representation = self.facenet.predict(self.preprocess(face))[0,:]
        img2_representation = self.facenet.predict(self.preprocess(ref_img))[0,:]
        output = np.abs(img1_representation - img2_representation)
        return frame, self.verifier.predict([output])

cap = cv.VideoCapture(0)
ref_img = cv.imread("C:/Users/Zeynep/Desktop/FaceRecognition/dataset/snormal/7.png")
fv = FaceVerification()
count = 0
predictions = []
while True:
    ret, frame = cap.read(0)
    count += 1
    frame, prediction = fv.face_verify(ref_img, frame)
    predictions.append(prediction)
    if count % 100 == 0:
        print('% ',sum(predictions), 'verified.')
        predictions = []
    frame = cv.flip(frame, 1)
    cv.imshow('f', frame)
    key = cv.waitKey(10)
    if key == 27:
        break