from inception import *
model = InceptionResNetV1()
import numpy as np
import cv2 as cv

cascade_path = "/home/serkan/Desktop/DrowsinessDetection/DrowsinessDetection/cascades"

pro_txt = os.path.join(cascade_path, 'deploy.prototxt.txt')
model_path = os.path.join(cascade_path, 'res10_300x300_ssd_iter_140000.caffemodel')
ssd_face_model = cv.dnn.readNetFromCaffe(pro_txt, model_path)


def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(160, 160))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def get_face_ssdnet(frame):
    (h, w) = frame.shape[:2]
    resized_image = cv.resize(frame, (300, 300))
    blob = cv.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    ssd_face_model.setInput(blob)
    detections = ssd_face_model.forward()
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int32")
    face = frame[startY:endY, startX:endX]
    cv.rectangle(frame, (startX, startY), (endX, endY), (42, 64, 127), 2)
    return face, frame 


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    face, frame = get_face_ssdnet(frame)
    resized = cv.resize(face, (160, 160))
    resized = img_to_array(resized) 
    resized = np.expand_dims(resized, axis = 0) 
    resized = preprocess_input(resized) 
    img1_representation = l2_normalize(model.predict(preprocess_image('test.jpg'))[0,:])
    img2_representation = l2_normalize(model.predict(resized)[0,:])

    cos_dist = findCosineDistance(img1_representation, img2_representation)
    if cos_dist < 0.07:
        print('ayni kisiler')
    else:
        print('ayni kisi degiller')
    
    print(cos_dist)
    cv.imshow('Frame',frame)
    if cv.waitKey(1) == 27: 
        break

cap.release()
