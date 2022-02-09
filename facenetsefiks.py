import cv2
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os

cascade_path = "C:/Users/Zeynep/Desktop/FaceD"
pro_txt = os.path.join(cascade_path, 'deploy.prototxt.txt')
model_path = os.path.join(cascade_path, 'res10_300x300_ssd_iter_140000.caffemodel')
ssd_face_model = cv2.dnn.readNetFromCaffe(pro_txt, model_path)

model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

from keras.models import model_from_json
model.load_weights("C:/Users/Zeynep/Desktop/deepface/vgg_face_weights.h5")

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_image_rt(image):
    img = cv2.resize(image, (224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

epsilon = 0.50

def get_face_ssdnet(frame):
    (h, w) = frame.shape[:2]
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    ssd_face_model.setInput(blob)
    detections = ssd_face_model.forward()
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int32")
    face = frame[startY:endY, startX:endX]
    cv2.rectangle(frame, (startX, startY), (endX, endY), (42, 64, 127), 2)
    return face

def verifyFace(img1, img2):
    #img1_representation = vgg_face_descriptor.predict(preprocess_image('C:/Users/Zeynep/Desktop/SVMfaceR/dataset/detectedfaces/%s' % (img1)))[0,:]
    img1_representation = vgg_face_descriptor.predict(preprocess_image_rt(img1))[0,:]
    img2_representation = vgg_face_descriptor.predict(preprocess_image_rt(img2))[0,:]
    cosine_similarity = findCosineSimilarity(img1_representation, img2_representation)
    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    
    print("Cosine similarity: ",cosine_similarity)
    print("Euclidean distance: ",euclidean_distance)
    
    if(cosine_similarity < epsilon):
        #print("verified... they are same person")
        return 1
    else:
        #print("unverified! they are not same person!")
        return 0
predictions = []
cap = cv2.VideoCapture(0)
#'C:/Users/Zeynep/Desktop/images/z121.jpg'
ref = cv2.imread('C:/Users/Zeynep/Desktop/images/s37.jpg')
#ref = get_face_ssdnet()
while True:
    ret, frame = cap.read(0)
    face = get_face_ssdnet(frame)
    predictions.append(verifyFace(frame, ref))
    if len(predictions) == 100:
        print("%", sum(predictions))
        predictions = []
    cv2.imshow('f', frame)
    key = cv2.waitKey(10)
    if key == 27:
        break
cap.release()