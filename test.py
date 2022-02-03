from keras.models import load_model
# load the model
model = load_model('/home/serkan/Desktop/FaceVerification/ClassFaceRec/facenet_keras.h5')
# summarize input and output shape
print(model.inputs)
print(model.outputs)