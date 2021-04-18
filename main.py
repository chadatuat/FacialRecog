# Chad Weirick
# UAT
# facial recognition .whl file courtesy of https://pypi.org/project/face-recognition/#files
# Based on walkthrough from Tutorialspoint by abhilash
# Datamodels supplied come from the above mentioned tutorial

import cv2
import dlib
import face_recognition
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

#load models for emotionals
exp_model = model_from_json(open("dataset/facial_expression_model_structure.json","r").read())
exp_model.load_weights('dataset/facial_expression_model_weights.h5')

#load models for age detections
age_protext = "dataset/age_deploy.prototxt"
age_caffemodel = "dataset/age_net.caffemodel"
age_cov_net = cv2.dnn.readNet(age_caffemodel, age_protext)

#declare the emotions and age_groupings labels and populate them
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
age_groupings = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

source_image = cv2.imread('images/sample3.jpg')
face_locations = face_recognition.face_locations(source_image, model="HoG")

print ("Total number of faces = "  + str(len(face_locations)))

for index, this_face_loc in enumerate(face_locations):
    top_loc, right_loc, bottom_loc, left_loc = this_face_loc
    this_face_image = source_image[top_loc:bottom_loc,left_loc:right_loc]
    print("face #" + str(index + 1))
    print("found at top/right/bottom/left grid coordinates: " + str(top_loc) + "," + str(right_loc) + "," + str(bottom_loc) + "," + str(left_loc))

    AGE_GENDER_MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    face_blob = cv2.dnn.blobFromImage(this_face_image, 1, (227, 227), AGE_GENDER_MODEL_MEAN_VALUES, swapRB=False)
    age_cov_net.setInput(face_blob)
    age_predictions = age_cov_net.forward()
    age = age_groupings[age_predictions[0].argmax()]
    print("age estimate: " + age)

    # need a BGR array in 2 coluins via numpy, 48/48
    this_face_image = cv2.cvtColor(this_face_image, cv2.COLOR_BGR2GRAY)
    this_face_image = cv2.resize(this_face_image, (48, 48))
    img_pixels = image.img_to_array(this_face_image)
    img_pixels = np.expand_dims(img_pixels, axis=0)
    img_pixels /= 255

    #pass through to the prediction models
    exp_predictions = exp_model.predict(img_pixels)
    max_index = np.argmax(exp_predictions[0])
    emotion_detected = emotions[max_index]

    print("emotional state estimated to be: " + emotion_detected)
