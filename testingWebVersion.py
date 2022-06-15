from keras.models import load_model
import numpy as np
import cv2 as cv
from signClasses import classes
from PIL import Image
import streamlit as st
##################################
model = load_model('model.h5')
width = 640
height = 480
# image = cv.imread("testSign.JPG")

#cap = cv.VideoCapture(URL)
threshold = 0.98
#cap.set(3, width)
#cap.set(4, height)
##################################

def preprocessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv.equalizeHist(img)
    img = img/255  # normalizing image.
    img = cv.resize(img, (32, 32))  # resizing it.
    img = img.reshape(1, 32, 32, 1)
    return img

def predict(img):
    processedImage = preprocessing(img)
    processedImage = np.array(processedImage)
    result = model.predict(processedImage)
    predictedClass = np.argmax(result)
    probVal = np.amax(result)
    return classes[predictedClass], probVal


st.title("Road sign detection")
uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image)
    imageData = np.asarray(image)
    predicted, probVal = predict(imageData)
    st.write(predicted)
    st.write(probVal)
    st.write("Created by: Akshma Atreja, Katerina Gkoltsou, Mina Hany, Rana Sahin \n students of Warsaw University Of Technology in Poland")
