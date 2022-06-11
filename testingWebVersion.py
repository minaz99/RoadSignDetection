from keras.models import load_model
import numpy as np
import cv2
from signClasses import classes
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import streamlit as st
import threading
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
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv2.equalizeHist(img)
    img = img/255  # normalizing image.
    img = cv2.resize(img, (32, 32))  # resizing it.
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
