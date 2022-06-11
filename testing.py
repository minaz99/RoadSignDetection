from keras.models import load_model
import numpy as np
import cv2 as cv
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


mainWindow = Tk()
counter = 0
label = Label(mainWindow,height=300, width=300)

def openFn():
    filename = filedialog.askopenfilename(title='open')
    list = mainWindow.pack_slaves()
    global counter
    global realTimeEntry
    for l in list:
        print(l.pack)
        if (l.widgetName == "canvas" or l.widgetName == "button") and counter == 0:
            l.destroy()
        elif (l.widgetName == "label" or l.widgetName == "button" or l.widgetName == "canvas") and counter > 0:
            l.destroy()
    counter += 1
    return filename


mainWindow.geometry('650x650')
mainWindow.title("Road Sign Recognition")
mainWindow.configure(bg="#670350")
mainWindow.resizable(False, False)
spacer2 = Label(mainWindow, text="",bg="#670350").pack()
canv = Canvas(mainWindow,width=300, height=300, bg="black").pack()
spacer3 = Label(mainWindow, text="",bg="#670350").pack()
#cap = cv.VideoCapture(URL)
cap = cv.VideoCapture(0)
realTimeEntry = 0
def classify():

    loadedImg = openFn()
    canvas = Canvas(mainWindow, width=300, height=100, bg="#e69a4c")
    img = Image.open(loadedImg)
    img = img.resize((300, 300))
    imgTk = ImageTk.PhotoImage(img)
    imageToBePredicted = cv.imread(loadedImg)
    predicted, probVal = predict(imageToBePredicted)
    print(predicted)
    if probVal > threshold:
        pip = 1
        canvas.delete('all')
        predictAndProb = "\n" + predicted + "\n \n \n"

        canvas.create_text(150,50,text=predictAndProb,fill="#08508a",font='Helvetica 15 bold')
        canvas.create_text(150,50, text=probVal,fill="#0DA607",font='Helvetica 15 bold')
        canvas.pack(side='bottom')
    if counter > 1:
        spacer4 = Label(mainWindow, text="", bg="#670350").pack()
    label2 = Label(mainWindow,image=imgTk)
    label2.pack()
    spacer5 = Label(mainWindow, text="", bg="#670350").pack()
    newRealTime = Button(mainWindow, text="Real time detection", height=3, width=20, bg="#031e67", fg="white",command=realTime).pack()
    newClassification = Button(mainWindow, text="Sign classification", height=3, width=20, bg="#031e67", fg="white", command=classify).pack()

    mainWindow.mainloop()


def realTime():
    global counter
    global realTimeEntry
    if realTimeEntry == 0:
        counter = 0
    realTimeEntry += 1
    list = mainWindow.pack_slaves()
    for l in list:
        if (l.widgetName == "canvas" or l.widgetName == "button" or l.widgetName == "label") and counter == 0:
            l.destroy()
        elif l.widgetName == "canvas" and counter >= 0:
            l.destroy()
    counter += 1
    cv2image = cv.cvtColor(cap.read()[1], cv.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    canvas = Canvas(mainWindow, width=300, height=100, bg="#e69a4c")
    # Convert image to PhotoImage
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    predicted, probVal = predict(cv2image)
    print(predicted)
    label.pack()


    if probVal > threshold:

        canvas.delete('all')
        predictAndProb = "\n" + predicted + "\n \n \n"
        myText = canvas.create_text(150, 50, text=predictAndProb, fill="blue", font='Helvetica 15 bold')
        myText2 =  canvas.create_text(150, 50, text=probVal, fill="#0DA607", font='Helvetica 15 bold')
        canvas.itemconfig(myText, text=" ")
        canvas.itemconfig(myText2, text= " ")
        canvas.itemconfig(myText, text=predictAndProb)
        canvas.itemconfig(myText2,text=probVal)

        canvas.pack(side='bottom')
        # Repeat after an interval to capture continuously
    if counter == 1:
        spacer5 = Label(mainWindow, text="", bg="#670350").pack()
    if counter == 1:
        newRealTime = Button(mainWindow, text="Real time detection", height=3, width=20, bg="#031e67", fg="white",command=realTime).pack()
        newClassification = Button(mainWindow, text="Sign classification", height=3, width=20, bg="#031e67", fg="white", command=classify).pack()
    label.after(20, realTime)


def about():
    aboutWin = Tk()
    aboutWin.geometry('550x400')
    aboutWin.title("About")
    aboutWin.configure(bg="#670350")
    aboutWin.resizable(False, False)
    canvas = Canvas(aboutWin, width=550, height=400, bg="#670350")
    aboutUs = "Created by Akshma Atreja, Katerina Gkoltsou, Mina Hany and Rana Sahin." + "\n" + "             Students of Warsaw University Of Technology in Poland." + "\n\n" + "                            Road Sign Recognition version 1.4.2"
    canvas.create_text(275,200,text=aboutUs, fill="white", font='Helvetica 10 bold')
    canvas.pack()
    aboutWin.mainloop()


realT = Button(mainWindow,text="Real time detection", height=3, width=20,bg="#031e67",fg="white",command=realTime).pack()
classification = Button(mainWindow,text="Sign classification", height=3, width=20,bg="#031e67",fg="white", command=classify).pack()
About = Button(mainWindow, text="About", height=2, width=15, bg="#031e67", fg="white", command=about).pack()
mainWindow.mainloop()




