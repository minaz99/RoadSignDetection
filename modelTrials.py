from keras.models import load_model
import numpy as np
import cv2 as cv
from keras.utils.np_utils import to_categorical

from signClasses import classes
import pandas as pd
import os
model = load_model('model.h5')
pathTest = "DataSets/Test"  # path for testing datasets.
classNo = []  # list for storing image class no.
classNoTest = []
imagesTest = []
imagesProcessedForTraining = []
imagesProcessedForTesting = []
listOfSetsTest = os.listdir(pathTest)
images = []  # list for storing imported images.

noOfClassesTesting = len(listOfSetsTest)


def preprocessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv.equalizeHist(img)
    img = img/255  # normalizing image.
    img = cv.resize(img, (32, 32))  # resizing it.
    return img

def prepareDataSets(noOfSets, PATH, val):
    imgList = []
    classNoList = []
    if val == 0:  # training sets.
        for x in range(0, noOfSets):
            dataset = os.listdir(PATH + "/" + str(x))  # getting first directory.
            for y in dataset:  # getting images inside directory.
                image = cv.imread(PATH + "/" + str(x) + "/" + y)  # storing it as an img.
                imgList.append(image)
                classNoList.append(x)  # storing class no.
            print(x)
    elif val == 1:  # testing sets.
        dataset = os.listdir(PATH + "/")
        for x in dataset:
            image = cv.imread(PATH + "/" + x)  # storing it as an img.
            imgList.append(image)
        print("Finished .........")
        data = pd.read_csv("DataSets/Test.csv")
        for x in data.ClassId:
            classNoList.append(x)
    return imgList, classNoList

imagesTest, classNoTest = prepareDataSets(noOfClassesTesting, pathTest,1)




for x in imagesTest:
    imagesProcessedForTesting.append(preprocessing(x))



imagesProcessedForTesting = np.array(imagesProcessedForTesting)
classNoTest = np.array(classNoTest)
ClassNoTestBeforeCategorical = classNoTest




# reshaping arrays to add a depth of 1 for CNN.
imagesProcessedForTesting = imagesProcessedForTesting.reshape(imagesProcessedForTesting.shape[0], imagesProcessedForTesting.shape[1], imagesProcessedForTesting.shape[2], 1)

classNoTest = to_categorical(classNoTest, noOfClassesTesting)

#print(imagesProcessedForTesting.shape)
#print(classNoTest.shape)

#model.evaluate(imagesProcessedForTesting,classNoTest)

