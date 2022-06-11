import os
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sb
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from sklearn.metrics import accuracy_score, classification_report

##################
path = "DataSets/Train"  # path for training datasets.
pathTest = "DataSets/Test"  # path for testing datasets.
images = []  # list for storing imported images.
imagesTest = []
classNo = []  # list for storing image class no.
classNoTest = []
listOfSets = os.listdir(path)  # list of directories.
listOfSetsTest = os.listdir(pathTest)
noOfClasses = len(listOfSets)  # number of classes.
noOfClassesTesting = len(listOfSetsTest)
imagesProcessedForTraining = []
imagesProcessedForTesting = []
##################


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


def preprocessing(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # converting image to grayscale.
    img = cv.equalizeHist(img)
    img = img/255  # normalizing image.
    img = cv.resize(img, (32, 32))  # resizing it.
    return img


# importing datasets (training and testing).
print("importing training classes..........")
images, classNo = prepareDataSets(noOfClasses, path,0)
print("importing testing classes..........")
imagesTest, classNoTest = prepareDataSets(noOfClassesTesting, pathTest,1)


trainingClassesCount = []
testingClassesCount = []

for i in range(0,noOfClasses):
    counterTraining = 0
    counterTesting = 0
    for j in range(0,len(classNo)):
        if(classNo[j] == i):
            counterTraining = counterTraining + 1
    for k in range(0,len(classNoTest)):
        if(classNoTest[k] == i):
            counterTesting = counterTesting + 1
    testingClassesCount.append(counterTesting)
    trainingClassesCount.append(counterTraining)


plt.figure(figsize=(10,5))
plt.bar(range(0,43),trainingClassesCount, width= 0.8, color='#1797c2')
plt.title("Distribution of training datasets")
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(10,5))
plt.bar(range(0,43),testingClassesCount, width= 0.8, color='#eb3462')
plt.title("Distribution of testing datasets")
plt.xlabel("Class ID")
plt.ylabel("Count")
plt.show()

# processing images in testing and training sets.
for x in images:
    imagesProcessedForTraining.append(preprocessing(x))
for x in imagesTest:
    imagesProcessedForTesting.append(preprocessing(x))

# Conversion of list to array.
imagesProcessedForTraining = np.array(imagesProcessedForTraining)
classNo = np.array(classNo)
imagesProcessedForTesting = np.array(imagesProcessedForTesting)
classNoTest = np.array(classNoTest)
ClassNoTestBeforeCategorical = classNoTest



# reshaping arrays to add a depth of 1 for CNN.
imagesProcessedForTraining = imagesProcessedForTraining.reshape(imagesProcessedForTraining.shape[0], imagesProcessedForTraining.shape[1], imagesProcessedForTraining.shape[2], 1)
imagesProcessedForTesting = imagesProcessedForTesting.reshape(imagesProcessedForTesting.shape[0], imagesProcessedForTesting.shape[1], imagesProcessedForTesting.shape[2], 1)


# Now we augment the image for better training.
imgDataGenerator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.2, shear_range=0.1, rotation_range=10, horizontal_flip=False)
# Now we will help our image generator to calculate some statistics before performing the transformations.

imgDataGenerator.fit(imagesProcessedForTraining)  # so we generate the images as the training is going, we just want the
# imgDataGenerator to know a bit about the data sets before we send it for the training process.


# transforming arrays of class numbers to a matrix which has binary values,
# where columns equal to number of data in array, this is needed to transform training data for
# using in loss function, categorical crossentropy
# before passing it the model for training, to_categorical mainly used when training data uses classes as numbers.
classNo = to_categorical(classNo, noOfClasses)
classNoTest = to_categorical(classNoTest, noOfClassesTesting)
batch = imgDataGenerator.flow(imagesProcessedForTraining, classNo, batch_size=20)
i, l=next(batch)

#print(classNo[0])

# creating Convolutional Neural Network (CNN)

stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, 1)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
history = model.fit(imagesProcessedForTraining, classNo,epochs=32, verbose=1)


predict = model.predict(imagesProcessedForTesting)

result = []

for i in range(0, len(classNoTest)):
    result.append(np.argmax(predict[i]))

resultArray = np.array(result)
resultArray.reshape((1,resultArray.shape[0]))
print("Classification Report: \n", classification_report(ClassNoTestBeforeCategorical, resultArray))
hotNCold = tf.math.confusion_matrix(ClassNoTestBeforeCategorical, resultArray)
# print(hotNCold)
plt.figure(figsize= (60,60))
sb.set()
heatMap = sb.heatmap(hotNCold,annot=True,fmt='d',cmap="YlGnBu")
plt.xlabel("predicted")
plt.ylabel("Actual")
plt.show()
# model.save("model2.h5")


plt.figure(0)
plt.plot(history.history['acc'], label='training accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()
plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()


# do something about precision and recall not just accuracy






















