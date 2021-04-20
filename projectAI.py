# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:22:17 2021

@author: Julien Camboulives - Martin de Foresta - Corentin Fauchet - Lisa Roulet
"""

#Part 1 : Data preprocessing
# a. Importing the data
import pandas as pd

table = pd.read_csv("list_attr_celeba.csv") 

# b. Keeping only useful data
images = table.iloc[:100000,0].values
blackhair = table.iloc[:100000, 9].values

#c .Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
images_train, images_test, blackhair_train, blackhair_test=train_test_split(images,blackhair,test_size=0.2,random_state=0)


# d. Creating directories
import os

#os.mkdir('Images')
#os.chdir('Images')
#os.mkdir('TrainImages')
#os.mkdir('TestImages')

#os.chdir('TrainImages')
#os.mkdir('Blackhair')
#os.mkdir('NoBlackhair')
destTrainBlackhair=os.path.join(os.getcwd(),'Blackhair')
destTrainNoBlackhair=os.path.join(os.getcwd(),'NoBlackhair')

#os.chdir('..')
#os.chdir('TestImages')
#os.mkdir('Blackhair')
#os.mkdir('NoBlackhair')
destTestBlackhair=os.path.join(os.getcwd(),'Blackhair')
destTestNoBlackhair=os.path.join(os.getcwd(),'NoBlackhair')

#os.chdir('..')
#os.chdir('..')

# e. Sorting images into directories
import shutil

baseSrc=os.path.join(os.getcwd(),'C:/Users/33635/Desktop/Intelligence artificielle/project/img_align_celeba')
increment=0
for trainImgId in images_train:
    src=os.path.join(baseSrc,trainImgId)
    if blackhair_train[increment]==-1:
        shutil.copy(src,destTrainNoBlackhair)
    else:
        shutil.copy(src,destTrainBlackhair)
    increment=increment+1

increment=0
for testImgId in images_test:
    src=os.path.join(baseSrc,testImgId)
    if blackhair_test[increment]==-1:
        shutil.copy(src,destTestNoBlackhair)
    else:
        shutil.copy(src,destTestBlackhair)
    increment=increment+1
    
    
#Part 2 Build the CNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D #if video Conv3D (+time)
from keras.layers import MaxPooling2D
from keras.layers import Flatten

#Initializing the CNN
classifier=Sequential()

#Step1- Convolution

classifier.add(Conv2D(filters=32 , kernel_size=(3,3), activation='relu', input_shape=(64,64,3)))
# 32 feature detectors (filters) / based on cpu issues
#strides is equal to 1 by default
#RGB images and so the images are represented in a 3D array
#64x64 as image size also based on cpu ressources and it is a good tradeof
#between model accuracy (image resolution) and cpu

#Step 2- Pooling 
#size/2 if even and size is odd -> size/2+1 
#goal : reduce the size of the feature maps (outputs of the convolution)
#while keeping the most important information (features)
#it will allow us also to reduce the time complexity without reducing the performance model
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Adding a second convolution layer to reduce overfitting
classifier.add(Conv2D(32, kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Step 3 -Flattening
classifier.add(Flatten())

#Step 4 - Full connection
classifier.add(Dense(units=128 , activation='relu'))
#Hidden layer with 128 nodes (neurones)
classifier.add(Dense(units=1, activation='sigmoid'))
#Output layer

#Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 
#categorical cross entropy if multi class

classifier.summary()

#Part 3 : Fitting the CNN to the images
#image augmentation trick
#It allows us to enrich our image dataset without getting
# new images 
# the images that we have rotated, shifted , tweaked etc
# as a result we reduce the overfitting problem

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

training_set=train_datagen.flow_from_directory(
        'Images/TrainImages',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

test_set=test_datagen.flow_from_directory(
        'Images/TestImages',
        target_size=(64,64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=(8000/32),
        epochs=25,
        validation_data=test_set,
        validation_steps=(2000/32))

#Save the model
classifier.save("my_Celebrity_Blackhair_model.h5")

#Load the model
from tensorflow.keras.models import load_model
classifier=load_model('my_Celebrity_Blackhair_model.h5')

#we generate the first prediction
Y_pred=classifier.predict(test_set)
Y_pred=(Y_pred>0.5)
for i in range(0,len(blackhair_test),1):
        if blackhair_test[i]==-1 :
                blackhair_test[i]=0

#for the confusion matrix
#we generate the cm with sklearn first
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(blackhair_test,Y_pred)

from sklearn.metrics import accuracy_score
a=accuracy_score(Y_pred,blackhair_test)
print('Accuracy is:',a*100)

#single prediction 
import numpy as np
from keras.preprocessing import image

test_image=image.load_img('prediction_blonde.jpeg',
                          target_size=(64,64))
test_image=image.img_to_array(test_image).astype('float32')/255
test_image=np.expand_dims(test_image,axis=0)
result=classifier.predict(test_image)
print(result)

training_set.class_indices
if result >=0.5:
    prediction='Cette personne ne semble pas avoir les cheveux noirs'
else:
    prediction='Cette personne semble avoir les cheveux noirs'

print(prediction)
