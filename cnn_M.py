# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:46:27 2019

@author: emili
"""
##Importing Keras Libraries
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

##Initializing the CNN
classifier = Sequential()

##Step 1: Convolution
classifier.add(Convolution2D(32,3,3, input_shape = (64, 64 ,3), activation = "relu")) 
##Param 1: nb_filter= number of featura detectors and dimensions rowsXcolumns
##Param 2: input_shape=(3,256,256) número de canales y tamaño de las imágenes de input 
##Param 3: Activation Function para obtener non-linearity 

##Step 2: Max Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))
##Param 1: Tamaño del pool Map (2X2)

##Al aplicar los dos pasos anteriores conseguimos información de la imágen no solamente sobre sus features
## si no también sobre sus relaciones espaciales.

##Podemos agregar una segunda convolutional layer para mejorar el procesamiento de la imágen
classifier.add(Convolution2D(32,3,3, activation = "relu")) 
classifier.add(MaxPooling2D(pool_size = (2,2)))

##Step 3: Flattening
classifier.add(Flatten())

##Usaremos el vector del Flatten para el input de nuestra ANN que tendrá fully connected layers
##ANN son geniales clasificadores para problemas no lineales y la clasificación de imágenes es un problema
## NO LINEAL (afortunadamente)

##Step 4: Full Connection
##Utilizamos Dense para crear una fully connected layer
classifier.add(Dense(output_dim = 128, activation = "relu"))
##Param 1: Número de nodos en nuestra fully connected hidden layer

classifier.add(Dense(output_dim = 1, activation = "sigmoid"))
##Output Layer

## Compiling CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

##Part 2: Fitting CNN to the images
from keras.preprocessing.image import ImageDataGenerator
 
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

##Esta parte es posible gracias a la manera en que organizamos nuestro folders de manera manual.
training_set = train_datagen.flow_from_directory(
                'dataset/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')


test_set = test_datagen.flow_from_directory(
                'dataset/test_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='binary')

classifier.fit_generator(
                    training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=2000)

