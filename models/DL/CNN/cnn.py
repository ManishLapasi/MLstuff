from requests import models
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


## augmentation class for images
## this includes zooming, flipping, etc. prevents overfitting
## this also includes feature scaling (rescale option)
idg_train = ImageDataGenerator(rescale=1./255,horizontal_flip=True,shear_range=0.2,zoom_range=0.2)

## augment training dataset
train_dataset = idg_train.flow_from_directory('../../../../dataset/training_set',target_size=(64,64),batch_size=32,class_mode='binary')

## feature scaling on test dataset
idg_test = ImageDataGenerator(rescale=1./255)
test_dataset = idg_test.flow_from_directory('../../../../dataset/test_set/',target_size=(64,64),batch_size=32,class_mode='binary')

## initialise cnn
cnn = tf.keras.models.Sequential()

## add convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu',input_shape=[64,64,3]))

## add pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

## add second convolutional layer and second pooling layer
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

## add flattening layer
cnn.add(tf.keras.layers.Flatten())

## fully connected layer
cnn.add(tf.keras.layers.Dense(units=128,activation='relu'))

## output layer
cnn.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

## compile CNN
cnn.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

## train cnn
cnn.fit(x = train_dataset, validation_data=test_dataset, epochs=10)

## prediction

import numpy as np
from tensorflow.keras.preprocessing import image

train_dataset.class_indices

for img in ['../../../../dataset/single_prediction/cat_or_dog_1.jpg','../../../../dataset/single_prediction/cat_or_dog_2.jpg']:
    test_image = image.load_img(img,target_size=(64,64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = cnn.predict([[test_image]])
    if(result[0][0]==1):print('dog')
    else: print('cat')

