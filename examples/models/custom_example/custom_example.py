#Adapted from tutorial: https://elitedatascience.com/keras-tutorial-deep-learning-in-python

import numpy as np
from keras.models import Sequential
# importing layers permitted by the conversion tool
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, DepthwiseConv2D, Flatten, Concatenate
from keras.utils import np_utils
from keras.datasets import mnist 
from matplotlib import pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

     
model = Sequential()

model.add(Conv2D(32, 3, activation='relu', input_shape=(1,28,28)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=32, nb_epoch=1, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)

print(score)

with open("custom_example.json", "w") as text_file:
    text_file.write(model.to_json())
model.save_weights("custom_example.h5")
