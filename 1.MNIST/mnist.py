# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 20:05:05 2020

@author: Eason
"""

import tensorflow as tf
import datetime
from matplotlib import pyplot as plt # show image

# load mnist dataset
mnist = tf.keras.datasets.mnist

# load training and testing dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0

# show number of training dataset 
print(len(x_train))
# show the dimension of a training data
print(x_train[0].shape)


# buile neural network
model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'), # dense full connection layer
        tf.keras.layers.Dropout(0.2), # prevent over training
        tf.keras.layers.Dense(10, activation='softmax')
        ])

# Flatten :  keep the first dimension (batch size) and change the remain multi-dimension data to one dimension
# that is, (None, 32,32,3) -> (None, 3072)

# setting backpropagation function
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# create tensorboard in visialbe UI
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# connect the training dataset to neural network
model.fit(x_train, y_train,
          epochs=5,
          callbacks=[tensorboard_callback])
# use testing dataset
model.evaluate(x_test, y_test, verbose=2) # verbose: showing process mode: 0,1,2

# show testing image result
plt.imshow(x_test[0])
plt.show()
