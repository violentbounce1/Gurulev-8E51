# импортируем необходимые библиотеки
import os
import numpy as np
import tensorflow as tf
import keras

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, Flatten
from keras.optimizers import adam
from keras.layers.core import Flatten

# импортируем датасет MNIST Fashion

from keras.datasets import fashion_mnist

((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

# производим reshape обучающей и тестовой части датасета

x_test = np.reshape(x_test, (x_test.shape[0], 28, 28, 1))
x_train = np.reshape(x_train, (x_train.shape[0], 28, 28, 1))

# проверим размерность обучающего и тестового наборов

print("Training set (images) shape: {shape}".format(shape=x_train.shape))
print("Training set (labels) shape: {shape}".format(shape=y_train.shape))
print("Test set (images) shape: {shape}".format(shape=x_test.shape))
print("Test set (labels) shape: {shape}".format(shape=y_test.shape))

# обучающие образцы датасета

num_train, depth, height, width = x_train.shape

# Fashion MNIST имеет 10 уникальных классов

num_classes = np.unique(y_train).shape[0]

# One-hot encoding

Y_train = keras.utils.to_categorical(y_train, num_classes)
Y_test = keras.utils.to_categorical(y_test, num_classes)

# создаем структуру нейронной сети - полносвязный слой с 784 нейронами, сверточный с 20, два полносвязных со 128, выходной слой с 10 - 10 классов

model = Sequential()
model.add(Dense(784, activation='relu'))
model.add(Conv2D(20, (5, 5), activation='relu', input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, Y_train, epochs=3, batch_size=32)
score = model.evaluate(x_test, Y_test, batch_size=32)
print(np.mean(score)*100)

