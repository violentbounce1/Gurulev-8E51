# импортируем необходимые библиотеки

import random
import tensorflow as tf
import pickle

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers.core import Flatten


# импортируем датасет Fashion MNIST

from keras.datasets import fashion_mnist

((x_train, y_train), (x_test, y_test)) = fashion_mnist.load_data()

# определим классы изображений

fashion_mnist_labels = ["T-shirt/top",  # класс 0
                        "Trouser",      # класс 1
                        "Pullover",     # класс 2
                        "Dress",        # класс 3
                        "Coat",         # класс 4
                        "Sandal",       # класс 5
                        "Shirt",        # класс 6
                        "Sneaker",      # класс 7
                        "Bag",          # класс 8
                        "Ankle boot"]   # класс 9

# проверим случайное изображение

img_index = random.randint(0, 59999)
label_index = y_train[img_index]
print ("y = " + str(label_index) + " " +(fashion_mnist_labels[label_index]))

# нормализуем данные

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# выведем число обучающих и тестовых изображений

print("Number of train data - " + str(len(x_train)))
print("Number of test data - " + str(len(x_test)))

# разобьем данные на тестовую и валидационную выборку

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

# изменим вид входных данных с (28, 28) на (28, 28, 1) для работы с Conv2D

w, h = 28, 28
x_train = x_train.reshape(x_train.shape[0], w, h, 1)
x_valid = x_valid.reshape(x_valid.shape[0], w, h, 1)
x_test = x_test.reshape(x_test.shape[0], w, h, 1)

# one-hot encoding - имеем изображения 10 классов

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_valid = tf.keras.utils.to_categorical(y_valid, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# выведем количество изображений во всех наборах данных

print(x_train.shape[0], 'train set')
print(x_valid.shape[0], 'validation set')
print(x_test.shape[0], 'test set')

# создадим архитектуру нейронной сети

model = Sequential()

model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# посмотрим количество параметров модели и т.п.

model.summary()

# компиляция модели и сохранение модели с лучшими весами после каждой эпохи

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, save_best_only=True)

history = model.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=(x_valid, y_valid), callbacks=[checkpointer])

# генерируем описание модели в формате json

model_json = model.to_json()

# записываем модель в файл

json_file = open("model.json", "w")
json_file.write(model_json)
json_file.close()

# сохраняем данные о точности модели в pickle-контейнер

with open('history.pickle', 'wb') as f:
    pickle.dump(history, f)