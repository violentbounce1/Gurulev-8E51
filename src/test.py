import matplotlib.pyplot as plt
import pickle

from keras.models import model_from_json

# загрузим файл модели

json_file = open("model.json", "r")
loaded_model_json = json_file.read()
json_file.close()

# создаем модель на основе загруженных данных

loaded_model = model_from_json(loaded_model_json)

# загружаем веса в модель

loaded_model.load_weights("model.weights.best.hdf5")

# компилируем модель

loaded_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# загрузим pickle-контейнер с данными о точности модели

with open('history.pickle', 'rb') as f:
    history = pickle.load(f)

# построим график зависимости точности на тестовых и валидационных данных от эпохи

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('acc_fig.png', format='png', dpi=100)
plt.show()