# Гурулев Александр, группа 8Е51 - Классификатор предметов одежды по их изображениям

Данная программа представляет собой классификатор предметов одежды по их изображениям. В данном случае, используется датасет Fashion MNIST, представляющий собой обучающий набор из 60000 ч/б изображений одежды размером 28x28, который разбиваем на обучающую выборку из 55000 изображений, тестовый набор из 10000 изображений и валидационный набор из 5000 изображений.

Архитектура нейронной сети представляет собой многослойную структуру, где:

1 слой - сверточный с 64 фильтрами, акт.функция - relu

2 слой - сверточный с 32 фильтрами, акт.функция - relu

3 слой - полносвязный, 256 нейронов, акт.функция - relu

Выходной слой - полносвязный с 10 нейронами и акт.функцией softmax

Ссылка на датасет - https://github.com/zalandoresearch/fashion-mnist#loading-data-with-other-machine-learning-libraries
