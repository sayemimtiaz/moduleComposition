import keras
from keras.datasets import mnist, fashion_mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.utils import shuffle


def get_mnist_data(hot=True):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if hot:
        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test, 10


def get_fmnist_data(hot=True):
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    if hot:
        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

    return x_train, y_train, x_test, y_test, 10


def unarize(x_train_original, y_train_original, x_test_original, y_test_original, class1, label1):
    # train data change
    x_train = x_train_original[y_train_original == class1]
    y_train = []
    for x in y_train_original[y_train_original == class1]:
        y_train.append(label1)
    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_test = x_test_original[y_test_original == class1]  # test Data change
    y_test = []
    for x in y_test_original[y_test_original == class1]:
        y_test.append(label1)
    y_test = np.array(y_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test
