import math

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

def unarize(x_train_original, y_train_original, x_test_original, y_test_original, class1, label1):
    # train data change
    x_train = x_train_original[y_train_original != class1]
    y_train = []
    for x in y_train_original[y_train_original != class1]:
        y_train.append(label1)
    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_test = x_test_original[y_test_original != class1]  # test Data change
    y_test = []
    for x in y_test_original[y_test_original != class1]:
        y_test.append(label1)
    y_test = np.array(y_test)
    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test

def makeScalar(data):
    new_data = []
    for i in range(len(data)):
        if type(data[i]) == list or type(data[i]) is np.ndarray:
            if len(data[i]) > 1:
                raise Exception('Data should not be hot encoded')

            new_data.append(data[i][0])
        else:
            new_data.append(data[i])

    return np.asarray(new_data)


def getIndexesMatchingSubset(Y, match):
    indexes = []
    for i in range(len(Y)):
        if Y[i] in match:
            indexes.append(i)
    return indexes


def sample(data, num_sample=-1, num_classes=None, balance=True, sample_only_classes=None, seed=None):
    data_x, data_y = data
    data_y = makeScalar(data_y)
    flag = {}
    all_chosen_index = []

    if seed is not None:
        np.random.seed(seed)

    if sample_only_classes is not None:
        num_classes = len(sample_only_classes)
    if balance and num_sample > 0:
        num_sample = int(math.ceil(num_sample / num_classes))
    for y in data_y:
        if y in flag:
            continue
        if sample_only_classes is not None and y not in sample_only_classes:
            continue
        flag[y] = 1

        class_all_index = getIndexesMatchingSubset(data_y, [y])

        if num_sample == -1 or num_sample > len(class_all_index):
            chosen_index = np.random.choice(class_all_index, len(class_all_index), replace=False)
        else:
            chosen_index = np.random.choice(class_all_index, num_sample, replace=False)
        all_chosen_index.extend(chosen_index)

    np.random.shuffle(all_chosen_index)
    data_x, data_y = data_x[all_chosen_index], data_y[all_chosen_index]

    return data_x, data_y
