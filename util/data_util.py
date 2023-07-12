import math

import keras
from keras.datasets import mnist, fashion_mnist
from keras.utils.np_utils import to_categorical
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow_datasets as tfds


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


def loadTensorFlowDataset(datasetName, hot=True):
    (x_train, y_train), \
    (x_test, y_test) = \
        tfds.as_numpy(tfds.load(datasetName, split=['train', 'test'], batch_size=-1, as_supervised=True))

    # img = Image.fromarray(x_train[5].astype(np.uint8), 'RGB')
    # img.show()

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

    x_train, x_test = asTypeBoth(x_train, x_test)
    x_train, x_test = normalizeBoth(x_train, x_test)

    # img = Image.fromarray(x_train[5].astype(np.uint8), 'RGB')
    # img.show()

    num_classes = max(y_train.max() + 1, y_test.max() + 1)
    if hot:
        y_train, y_test = oneEncodeBoth(y_train, y_test)

    return x_train, y_train, x_test, y_test, num_classes


# default shape is 28*28
def normalize(data, by=255):
    data /= 255
    return data


def normalizeBoth(x_train, x_test, by=255):
    return normalize(x_train, by), normalize(x_test, by)


def oneEncode(data):
    data = to_categorical(data)
    return data


def reshape(data, shape=None):
    if shape is None:
        shape = [28, 28]
    data = tf.image.resize(data, list(shape))
    data = data.numpy()
    return data


def oneEncodeBoth(y_train, y_test):
    return oneEncode(y_train), oneEncode(y_test)


def asType(data, as_type='float32'):
    data = data.astype(as_type)
    return data


def asTypeBoth(x_train, x_test, as_type='float32'):
    return asType(x_train, as_type=as_type), asType(x_test, as_type=as_type)


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


def make_reuse_dataset(x_train_original, y_train_original, x_test_original, y_test_original, classes, labels):
    # train data change
    x_train = x_train_original[np.where(np.isin(y_train_original, classes))]
    y_train = []
    for y in y_train_original[np.where(np.isin(y_train_original, classes))]:
        y_train.append(labels[y])

    y_train = np.array(y_train)

    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    x_test = x_test_original[np.where(np.isin(y_test_original, classes))]
    y_test = []
    for y in y_test_original[np.where(np.isin(y_test_original, classes))]:
        y_test.append(labels[y])

    y_test = np.array(y_test)

    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test


def combine_for_reuse(modules, data):
    lblCntr = 1
    labels = {}
    flag = False
    combo_str = ''
    for _d in modules:
        labels[_d] = {}
        classes = []
        tmp_labels = {}
        combo_str += '(' + str(_d) + ':'
        for _c in modules[_d]:
            combo_str += str(_c) + '-'
            classes.append(_c)
            tmp_labels[_c] = lblCntr
            labels[_d][_c] = lblCntr
            lblCntr += 1
        combo_str += ')'

        if not flag:
            xT, yT, xt, yt = make_reuse_dataset(data[_d][0], data[_d][1], data[_d][2],
                                                data[_d][3], classes, tmp_labels)
        else:
            xT1, yT1, xt1, yt1 = make_reuse_dataset(data[_d][0], data[_d][1], data[_d][2],
                                                    data[_d][3], classes, tmp_labels)
            xT = np.concatenate((xT, xT1))
            yT = np.concatenate((yT, yT1))
            xt = np.concatenate((xt, xt1))
            yt = np.concatenate((yt, yt1))
        flag = True

    xT, yT = shuffle(xT, yT, random_state=0)
    xt, yt = shuffle(xt, yt, random_state=0)

    return xT, yT, xt, yt, combo_str, labels,lblCntr


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


def load_data_by_name(dataset, hot=True):
    if dataset == 'mnist':
        return get_mnist_data(hot=hot)
    if dataset == 'fmnist':
        return get_fmnist_data(hot=hot)
    if dataset == 'kmnist' or dataset == 'emnist':
        return loadTensorFlowDataset(datasetName=dataset, hot=hot)


def sample_and_combine_train_positive(data, targetMod, combo, negativeModule, positiveModule, num_sample=500, seed=19):
    x = {}
    i = 0
    temp_y = []
    negSampleCount = 0
    for (d, c) in combo:
        if (d, c) == targetMod:
            continue
        x[i], _ = sample((data[d][0], data[d][1]), sample_only_classes=[c],
                         balance=True, num_sample=num_sample, seed=seed)
        negSampleCount += len(x[i])
        for j in range(len(x[i])):
            temp_y.append(negativeModule)
        i += 1

    x[i], _ = sample((data[targetMod[0]][0], data[targetMod[0]][1]), sample_only_classes=[targetMod[1]],
                     balance=True, num_sample=negSampleCount, seed=seed)
    for i in range(len(x[i])):
        temp_y.append(positiveModule)

    mx = x[0]
    for i in range(1, len(x)):
        mx = np.concatenate((mx, x[i]))

    my = to_categorical(temp_y, data[targetMod[0]][4])

    mx, my = shuffle(mx, my, random_state=0)

    return mx, my


def sample_and_combine_test_positive(data, targetMod, combo, negativeModule, positiveModule,
                                     num_sample=500, seed=19, justNegative=False):
    x = {}
    i = 0
    temp_y = []
    if not justNegative:
        nl = len(combo) - 1
        nl = int(math.ceil(num_sample / (2 * nl)))
    else:
        nl = num_sample
    for (d, c) in combo:
        if (d, c) == targetMod and justNegative:
            continue
        if (d, c) != targetMod:
            x[i], _ = sample((data[d][2], data[d][3]), sample_only_classes=[c],
                             balance=True, num_sample=nl, seed=seed)
        else:
            x[i], _ = sample((data[d][2], data[d][3]), sample_only_classes=[c],
                             balance=True, num_sample=num_sample, seed=seed)

        for j in range(len(x[i])):
            if (d, c) == targetMod:
                temp_y.append(positiveModule)
            else:
                temp_y.append(negativeModule)

        i += 1

    mx = x[0]
    for i in range(1, len(x.keys())):
        mx = np.concatenate((mx, x[i]))

    my = to_categorical(temp_y, data[targetMod[0]][4])

    mx, my = shuffle(mx, my, random_state=0)

    return mx, my

# loadTensorFlowDataset('kmnist')
