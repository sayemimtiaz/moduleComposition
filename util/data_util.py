import math
from collections import Counter

import keras
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
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

    # class_counts = Counter(y_test)
    # for class_label, count in class_counts.items():
    #     print(f'Class {class_label}: {count} samples')

    # Calculate average and standard deviation of the sample counts
    # counts = np.array(list(class_counts.values()))
    # average = np.mean(counts)
    # std_dev = np.std(counts)

    # Determine classes within 1 standard deviation of the average
    # within_one_std = [class_label for class_label, count in class_counts.items() if abs(count - average) <= std_dev]

    # print(f'\nAverage number of samples: {average}')
    # print(f'Standard deviation: {std_dev}')
    # print(f'Number of classes within 1 standard deviation of the average: {len(within_one_std)}')
    # # Calculate quartiles
    # quartiles = np.percentile(counts, [25, 50, 75])
    # Q1, Q2, Q3 = quartiles
    # # Output quartiles
    # print(f'\nQuartiles:')
    # print(f'Q1 (25th percentile): {Q1}')
    # print(f'Q2 (50th percentile / median): {Q2}')
    # print(f'Q3 (75th percentile): {Q3}')

    # within_500_to_1500 = [class_label for class_label, count in class_counts.items() if 500 <= count <= 1500]
    # print(f'\nNumber of classes with counts between 500 and 1500: {len(within_500_to_1500)}')

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


def make_reuse_dataset(x_train_original, y_train_original, x_test_original, y_test_original, classes, labels,
                       num_sample_train=-1, num_sample_test=-1, is_train_rate=False):
    x_train = []
    y_train = []

    # train data change
    if num_sample_train == -1:
        x_train = x_train_original[np.where(np.isin(y_train_original, classes))]
        y_train = y_train_original[np.where(np.isin(y_train_original, classes))]
    else:
        flag = False
        for c in classes:
            x_train_temp = x_train_original[np.where(y_train_original == c)]
            y_train_temp = y_train_original[np.where(y_train_original == c)]

            nst = num_sample_train
            if is_train_rate:
                nst = int(len(x_train_temp) * nst)
            chosen_index = np.random.choice(range(len(x_train_temp)), min(len(x_train_temp), nst), replace=False)

            x_train_temp, y_train_temp = x_train_temp[chosen_index], y_train_temp[chosen_index]

            if not flag:
                x_train, y_train = x_train_temp, y_train_temp
                flag = True
            else:
                x_train = np.concatenate((x_train, x_train_temp))
                y_train = np.concatenate((y_train, y_train_temp))
    for y in range(len(y_train)):
        y_train[y] = labels[y_train[y]]
    y_train = np.array(y_train)
    x_train, y_train = shuffle(x_train, y_train, random_state=0)

    if num_sample_test == -1:
        x_test = x_test_original[np.where(np.isin(y_test_original, classes))]
        y_test = y_test_original[np.where(np.isin(y_test_original, classes))]
    else:
        flag = False
        for c in classes:
            x_test_temp = x_test_original[np.where(y_test_original == c)]
            y_test_temp = y_test_original[np.where(y_test_original == c)]

            num_sample = int(len(x_test_temp)*num_sample_test)

            chosen_index = np.random.choice(range(len(x_test_temp)), min(len(x_test_temp), num_sample),
                                            replace=False)

            x_test_temp, y_test_temp = x_test_temp[chosen_index], y_test_temp[chosen_index]

            if not flag:
                x_test, y_test = x_test_temp, y_test_temp
                flag = True
            else:
                x_test = np.concatenate((x_test, x_test_temp))
                y_test = np.concatenate((y_test, y_test_temp))

    for y in range(len(y_test)):
        y_test[y] = labels[y_test[y]]

    y_test = np.array(y_test)

    x_test, y_test = shuffle(x_test, y_test, random_state=0)

    return x_train, y_train, x_test, y_test


def combine_for_reuse(modules, data, num_sample_train=-1, num_sample_test=-1, is_train_rate=False):
    lblCntr = 1
    labels = {}
    flag = False
    for _d in modules:
        labels[_d] = {}
        classes = []
        tmp_labels = {}
        for _c in modules[_d]:
            classes.append(_c)
            tmp_labels[_c] = lblCntr
            labels[_d][_c] = lblCntr
            lblCntr += 1

        if not flag:
            xT, yT, xt, yt = make_reuse_dataset(data[_d][0], data[_d][1], data[_d][2],
                                                data[_d][3], classes, tmp_labels, num_sample_train=num_sample_train,
                                                num_sample_test=num_sample_test, is_train_rate=is_train_rate)
        else:
            xT1, yT1, xt1, yt1 = make_reuse_dataset(data[_d][0], data[_d][1], data[_d][2],
                                                    data[_d][3], classes, tmp_labels, num_sample_train=num_sample_train,
                                                    num_sample_test=num_sample_test, is_train_rate=is_train_rate)
            xT = np.concatenate((xT, xT1))
            yT = np.concatenate((yT, yT1))
            xt = np.concatenate((xt, xt1))
            yt = np.concatenate((yt, yt1))
        flag = True

    xT, yT = shuffle(xT, yT, random_state=0)
    xt, yt = shuffle(xt, yt, random_state=0)

    return xT, yT, xt, yt, labels, lblCntr


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


def sample_train_ewc(data, targetMod, combo, negativeModule, positiveModule,
                     num_sample=500, seed=19,
                     includePositive=True, numMemorySample=1):
    x = {}
    i = 0
    temp_y = []
    # a_y =[]
    negSampleCount = 0
    for (d, c, m) in combo:
        if (d, c, m) == targetMod:
            continue
        x[i], _ = sample((data[d][0], data[d][1]), sample_only_classes=[c],
                         balance=True, num_sample=num_sample, seed=seed)
        negSampleCount += len(x[i])
        for j in range(len(x[i])):
            temp_y.append(negativeModule)
            # a_y.append(0)
        i += 1

    if includePositive:
        # posSampleCount = math.ceil(negSampleCount * positiveRatio)
        x[i], _ = sample((data[targetMod[0]][0], data[targetMod[0]][1]), sample_only_classes=[targetMod[1]],
                         balance=True, num_sample=numMemorySample, seed=seed)

        for i in range(len(x[i])):
            temp_y.append(positiveModule)
            # a_y.append(1)

    mx = x[0]
    for i in range(1, len(x)):
        mx = np.concatenate((mx, x[i]))

    my = to_categorical(temp_y, data[targetMod[0]][4])

    # ay = to_categorical(a_y)

    # mx, my, ay = shuffle(mx, my, ay, random_state=0)
    mx, my = shuffle(mx, my, random_state=0)

    return mx, my


def sample_test_ewc(data, targetMod, combo, negativeModule, positiveModule,
                    num_sample=50, seed=19, positiveRatio=1):
    x = {}
    i = 0
    temp_y = []
    # a_y =[]
    negSampleCount = 0
    for (d, c, m) in combo:
        if (d, c, m) == targetMod:
            continue
        x[i], _ = sample((data[d][2], data[d][3]), sample_only_classes=[c],
                         balance=True, num_sample=num_sample, seed=seed)
        negSampleCount += len(x[i])
        for j in range(len(x[i])):
            temp_y.append(negativeModule)
            # a_y.append(0)
        i += 1

    posSampleCount = math.ceil(negSampleCount * positiveRatio)
    x[i], _ = sample((data[targetMod[0]][2], data[targetMod[0]][3]), sample_only_classes=[targetMod[1]],
                     balance=True, num_sample=posSampleCount, seed=seed)

    for i in range(len(x[i])):
        temp_y.append(positiveModule)
        # a_y.append(1)

    mx = x[0]
    for i in range(1, len(x)):
        mx = np.concatenate((mx, x[i]))

    my = to_categorical(temp_y, data[targetMod[0]][4])

    # ay = to_categorical(a_y)

    # mx, my, ay = shuffle(mx, my, ay, random_state=0)
    mx, my = shuffle(mx, my, random_state=0)

    return mx, my


def sample_and_combine_test_positive(data, targetMod, combo, negativeModule, positiveModule,
                                     num_sample=500, seed=19, justNegative=False):
    x = {}
    i = 0
    temp_y = []
    # a_y = []
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
                # a_y.append(1)
            else:
                temp_y.append(negativeModule)
                # a_y.append(0)

        i += 1

    mx = x[0]
    for i in range(1, len(x.keys())):
        mx = np.concatenate((mx, x[i]))

    my = to_categorical(temp_y, data[targetMod[0]][4])
    # ay = to_categorical(a_y)

    mx, my = shuffle(mx, my, random_state=0)

    return mx, my

# loadTensorFlowDataset('kmnist')
