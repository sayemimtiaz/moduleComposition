import csv
import math
import os
import random
import time

import numpy as np
import scipy
from keras import Input, Model, Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, TimeDistributed, RepeatVector, LSTM, GRU, Dropout, Concatenate, Flatten, Average, \
    ZeroPadding1D, Reshape, Lambda, Conv2D, AveragePooling2D
from keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_type.constants import Constants, ALL_GATE
from data_type.enums import ActivationType, LayerType, getLayerType, getActivationType
from data_type.modular_layer_type import ModularLayer
import tensorflow as tf

from util.data_util import load_data_by_name


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference
    # return scipy.special.softmax(x)


def sigmoid(x):
    return scipy.special.expit(x)


def tanh(x):
    return np.tanh(x)


def relu(x_t):
    x_t[x_t < 0] = 0
    return x_t


def initModularLayers(layers, timestep=None):
    myLayers = []
    first = True
    lastLayerTimestep = None
    for serial, _layer in enumerate(layers):
        l = ModularLayer(_layer, timestep=timestep)

        if l.type == LayerType.TimeDistributed:
            if lastLayerTimestep is None:
                l.initTimeDistributedWeights(timestep)
            else:
                l.initTimeDistributedWeights(lastLayerTimestep)
            l.setHiddenState()

        l.first_layer = first
        l.layer_serial = serial

        if l.timestep is not None:
            lastLayerTimestep = l.timestep

        if not first:
            myLayers[len(myLayers) - 1].next_layer = l

        if len(layers) == serial + 1:
            if l.type == LayerType.Activation:
                myLayers[-1].last_layer = True
                myLayers[-1].DW = myLayers[-1].W
                myLayers[-1].DB = myLayers[-1].B
            else:
                l.last_layer = True
                l.DW = l.W
                l.DB = l.B

        myLayers.append(l)
        first = False

    return myLayers


def isNodeActive(layer, nodeNum, threshold=None, timestep=None):
    hs = 0
    if timestep is not None:
        hs = layer.hidden_state[timestep][0, nodeNum]
    else:
        hs = layer.hidden_state[0, nodeNum]

    if layer.activation == ActivationType.Relu:
        return not (hs <= 0)
    if layer.activation == ActivationType.Sigmoid:
        return not (hs < 10)
    if layer.activation == ActivationType.Tanh:
        return not (-0.1 <= hs <= 0.1)


def getDeadNodePercent(layer, timestep=None, isPrint=False):
    W = layer.DW
    if timestep is not None:
        W = layer.DW[timestep]

    totalDeadPercdnt = 0.0
    if isPrint:
        print('Dead Node in ' + str(layer.type) + ':')
    if type(W) == list:
        for ts, w in enumerate(W):
            # print('Timestep: '+str(ts))
            alive = 0
            dead = 0
            for r in range(w.shape[0]):
                for c in range(w.shape[1]):
                    if w[r][c] == 0.0:
                        dead += 1
                    else:
                        alive += 1
            p = 0.0
            if alive + dead != 0:
                p = (dead / (alive + dead)) * 100.0
            totalDeadPercdnt += p
        avgDeadPercent = totalDeadPercdnt / (len(W) + 1)
        if isPrint:
            print('Average dead node: ' + str(avgDeadPercent) + '%')

    else:
        alive = 0
        dead = 0
        for r in range(W.shape[0]):
            for c in range(W.shape[1]):
                if W[r][c] == 0.0:
                    dead += 1
                else:
                    alive += 1

        p = 0.0
        if alive + dead != 0:
            p = (dead / (alive + dead)) * 100.0
        if isPrint:
            print('Dead node: ' + str(p) + '%')


def areArraysSame(a, b):
    for i in range(len(a)):
        if a[i].argmax() != b[i].argmax():
            print(a[i].argmax(), b[i].argmax())
            return False
    return True


def shouldRemove(_layer):
    if _layer.last_layer:
        return False
    if _layer.type == LayerType.Embedding \
            or _layer.type == LayerType.RepeatVector \
            or _layer.type == LayerType.Flatten or \
            _layer.type == LayerType.Dropout:
        return False
    return True


def isIntrinsicallyTrainableLayer(_layer):
    if _layer.type == LayerType.Embedding \
            or _layer.type == LayerType.RepeatVector \
            or _layer.type == LayerType.Flatten \
            or _layer.type == LayerType.Dropout \
            or _layer.type == LayerType.Input \
            or _layer.type == LayerType.Activation:
        return False
    return True


def repopulateModularWeights(modularLayers, module_dir, moduleNo, only_decoder=False):
    # print('module_dir>>', module_dir)
    from modularization.concern.concern_identification_encoder_decoder import ConcernIdentificationEnDe
    # module=module_dir
    module = load_model(
        os.path.join(module_dir, 'module' + str(moduleNo) + '.h5'))
    for layerNo, _layer in enumerate(modularLayers):
        if _layer.type == LayerType.RepeatVector \
                or _layer.type == LayerType.Flatten \
                or _layer.type == LayerType.Input \
                or _layer.type == LayerType.Dropout:
            continue
        if _layer.type == LayerType.RNN or _layer.type == LayerType.LSTM or _layer.type == LayerType.GRU:
            if only_decoder and not ConcernIdentificationEnDe.is_decoder_layer(_layer):
                modularLayers[layerNo].DW, modularLayers[layerNo].DU, \
                modularLayers[layerNo].DB = module.layers[layerNo].get_weights()

            elif Constants.UNROLL_RNN:
                for ts in range(_layer.timestep):
                    tempModel = load_model(
                        os.path.join(module_dir, 'module' + str(moduleNo) + '_layer' + str(layerNo) + '_timestep' + str(
                            ts) + '.h5'))
                    modularLayers[layerNo].DW[ts], modularLayers[layerNo].DU[ts], \
                    modularLayers[layerNo].DB[ts] = tempModel.layers[layerNo].get_weights()
            else:
                modularLayers[layerNo].DW, modularLayers[layerNo].DU, \
                modularLayers[layerNo].DB = module.layers[layerNo].get_weights()

        elif _layer.type == LayerType.Embedding:
            modularLayers[layerNo].DW = module.layers[layerNo].get_weights()[0]
        else:
            modularLayers[layerNo].DW, \
            modularLayers[layerNo].DB = module.layers[layerNo].get_weights()


def trainModelAndPredictInBinary(modelPath, X_train, Y_train, X_test, Y_test, epochs=100, batch_size=32, verbose=0
                                 , nb_classes=2, activation='softmax'):
    model = load_model(modelPath)
    model.pop()
    model.add(Dense(units=nb_classes, activation=activation, name='output'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True)
    # es = EarlyStopping(monitor='loss', mode='min', patience=3)

    start = time.time()
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              verbose=verbose,
              callbacks=[es]
              )
    end = time.time()
    train_time = end - start
    start = time.time()
    pred = model.predict(X_test[:len(Y_test)], verbose=verbose)
    pred = pred.argmax(axis=-1)
    if len(Y_test.shape) > 1:
        score = accuracy_score(pred, Y_test.argmax(-1))
    else:
        score = accuracy_score(pred, Y_test)
    end = time.time()
    eval_time = (end - start) / len(X_test)

    return score, train_time, eval_time


def getBottleneckFeatures(model, data):
    new_model = Model(inputs=model.input, outputs=model.layers[-4].output)

    output = new_model.predict(data)

    return output


def getBottleneckModule(module):
    new_model = Model(inputs=module.input, outputs=module.layers[-3].output)

    return new_model


def getStackedLeNet(modules, featureCnn=False):
    def add_pad_layer(data, desired_input_len=100):
        return tf.pad(data, [[0, 0], [0, desired_input_len - tf.shape(data)[1]]])

    desired_output_size = 84
    if featureCnn:
        flatten_size = [256, 352, 448, 544]
        desired_output_size = 256
        for _d in modules:
            for _c in modules[_d]:
                for _m in modules[_d][_c]:
                    desired_output_size = max(desired_output_size, flatten_size[_m - 1])

    inputLayer = Input(shape=(28, 28, 1))

    frozen_modules = []
    for _d in modules:
        for _c in modules[_d]:

            for _m in modules[_d][_c]:

                myLayers = modules[_d][_c][_m].layers

                current = inputLayer
                for layer in myLayers[:-2]:
                    if getLayerType(layer) == LayerType.Conv2D:
                        current = Conv2D(layer.filters, activation='relu',
                                         kernel_size=layer.kernel_size, strides=layer.strides,
                                         weights=layer.get_weights(), trainable=False)(current)
                    elif getLayerType(layer) == LayerType.AveragePooling2D:
                        current = AveragePooling2D(pool_size=layer.pool_size, strides=layer.strides)(current)
                    elif getLayerType(layer) == LayerType.Flatten:
                        current = Flatten()(current)
                    elif getLayerType(layer) == LayerType.Dense:
                        current = Dense(layer.units, activation='relu',
                                        weights=layer.get_weights(), trainable=False)(current)

                current = Lambda(lambda x: add_pad_layer(x, desired_input_len=desired_output_size))(current)

                frozen_modules.append(current)

    current = Average()(frozen_modules)

    model = Model(inputs=inputLayer, outputs=current)

    return model, len(frozen_modules)


def getStackedModel(modules):
    def select_top_k(x, k=100):
        top_values, _ = tf.math.top_k(x, k=k, sorted=True)
        return top_values

    def select_first_n(x, n):
        return x[:, :n]

    def add_pad_layer(data, desired_input_len=100):
        return tf.pad(data, [[0, 0], [0, desired_input_len - tf.shape(data)[1]]])

    inputLayer = Input(shape=(28, 28))

    flat = Flatten()(inputLayer)

    frozen_modules = []
    for _d in modules:
        for _c in modules[_d]:

            myLayers = modules[_d][_c].layers

            current = flat
            for layer in myLayers[:-2]:
                if getLayerType(layer) == LayerType.Dense:
                    current = Dense(layer.units, activation='relu',
                                    weights=layer.get_weights(), trainable=False)(current)

            # current = Lambda(lambda x: select_first_n(x, n))(current)
            # print(current.shape[1])
            # current = add_pad_layer(current, current_input_len=current.shape[1], desired_input_len=100)

            # current = Lambda(select_top_k)(current)
            # current = Reshape((100, 1))(current)
            # current = Lambda(lambda x: tf.reduce_mean(x, axis=1))(current)

            current = Lambda(lambda x: add_pad_layer(x, desired_input_len=100))(current)

            frozen_modules.append(current)

    current = Average()(frozen_modules)
    # current = Concatenate()(frozen_modules)
    # current = Lambda(select_top_k)(current)

    model = Model(inputs=inputLayer, outputs=current)

    return model, len(frozen_modules)


def getStackedPredict(modules, data):
    accumulator = np.zeros((data.shape[0], 100))
    num_module = 0
    mod_times = []
    for _d in modules:
        for _c in modules[_d]:
            start = time.time()
            accumulator += modules[_d][_c].predict(data, verbose=0)
            num_module += 1
            end = time.time()
            mod_times.append(end - start)
    accumulator /= num_module
    return accumulator, np.asarray(mod_times).mean()


def trainDynamicInterface(cMod, numMod, X_train, Y_train, X_test, Y_test, epochs=30, batch_size=32,
                          verbose=0
                          , nb_classes=2):
    # X_train, stack_time = getStackedPredict(modules, X_train)

    start = time.time()
    X_train = cMod.predict(X_train, verbose=0)
    end = time.time()
    stack_time = (end - start) / numMod

    model = Sequential()
    # model.add(Dense(120, activation='relu', input_shape=X_train.shape[1:]))
    # model.add(Dense(84, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax', input_shape=X_train.shape[1:]))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=3, restore_best_weights=True)

    start = time.time()
    model.fit(X_train, Y_train,
              epochs=epochs,
              batch_size=batch_size,
              validation_split=0.1,
              verbose=verbose,
              callbacks=[es]
              )

    end = time.time()

    train_time = stack_time + (end - start)

    # X_test, stack_time = getStackedPredict(modules, X_test)
    start = time.time()
    X_test = cMod.predict(X_test, verbose=0)
    end = time.time()

    stack_time = (end - start) / numMod

    start = time.time()
    pred = model.predict(X_test[:len(Y_test)], verbose=0)
    pred = pred.argmax(axis=-1)
    score = accuracy_score(pred, Y_test[:len(Y_test)])
    end = time.time()
    infer_time = end - start
    infer_time /= len(X_test)
    infer_time += (stack_time / len(X_test))
    return score, train_time, infer_time


def binarize_multi_label(y, class1, class2):
    class1Found = False
    class2Found = False
    for j in range(len(y)):
        if y[j] == class1:
            class1Found = True
        if y[j] == class2:
            class2Found = True

    if class1Found and class2Found:
        return [class1, class2]
    elif class1Found:
        return [0, class1]
    elif class2Found:
        return [0, class2]

    return []


def calculate_50th_percentile_of_nodes_rolled(observed_values, refLayer, normalize=True, overrideReturnSequence=False):
    num_node = refLayer.num_node

    if ALL_GATE:
        if refLayer.type == LayerType.LSTM:
            num_node = num_node * 4
        if refLayer.type == LayerType.GRU:
            num_node = num_node * 3

    for nodeNum in range(num_node):
        tl = []

        if ALL_GATE and ((refLayer.type == LayerType.LSTM and (
                nodeNum < 2 * refLayer.num_node or nodeNum >= 3 * refLayer.num_node)) or \
                         (refLayer.type == LayerType.GRU and nodeNum < 2 * refLayer.num_node)):
            inactive = 0
            active = 0
            for o in observed_values:
                if math.fabs(o[:, nodeNum]) < 0.5:
                    inactive += 1
                else:
                    active += 1
            refLayer.median_node_val[:, nodeNum] = active / (active + inactive)
        else:
            for o in observed_values:
                if not overrideReturnSequence and refLayer.return_sequence:
                    tl.append(math.fabs(o[refLayer.timestep - 1, :, nodeNum]))

                else:
                    tl.append(math.fabs(o[:, nodeNum]))
            tl = np.asarray(tl).flatten()
            # median = np.percentile(tl, 50)
            median = get_mean_minus_outliers(tl)
            refLayer.median_node_val[:, nodeNum] = median
    #
    # df_describe = pd.DataFrame(refLayer.median_node_val.flatten())
    # print(df_describe.describe())

    if normalize:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(refLayer.median_node_val.reshape(-1, 1))

        # df_describe = pd.DataFrame(scaled.flatten())
        # print(df_describe.describe())

        scaled = scaled.reshape(1, -1)
        refLayer.median_node_val = scaled


def get_mean_minus_outliers(data, m=2.):
    data = np.asarray(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    d = data[s < m]
    return np.mean(d)


def calculate_50th_percentile_of_nodes_unrolled(observed_values, refLayer, normalize=True):
    num_node = refLayer.num_node
    if ALL_GATE:
        if refLayer.type == LayerType.LSTM:
            num_node = num_node * 4
        if refLayer.type == LayerType.GRU:
            num_node = num_node * 3
    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            tl = []

            if ALL_GATE and ((refLayer.type == LayerType.LSTM and (
                    nodeNum < 2 * refLayer.num_node or nodeNum >= 3 * refLayer.num_node)) or \
                             (refLayer.type == LayerType.GRU and nodeNum < 2 * refLayer.num_node)):
                inactive = 0
                active = 0
                for o in observed_values:
                    if math.fabs(o[ts][:, nodeNum]) < 0.5:
                        inactive += 1
                    else:
                        active += 1
                refLayer.median_node_val[ts][:, nodeNum] = active / (active + inactive)
            else:
                for o in observed_values:
                    tl.append(math.fabs(o[ts][:, nodeNum]))

                tl = np.asarray(tl).flatten()
                # median = np.percentile(tl, 50)
                # median=np.mean(tl)
                median = get_mean_minus_outliers(tl)
                refLayer.median_node_val[ts][:, nodeNum] = median

    if normalize:
        for ts in range(refLayer.timestep):
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(refLayer.median_node_val[ts].reshape(-1, 1))

            scaled = scaled.reshape(1, -1)
            refLayer.median_node_val[ts] = scaled


def calculate_relative_importance(observed_values, refLayer, noRemoveThreshold=0.5, percentile=50,
                                  overrideReturnSequence=False):
    maxValue = 0.0
    num_node = refLayer.num_node
    if refLayer.type == LayerType.LSTM:
        num_node = num_node * 4
    for nodeNum in range(num_node):
        tl = []
        for o in observed_values:
            if not overrideReturnSequence and refLayer.return_sequence:
                tl.append(math.fabs(o[refLayer.timestep - 1][:, nodeNum]))

            else:
                tl.append(math.fabs(o[:, nodeNum]))
        tl = np.asarray(tl).flatten()
        median = np.percentile(tl, percentile)
        refLayer.median_node_val[:, nodeNum] = median
        maxValue = max(maxValue, median)

    for nodeNum in range(num_node):
        if refLayer.median_node_val[:, nodeNum] > noRemoveThreshold:
            refLayer.median_node_val[:, nodeNum] = 1.0
        else:
            refLayer.median_node_val[:, nodeNum] = refLayer.median_node_val[:, nodeNum] / maxValue


def calculate_relative_importance_unrolled(observed_values, refLayer, noRemoveThreshold=0.5, percentile=50):
    maxValue = {}
    num_node = refLayer.num_node
    if refLayer.type == LayerType.LSTM:
        num_node = num_node * 4
    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            if ts not in maxValue:
                maxValue[ts] = 0.0

            tl = []
            for o in observed_values:
                tl.append(math.fabs(o[ts][:, nodeNum]))

            tl = np.asarray(tl).flatten()
            median = np.percentile(tl, percentile)
            refLayer.median_node_val[ts][:, nodeNum] = median
            maxValue[ts] = max(maxValue[ts], median)

    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            if refLayer.median_node_val[ts][:, nodeNum] > noRemoveThreshold:
                refLayer.median_node_val[ts][:, nodeNum] = 1.0
            else:
                refLayer.median_node_val[ts][:, nodeNum] = refLayer.median_node_val[ts][:, nodeNum] / \
                                                           maxValue[ts]


def get_max_without(a, exceptIdx):
    b = np.append(a[0:exceptIdx], a[exceptIdx + 1:])
    return b.max()


def calculate_active_rate_rolled(observed_values, refLayer):
    num_node = refLayer.num_node

    for nodeNum in range(num_node):
        inactiveCount = 0
        activeCount = 0
        for o in observed_values:
            if o[nodeNum] <= 0.0:
                inactiveCount += 1
            else:
                activeCount += 1

        refLayer.median_node_val[:, nodeNum] = activeCount / (activeCount + inactiveCount)


def calculate_active_rate_unrolled(observed_values, refLayer):
    num_node = refLayer.num_node
    for nodeNum in range(num_node):
        for ts in range(refLayer.timestep):
            inactiveCount = 0
            activeCount = 0
            for o in observed_values:

                if o[ts][:, nodeNum] <= 0.0:
                    inactiveCount += 1
                else:
                    activeCount += 1

            refLayer.median_node_val[ts][:, nodeNum] = activeCount / (activeCount + inactiveCount)


def extract_model_name(model_path):
    if model_path.find('/') != -1:
        model_path = model_path[model_path.rindex('/') + 1:]
    return model_path[:-3]


def find_modules(ds, _c, mc, combos, totalModuleCount, data, num_model=4):
    rs = len(ds)
    for _i, _d in enumerate(ds):
        if mc > totalModuleCount:
            return False
        if _i + 1 == len(ds):
            tc = min(mc - rs + 1, data[_d][4])
        else:
            tc = random.randint(1, min(mc - rs + 1, data[_d][4]))

        # tmp = []
        # for _mo1 in range(data[_d][4]):
        #     for _mo2 in range(1, num_model + 1):
        #         tmp.append((_mo1, _mo2))
        # tmpidx = np.random.choice(len(tmp),
        #                           tc, replace=False)
        # new_tmp = []
        # for _mo1 in tmpidx:
        #     new_tmp.append(tmp[_mo1])

        tmp = np.random.choice(range(data[_d][4]),
                               tc, replace=False)
        new_tmp = []
        for _mo1 in tmp:
            new_tmp.append((_mo1, random.randint(1, num_model)))

        combos[_c][_d] = new_tmp

        mc -= tc
        rs -= 1
        totalModuleCount -= data[_d][4]

    return mc == 0


def get_combos(data, datasets, total_combination=10, _seed=19, debug=True):
    combos = {}

    random.seed(_seed)
    np.random.seed(_seed)

    for _c in range(total_combination):
        combos[_c] = {}
        ds = random.randint(2, len(datasets))
        ds = np.random.choice(datasets, ds, replace=False)
        totalModuleCount = 0
        for _d in ds:
            totalModuleCount += data[_d][4]

        mc = random.randint(len(ds), totalModuleCount)

        while True:
            if find_modules(ds, _c, mc, combos, totalModuleCount, data):
                break
            if debug:
                print('Retrying combo matching: ' + str(_c))
    # print(combos)
    return combos


def load_combos(start=0, end=199):
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    combos = []

    fileName = os.path.join(base_path, "result", "combinations.csv")
    with open(fileName, 'r') as file:
        _i = 0

        for row in file:
            if _i < start:
                _i += 1
                continue
            if _i > end:
                break
            # combos[_i] = {}
            combo = []
            tcmb = row.strip()

            tcmb = tcmb.replace('(', '')
            tcmb = tcmb.split(')')[:-1]

            for _c in tcmb:
                tc = _c.split(':')
                _d = tc[0].strip()
                tc = tc[1].split('-')

                rt = []
                for r in tc:
                    r = r.replace('[', '')
                    r = r.replace(']', '')
                    r = r.split(',')
                    # rt.append((int(r[0]), int(r[1])))
                    combo.append((_d, int(r[0]), int(r[1])))
                # rt = sorted(rt, key=lambda x: x[0])
                # combos[_i][_d] = rt
            _i += 1

            combos.append(combo)

    return combos


def create_combos():
    datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']

    data = {}
    for _d in datasets:
        data[_d] = load_data_by_name(_d, hot=False)

    combos = get_combos(data, datasets, 200, _seed=11)

    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    out = open(os.path.join(base_path, "result", "combinations.csv"), "w")

    for _cmb in combos.keys():
        s = ''
        for _d in combos[_cmb].keys():
            s += '(' + _d + ':'
            for _c, _m in combos[_cmb][_d]:
                s += '[' + str(_c) + ',' + str(_m) + ']'
                if (_c, _m) != combos[_cmb][_d][-1]:
                    s += '-'
            s += ')'
        out.write(s + '\n')

    out.close()

# create_combos()
# load_combos()
