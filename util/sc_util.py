import time

import tensorflow as tf
from keras.layers import Dropout, Activation, Flatten, AveragePooling2D, Conv2D, Dense
import numpy as np
from sklearn.metrics import accuracy_score
from util.data_util import combine_for_reuse


def validate_layer(layer):
    if isinstance(layer, AveragePooling2D):
        return True
    if isinstance(layer, Dense):
        return True
    if isinstance(layer, Activation):
        return True
    return False


def get_activation_pattern(model, x):
    start = time.time()
    obs = {}
    for layerId, layer in enumerate(model.layers):
        if layerId not in obs:
            obs[layerId] = []

    for _x in x:
        _x = tf.expand_dims(_x, axis=0)
        _x = tf.expand_dims(_x, axis=-1)

        for layerId, layer in enumerate(model.layers):
            if isinstance(layer, Dropout):
                continue
            layer_output = layer(_x)

            if validate_layer(layer):
                obs[layerId].append(layer_output[0].numpy().flatten())

            _x = layer_output

    f_means = []
    f_stds = []
    for layerId in obs.keys():
        if len(obs[layerId]) == 0:
            continue
        means = np.mean(obs[layerId], axis=0)
        stds = np.std(obs[layerId], axis=0)
        f_means.extend(means)
        f_stds.extend(stds)

    end = time.time()
    return np.asarray(f_means), np.asarray(f_stds), end - start


def infer_activation_pattern(model, _x):
    obs = []

    _x = tf.expand_dims(_x, axis=0)
    _x = tf.expand_dims(_x, axis=-1)

    for layerId, layer in enumerate(model.layers):
        if isinstance(layer, Dropout):
            continue
        layer_output = layer(_x)

        if validate_layer(layer):
            obs.extend(layer_output[0].numpy().flatten())

        _x = layer_output

    return _x, np.asarray(obs)


def get_logit(model, x, c):
    for layerId, layer in enumerate(model.layers):
        if isinstance(layer, Dropout) or isinstance(layer, Activation):
            continue
        layer_output = layer(x)
        x = layer_output
    return x[0][c]


def getDistance(mean, std, obs):
    valid_indices = std != 0.0
    z_score = np.sum(np.abs((obs[valid_indices] - mean[valid_indices]) / std[valid_indices]))
    return z_score


def evaluate_sc(modules, data, cPattern,
                num_sample=100, num_module=0):
    _, _, xt, yt, labels, num_classes = combine_for_reuse(modules, data, num_sample_test=num_sample)

    predLabels = []
    start = time.time()

    for i in range(0, len(yt)):

        tPattern = {}
        normalPreds = {}
        for _d in modules:
            tPattern[_d] = {}
            normalPreds[_d] = {}
            for _c in modules[_d]:
                tPattern[_d][_c] = {}
                normalPreds[_d][_c] = {}
                for _m in modules[_d][_c]:
                    pred, mean = infer_activation_pattern(modules[_d][_c][_m], xt[i])
                    tPattern[_d][_c][_m] = mean
                    normalPreds[_d][_c][_m] = pred

        minDis = None
        kds = None
        for _d in modules:
            for _c in modules[_d]:
                for _m in modules[_d][_c]:
                    dis = getDistance(cPattern[_d][_c][_m][0], cPattern[_d][_c][_m][1], tPattern[_d][_c][_m])
                    if minDis is None or dis < minDis:
                        minDis = dis
                        kds = _d

        tmpPreds = []
        tmpCs = []
        for _c in modules[kds]:
            for _m in modules[kds][_c]:
                tmpPreds.append(normalPreds[kds][_c][_m][0][_c])
                tmpCs.append(_c)

        kdc = tmpPreds.index(max(tmpPreds))
        predLabels.append(labels[kds][tmpCs[kdc]])

    predLabels = np.asarray(predLabels)
    predLabels = predLabels.flatten()
    modScore = accuracy_score(predLabels, np.asarray(yt).flatten())
    end = time.time()

    inferTime = (end - start) / len(yt)
    inferTime /= num_module
    print("Modularized Accuracy: " + str(modScore))

    return modScore, inferTime


def evaluate_logit(modules, data,
                   num_sample=100):
    _, _, xt, yt, combo_str, labels, num_classes = combine_for_reuse(modules, data, num_sample_train=num_sample)
    print('Evaluating ' + combo_str)
    xt = xt[:100]
    yt = yt[:100]
    predLabels = []
    print(len(yt))
    for i in range(0, len(yt)):
        _x = tf.expand_dims(xt[i], axis=0)

        maxLogit = None
        ks = None
        for _d in modules:
            for _c in modules[_d]:
                logit = get_logit(modules[_d][_c], _x, _c)

                if maxLogit is None or logit > maxLogit:
                    maxLogit = logit
                    ks = _d, _c

        predLabels.append(labels[ks[0]][ks[1]])

    predLabels = np.asarray(predLabels)
    predLabels = predLabels.flatten()
    modScore = accuracy_score(predLabels, np.asarray(yt).flatten())

    print("Modularized Accuracy: " + str(modScore))

    return combo_str, modScore
