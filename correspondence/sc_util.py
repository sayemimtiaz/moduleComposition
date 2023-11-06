from time import time

import tensorflow as tf
from keras.layers import Dropout, Activation, Flatten
import numpy as np
from sklearn.metrics import accuracy_score
from util.data_util import combine_for_reuse


def get_activation_pattern(model, x, c=None):
    obs = {}
    for layerId, layer in enumerate(model.layers):
        if layerId not in obs:
            obs[layerId] = []

    for _x in x:
        _x = tf.expand_dims(_x, axis=0)

        for layerId, layer in enumerate(model.layers):
            if isinstance(layer, Dropout):
                continue
            layer_output = layer(_x)
            if not isinstance(layer, Flatten):

                if layer.activation.__name__.lower() == 'relu' or (
                        layer_output.shape[1] < 100 and layer.activation.__name__.lower() != 'softmax'):
                    obs[layerId].append(layer_output[0].numpy())

            _x = layer_output

    f_means=[]
    f_stds=[]
    for layerId in obs.keys():
        if len(obs[layerId])==0:
            continue
        means = np.mean(obs[layerId], axis=0)
        stds = np.std(obs[layerId], axis=0)
        f_means.extend(means)
        f_stds.extend(stds)

    return np.asarray(f_means), np.asarray(f_stds)


def get_logit(model, x, c):
    for layerId, layer in enumerate(model.layers):
        if isinstance(layer, Dropout) or isinstance(layer, Activation):
            continue
        layer_output = layer(x)
        x = layer_output
    return x[0][c]


def getDistance(mean, std,obs):
    # bina = (a[:,0] > 0).astype(int)
    # binb = (b[:, 0] > 0).astype(int)
    # bina=bina[:100]
    # binb = binb[:100]
    valid_indices = std != 0.0
    z_score = np.sum(np.abs((obs[valid_indices] - mean[valid_indices]) / std[valid_indices]))
    return z_score

def evaluate_sc(modules, data, cPattern,
                num_sample=100):
    _, _, xt, yt, combo_str, labels, num_classes = combine_for_reuse(modules, data, num_sample_train=num_sample)
    n_eval=1000
    xt = xt[:n_eval]
    yt = yt[:n_eval]
    print('Evaluating ' + combo_str)

    predLabels = []
    start = time()

    for i in range(0, len(yt)):
        _x = tf.expand_dims(xt[i], axis=0)

        tPattern = {}
        for _d in modules:
            tPattern[_d] = {}
            for _c in modules[_d]:
                tPattern[_d][_c] = get_activation_pattern(modules[_d][_c], _x, _c)

        minDis = None
        ks = None
        for _d in modules:
            for _c in modules[_d]:
                dis = getDistance(cPattern[_d][_c][0], cPattern[_d][_c][1], tPattern[_d][_c][0])
                if minDis is None or dis < minDis:
                    minDis = dis
                    ks = _d, _c

        predLabels.append(labels[ks[0]][ks[1]])

    predLabels = np.asarray(predLabels)
    predLabels = predLabels.flatten()
    modScore = accuracy_score(predLabels, np.asarray(yt).flatten())
    end = time()

    print("Modularized Accuracy: " + str(modScore))

    return combo_str, modScore, (end-start)/n_eval


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
