import csv
import os
import pickle
import random
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from keras.utils import to_categorical

from data_type.constants import Constants, DEBUG
from update_concern.ewc_trainer import train
from util.common import trainModelAndPredictInBinary
from util.data_util import sample, unarize, combine_for_reuse

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def update_module(module=None, old_train_x=None, old_train_y=None, new_train_x=None, new_train_y=None,
                  val_data=None, use_ewc=True, use_fim=False, use_incdet=False, ewc_lambda=1.0,
                  incdet_thres=1e-6):
    result = train(module, (new_train_x, new_train_y), (old_train_x, old_train_y),
                   use_fim=use_fim, use_ewc=use_ewc, ewc_samples=500, prior_mask=None,
                   fim_samples=500, fim_threshold=1e-3, val_data=val_data,
                   use_incdet=use_incdet, incdet_thres=incdet_thres, ewc_lambda=ewc_lambda)
    return result


def evaluate_ewc(modules, data, num_sample=100, num_module=0):
    _, _, xt, yt, labels, num_classes = combine_for_reuse(modules, data, num_sample_test=num_sample)

    start = time.time()

    preds = {}
    for _d in modules:
        preds[_d] = {}
        for _c in modules[_d]:
            preds[_d][_c] = {}
            for _m in modules[_d][_c]:
                preds[_d][_c][_m] = modules[_d][_c][_m].predict(xt, verbose=0)

    predLabels = []
    for i in range(0, len(yt)):
        ks = None
        maxp = None
        for _d in modules:
            for _c in modules[_d]:
                for _m in modules[_d][_c]:
                    pd = preds[_d][_c][_m][i][_c]
                    if maxp is None or maxp < pd:
                        maxp = pd
                        ks = (_d, _c)

        predLabels.append(labels[ks[0]][ks[1]])

    predLabels = np.asarray(predLabels)
    predLabels = predLabels.flatten()
    modScore = accuracy_score(predLabels, np.asarray(yt).flatten())
    end = time.time()

    inferTime = (end - start) / len(yt)
    inferTime /= num_module

    print("Modularized Accuracy: " + str(modScore))

    return modScore, inferTime


def evaluate_scratch(modules, data,
                     num_sample_train=-1, num_sample_test=-1):
    scratch_model_path = os.path.join(base_path, 'h5', 'model_scratch1' + '.h5')

    xT, yT, xt, yt, labels, num_classes = combine_for_reuse(modules, data, num_sample_train=num_sample_train,
                                                            num_sample_test=num_sample_test)
    yT = to_categorical(yT)
    monScore, train_time, infer_time = trainModelAndPredictInBinary(scratch_model_path,
                                                                    xT, yT, xt, yt,
                                                                    nb_classes=num_classes)

    print("Trained Accuracy: " + str(monScore))
    print("Trained time: " + str(train_time))

    return monScore, train_time, infer_time
