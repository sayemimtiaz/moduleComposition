#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
import os

from keras.models import load_model

from evaluation.accuracy_computer import getModulePredictionAnyToOneUnrolled, getMonolithicModelAccuracyAnyToMany, \
    getMonolithicModelAccuracyAnyToOne
from evaluation.jaccard_computer import findMeanJaccardIndexRolled
from util.common import initModularLayers, repopulateModularWeights, extract_model_name
from util.data_util import get_mnist_data, get_fmnist_data, loadTensorFlowDataset


def evaluate_rolled(model_name):
    # model_name = 'model4_combined.h5'
    print('evaluating rolled: ' + model_name)
    model_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    print(model_path)
    model_name = os.path.join(model_path, model_name)

    if 'fmnist' in model_name:
        xT, yT, xt, yt, nb_classes = get_fmnist_data(hot=False)
    elif 'emnist' in model_name:
        xT, yT, xt, yt, nb_classes = loadTensorFlowDataset(datasetName='emnist', hot=False)
    elif 'kmnist' in model_name:
        xT, yT, xt, yt, nb_classes = loadTensorFlowDataset(datasetName='kmnist', hot=False)
    else:
        xT, yT, xt, yt, nb_classes = get_mnist_data(hot=False)

    labs = range(0, nb_classes)
    model = load_model(model_name)

    modules = []
    for m in labs:
        modules.append(load_model(os.path.join(model_path,
                                               'modules',
                                               extract_model_name(model_name),
                                               'module' + str(m) + '.h5')))
        # modularLayers = initModularLayers(model.layers)
        # repopulateModularWeights(modularLayers, os.path.join(model_path, 'modules', extract_model_name(model_name)), m)
        # modules.append(modularLayers)

    finalPred = []
    length = len(yt)
    p = []
    for m in labs:
        p.append(modules[m].predict(xt[:length]))
        # p.append(getModulePredictionAnyToOneUnrolled(modules[m], xt, yt, m))

    for i in range(0, length):
        maxPrediction = []
        for m in labs:
            maxPrediction.append(p[m][i][m])

        finalPred.append(maxPrediction.index(max(maxPrediction)))

    from sklearn.metrics import accuracy_score

    score = accuracy_score(finalPred, yt[:length])
    print("Modularized Accuracy: " + str(score))
    # pred = model.predict(xt[:length])
    # pred = pred.argmax(axis=-1)
    # score = accuracy_score(pred, yt[:length])
    score = getMonolithicModelAccuracyAnyToOne(model_name, xt, yt)
    print("Model Accuracy: " + str(score))

    print('mean jaccard index: ' + str(findMeanJaccardIndexRolled(model_name, model_path, nb_classes)))

# evaluate_rolled('h5/model4_combined.h5')
