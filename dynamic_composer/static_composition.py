#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 21:38:03 2020

@author:
"""
from imblearn.under_sampling import RandomUnderSampler
from keras.models import load_model
import itertools
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
import os
import numpy as np
from sklearn.utils import shuffle

from data_type.constants import Constants
from evaluation.accuracy_computer import getModulePredictionAnyToManyUnrolled, getMonolithicModelPredictionAnyToOne
from util.common import initModularLayers, repopulateModularWeights, binarize_multi_label, \
    trainModelAndPredictInBinaryForManyOutput, trainModelAndPredictInBinary, get_max_without
from util.data_util import get_mnist_data, get_fmnist_data, unarize

Constants.disableUnrollMode()

model1 = 'model_mnist'
model2 = 'model_fmnist'

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
out = open(os.path.join(base_path, "result", "static_"+model1+"_"+model2 + ".csv"), "w")
# base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))
module_path1 = os.path.join(base_path, 'modules', model1)
module_path2 = os.path.join(base_path, 'modules', model2)

scratch_model_path = os.path.join(base_path, 'h5', 'model_scratch.h5')
model_path1 = os.path.join(base_path, 'h5', model1 + '.h5')
model_path2 = os.path.join(base_path, 'h5', model2 + '.h5')

model1 = load_model(model_path1)
model2 = load_model(model_path2)

print(base_path)

x_train1, y_train1, x_test1, y_test1, nb_classes1 = get_mnist_data(hot=False)
x_train2, y_train2, x_test2, y_test2, nb_classes2 = get_fmnist_data(hot=False)

class1 = 0
class2 = 0
diff = 0.0
no_pairs = 0

modules1 = []
for m in range(nb_classes1):
    modules1.append(load_model(os.path.join(module_path1,
                                            'module' + str(m) + '.h5')))

modules2 = []
for m in range(nb_classes2):
    modules2.append(load_model(os.path.join(module_path2,
                                            'module' + str(m) + '.h5')))

out.write('Class 1,Class 2,Modularized Accuracy,Trained Model Accuracy,Convergance (Dynamic),Convergance (Scratch)\n')

nb_classes = min(nb_classes1, nb_classes2)

for class1 in range(nb_classes1):

    for class2 in range(nb_classes2):

        moduleClass1 = modules1[class1]
        moduleClass2 = modules2[class2]

        xT1, yT1, xt1, yt1 = unarize(x_train1, y_train1, x_test1, y_test1, class1, 0)
        xT2, yT2, xt2, yt2 = unarize(x_train2, y_train2, x_test2, y_test2, class2, 1)

        xT = np.concatenate((xT1, xT2))
        yT = np.concatenate((yT1, yT2))
        xt = np.concatenate((xt1, xt2))
        yt = np.concatenate((yt1, yt2))
        xT, yT = shuffle(xT, yT, random_state=0)
        xt, yt = shuffle(xt, yt, random_state=0)

        # enn = RandomUnderSampler(random_state=0)
        #
        # xT, yT = enn.fit_resample(xT, yT)

        # predClass1 = getMonolithicModelPredictionAnyToOne(moduleClass1, xt, yt)
        predClass1 = moduleClass1.predict(xt)

        predClass2 = moduleClass2.predict(xt)

        # predClass2 = getMonolithicModelPredictionAnyToOne(moduleClass2, xt, yt)

        finalPred = []
        for i in range(0, len(yt)):

            if predClass1[i][class1] >= predClass2[i][class2]:
                finalPred.append(0)
            else:
                finalPred.append(1)

        finalPred = np.asarray(finalPred)
        finalPred = finalPred.flatten()
        scoreModular = accuracy_score(finalPred, np.asarray(yt).flatten())
        print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(scoreModular))

        yT = to_categorical(yT)
        modelAccuracy, _ = trainModelAndPredictInBinary(scratch_model_path,
                                                     xT, yT, xt, yt)
        print("Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

        diff += (modelAccuracy - scoreModular)
        no_pairs += 1

        out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + '\n')

out.close()

diff = diff / no_pairs
print('Average loss of accuracy: ' + str(diff))
