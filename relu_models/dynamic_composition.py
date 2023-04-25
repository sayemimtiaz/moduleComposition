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
    trainModelAndPredictInBinaryForManyOutput, trainModelAndPredictInBinary, get_max_without, \
    compose_dynamically_and_train
from util.data_util import get_mnist_data, get_fmnist_data, unarize

Constants.disableUnrollMode()

model1 = 'model_mnist'
model2 = 'model_fmnist'

base_path = os.path.dirname(os.path.realpath(__file__))
out = open(os.path.join(base_path, "result", "dynamic_" + model1 + "_" + model2 + ".csv"), "w")
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
diffElapse = 0.0
elpaseModular = 0.0
elpaseScratch = 0.0
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

        # if len(xT)<10000:
        #     continue

        # enn = RandomUnderSampler(random_state=0)
        #
        # xT, yT = enn.fit_resample(xT, yT)
        yT = to_categorical(yT)
        scoreModular, modularTime = compose_dynamically_and_train(moduleClass1, moduleClass2, xT, yT, xt, yt)

        print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(scoreModular))

        modelAccuracy, scratchTime = trainModelAndPredictInBinary(scratch_model_path,
                                                                  xT, yT, xt, yt)
        print("Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

        diff += (modelAccuracy - scoreModular)
        diffElapse += (scratchTime - modularTime)

        elpaseModular += modularTime
        elpaseScratch += scratchTime

        no_pairs += 1

        print('Elapsed (modular vs. scratch): ', modularTime, scratchTime)

        out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + ',' +
                  str(modularTime) + ',' + str(scratchTime) + '\n')

out.close()

diff = diff / no_pairs
diffElapse = diffElapse / no_pairs
elpaseModular = elpaseModular / no_pairs
elpaseScratch = elpaseScratch / no_pairs

elpaseRate = ((elpaseModular - elpaseScratch) / elpaseScratch) * 100

print('Average loss of accuracy: ' + str(diff))
print('Average convergance accelaration: ' + str(elpaseRate))
print('Average modular time: ' + str(elpaseModular))
print('Average scratch time: ' + str(elpaseScratch))
