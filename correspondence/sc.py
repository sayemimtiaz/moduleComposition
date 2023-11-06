import os
from time import time

import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

from correspondence.sc_util import get_activation_pattern, evaluate_sc, evaluate_logit
from update_concern.ewc_trainer import evaluate
from update_concern.ewc_util import get_combos, update2, evaluate_composition2, load_combos, update_for_ablation
from util.common import trainModelAndPredictInBinary
from util.data_util import load_data_by_name, \
    sample_and_combine_train_positive, sample_and_combine_test_positive, sample, unarize, \
    sample_and_combine_train_positive_for_ablation, sample_and_combine_test_positive_for_ablation

doUntil = 100
num_sample = 10000
logOutput = True

is_load_combo = True
total_combination = 20
model_suffix = '5'
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

result = {}

if not is_load_combo:
    combos = get_combos(data, datasets, total_combination, _seed=11)
else:
    # name = 'static_repeat_0'
    name = 'update_model1'
    combos, scratchDict, scratch_time, modular_dict, modular_time = load_combos(name=name)

comboList = []
it = 0
for _cmb in combos.keys():
    if doUntil != -1 and it >= doUntil:
        break
    tList = []
    for _d in combos[_cmb].keys():
        for _c in combos[_cmb][_d]:
            tList.append((_d, _c))
    comboList.append(tList)
    it += 1

if logOutput:
    out = open(os.path.join(base_path, "result", "sc_model" + str(model_suffix) + ".csv"), "w")
    out.write(
        'Combination,Modularized Accuracy,Inference Time\n')

    out.close()

cPattern = {}
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "sc_model" + str(model_suffix) + ".csv"), "a")

    modules = {}
    tmp_update_time = []

    for (_d, _c) in comboList[_cmb]:
        print(_d, _c)

        if _d not in modules:
            modules[_d] = {}

        if _d not in cPattern:
            cPattern[_d] = {}

        module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + model_suffix,
                                         'module' + str(_c) + '.h5'))

        if _c not in cPattern[_d]:
            x, y = sample((data[_d][0], data[_d][1]), sample_only_classes=[_c],
                          balance=True, num_sample=num_sample, seed=29)
            cPattern[_d][_c] = get_activation_pattern(module, x, _c)

        modules[_d][_c] = module

    comboKey, score, eval_time = evaluate_sc(modules, data, cPattern)
    # evaluate_logit(modules, data)
    if logOutput:
        out.write(str(comboKey)
                  + ',' + str(score)+','+str(eval_time)
                  + '\n')

        out.close()
