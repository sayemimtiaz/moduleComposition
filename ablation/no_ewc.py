import os
import numpy as np
from keras.models import load_model
from update_concern.ewc_util import get_combos, update2, evaluate_composition2, load_combos, update_for_ablation
from util.data_util import load_data_by_name, \
    sample_and_combine_train_positive, sample_and_combine_test_positive, sample, unarize, \
    sample_and_combine_train_positive_for_ablation

includePositive = False
positiveRatio = 1
use_ewc = True

is_load_combo = True
mode = 'update'  # static or update
total_combination = 100
total_repeat = 1
model_suffix = ''
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

result = {}
scratchDict = {}
scratch_time = {}
modular_time = {}
modular_dict = {}

if not is_load_combo:
    combos = get_combos(data, datasets, total_combination, _seed=11)
else:
    combos, scratchDict, scratch_time, modular_dict, modular_time = load_combos()

# need to delete following two
# scratchDict = {}
# scratch_time = {}
modular_time = {}
modular_dict = {}

comboList = []
for _cmb in combos.keys():
    tList = []
    for _d in combos[_cmb].keys():
        for _c in combos[_cmb][_d]:
            tList.append((_d, _c))
    comboList.append(tList)

for rpi in range(total_repeat):
    out = open(os.path.join(base_path, "result", mode + "_repeat_" + str(rpi) + ".csv"), "w")
    out.write(
        'Combination,Modularized Accuracy,Trained Model Accuracy,Accuracy Delta,Update Time,Scratch Time\n')

    out.close()
    mod_time = {}
    for _cmb in range(len(comboList)):
        out = open(os.path.join(base_path, "result", mode + "_repeat_" + str(rpi) + ".csv"), "a")

        modules = {}
        tmp_update_time = []

        for (_d, _c) in comboList[_cmb]:
            print(_d, _c)
            module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + model_suffix,
                                             'module' + str(_c) + '.h5'))
            positiveModule = _c
            negativeModule = 0
            if _c == 0:
                negativeModule = 1

            if mode == 'update' and len(modular_dict) == 0:
                nx, ny = sample_and_combine_train_positive_for_ablation(data, (_d, _c), comboList[_cmb],
                                                                        negativeModule, positiveModule, num_sample=100,
                                                                        includePositive=includePositive,
                                                                        positiveRatio=positiveRatio)
                val_data = sample_and_combine_test_positive(data, (_d, _c), comboList[_cmb],
                                                            negativeModule,
                                                            positiveModule, num_sample=1000)

                tmp_update_time.append(
                    update_for_ablation(module, data[_d][0], data[_d][1], nx, ny, val_data=val_data, use_ewc=use_ewc))

            if _d not in modules:
                modules[_d] = {}

            modules[_d][_c] = module

        comboKey, modScore, monScore = evaluate_composition2(modules, data, scratchDict,
                                                             scratch_time, modular_dict,
                                                             mode="positive max",
                                                             model_suffix=model_suffix)

        if mode == 'update':
            if len(modular_dict) == 0:
                avgModTime = np.asarray(tmp_update_time).mean()
            else:
                avgModTime = modular_time[comboKey]
        else:
            avgModTime = 'N/A'

        if comboKey not in result:
            result[comboKey] = 0
        result[comboKey] += modScore

        avgMod = result[comboKey] / (rpi + 1)

        out.write(str(comboKey) + ',' +
                  str(avgMod) + ',' + str(scratchDict[comboKey])
                  + ',' + str(avgMod - scratchDict[comboKey])
                  + ',' + str(avgModTime)
                  + ',' + str(scratch_time[comboKey])
                  + '\n')

        out.close()
