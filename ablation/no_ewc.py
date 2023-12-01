import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

from update_concern.ewc_trainer import evaluate
from update_concern.ewc_util import get_combos, update2, evaluate_ewc, load_combos, update_module
from util.common import trainModelAndPredictInBinary
from util.data_util import load_data_by_name, \
    sample_and_combine_train_positive, sample_and_combine_test_positive, sample, unarize, \
    sample_train_ewc, sample_test_ewc

positiveRatioInValid = 1.0  # Try with: 0.0, 0.5, 1.0, 2.0, 4.0
trainModuleFromScratch = False
doUntil = 100
positiveRatioInTrain = 1.0
includePositive = True
use_ewc = False
ewc_lambda = 0.1
use_incdet = False
incdet_thres = 1e-6
num_sample = 500
logOutput = True

is_load_combo = True
mode = 'static'  # static or update or nll
total_combination = 100
total_repeat = 1
model_suffix = '1'
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']


base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

result = {}
scratchDict = {}
scratch_time = {}
# modular_time = {}
# modular_dict = {}

if not is_load_combo:
    combos = get_combos(data, datasets, total_combination, _seed=11)
else:
    # name = 'static_repeat_0'
    name='update_model1'
    combos, scratchDict, scratch_time, modular_dict, modular_time = load_combos(name=name)

# need to delete following two
scratchDict = {}
scratch_time = {}
modular_time = {}
modular_dict = {}

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

for rpi in range(total_repeat):
    if logOutput:
        out = open(os.path.join(base_path, "result", mode + "_repeat_" + str(rpi) + ".csv"), "w")
        out.write(
            'Combination,Modularized Accuracy,Trained Model Accuracy,Accuracy Delta,Update Time,Scratch Time,Modular Eval Time\n')

        out.close()
    mod_time = {}
    for _cmb in range(len(comboList)):
        if logOutput:
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
                nx, ny = sample_train_ewc(data, (_d, _c), comboList[_cmb],
                                          negativeModule, positiveModule,
                                          num_sample=num_sample,
                                          includePositive=includePositive,
                                          positiveRatio=positiveRatioInTrain)
                val_data = sample_test_ewc(data, (_d, _c), comboList[_cmb],
                                           negativeModule,
                                           positiveModule,
                                           positiveRatio=positiveRatioInValid)

                # _, _, jx, jy = unarize(data[_d][0], data[_d][1], data[_d][2], data[_d][3], _c, _c)
                # jy = to_categorical(jy, data[_d][4])
                # currentAccuracy = evaluate(module, (jx, jy))
                # print('After accuracy (just positive): ' + str(currentAccuracy))
                #
                # val_data = sample_and_combine_test_positive(data, (_d, _c), comboList[_cmb],
                #                                             negativeModule,
                #                                             positiveModule, num_sample=1000, justNegative=True)
                # currentAccuracy = evaluate(module, val_data)
                # print('After accuracy (just negative): ' + str(currentAccuracy))

                if trainModuleFromScratch:
                    scratch_model_path = os.path.join(base_path, 'h5', 'model_scratch' + model_suffix + '.h5')

                    _, elpas, module = trainModelAndPredictInBinary(scratch_model_path,
                                                                    nx, ny, val_data[0], val_data[1],
                                                                    nb_classes=data[_d][4])
                    tmp_update_time.append(elpas)
                else:
                    tmp_update_time.append(
                        update_module(module, data[_d][0], data[_d][1], nx, ny, val_data=val_data,
                                      use_ewc=use_ewc,
                                      use_incdet=use_incdet, ewc_lambda=ewc_lambda, incdet_thres=incdet_thres))

            if _d not in modules:
                modules[_d] = {}

            modules[_d][_c] = module

        comboKey, modScore, monScore,modEvalTime = evaluate_ewc(modules, data, scratchDict,
                                                                scratch_time, modular_dict,
                                                                evaluate_mode="positive max",
                                                                model_suffix=model_suffix,
                                                                num_sample=10 * num_sample)

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

        if logOutput:
            out.write(str(comboKey) + ',' +
                      str(avgMod) + ',' + str(scratchDict[comboKey])
                      + ',' + str(avgMod - scratchDict[comboKey])
                      + ',' + str(avgModTime)
                      + ',' + str(scratch_time[comboKey])+','+str(modEvalTime)
                      + '\n')

            out.close()
