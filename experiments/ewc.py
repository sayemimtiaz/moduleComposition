import os
import time
from keras.models import load_model
from update_concern.ewc_util import update_module, evaluate_ewc
from util.common import load_smallest_comobs, load_combos
from util.data_util import load_data_by_name, sample_train_ewc, sample_test_ewc
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_sample_test = 0.1
num_sample_train = 500
logOutput = True
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
start_index = 0
end_index = 199
numMemorySample = 500
positiveRatioInValid = 1.0

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

comboList = load_combos(start=start_index, end=end_index)
# comboList = load_smallest_comobs(bottom=1)
if logOutput:
    out = open(os.path.join(base_path, "result", "ewc_composition" + ".csv"), "w")
    out.write(
        'Combination ID,Modularized Accuracy,Precision,Recall,F1,AUC,Setup Time,Update Time,Inference Time\n')
    out.close()

cPattern = {}
moduleCount = {}
precisions=[]
recalls=[]
f1s=[]
aucs=[]
accuracis=[]
cmbId = start_index
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "ewc_composition" + ".csv"), "a")

    modules = {}
    moduleCount[_cmb] = 0
    setupTime = 0
    updateTime = 0
    start = time.time()
    for (_d, _c, _m) in comboList[_cmb]:
        print(_d, _c, _m)

        moduleCount[_cmb] += 1
        if _d not in modules:
            modules[_d] = {}
        if _c not in modules[_d]:
            modules[_d][_c] = {}

        # module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
        #                                  'module' + str(_c) + '.h5'))
        module_path=os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                             'module' + str(_c) + '.h5')
        positiveModule = _c
        negativeModule = 0
        if _c == 0:
            negativeModule = 1

        nx, ny = sample_train_ewc(data, (_d, _c, _m), comboList[_cmb],
                                  negativeModule, positiveModule,
                                  num_sample=num_sample_train,
                                  includePositive=True,
                                  numMemorySample=numMemorySample)
        val_data = sample_test_ewc(data, (_d, _c, _m), comboList[_cmb],
                                   negativeModule,
                                   positiveModule,
                                   positiveRatio=positiveRatioInValid)

        curSetupTime, curUpdateTime = update_module(module_path, data[_d][0], data[_d][1], nx, ny, val_data=val_data)
        setupTime += curSetupTime
        updateTime += curUpdateTime

        modules[_d][_c][_m] = load_model(os.path.join(base_path, 'modules', 'updated', 'model_' + _d + str(_m),
                                             'module' + str(_c) + '.h5'), compile=False)

    end = time.time()
    print('Overall time: ', (end - start))
    setupTime /= moduleCount[_cmb]
    updateTime /= moduleCount[_cmb]

    score, eval_time,precision, recall, f1, auc = evaluate_ewc(modules, data,
                                    num_sample=num_sample_test,
                                    num_module=moduleCount[_cmb])

    accuracis.append(score)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc)

    if logOutput:
        out.write(str(cmbId)
                  + ',' + str(score) + ',' + str(precision) +',' + str(recall) +
                  ',' + str(f1) + ',' + str(auc)
                  + str(setupTime) + ',' + str(updateTime) + ',' +
                  str(eval_time)
                  + '\n')

        out.close()
    cmbId += 1

print(f'Accuracy: {np.asarray(accuracis).mean()}')
print(f'Precision: {np.asarray(precisions).mean()}')
print(f'Recall: {np.asarray(recalls).mean()}')
print(f'F1: {np.asarray(f1s).mean()}')
print(f'AUC: {np.asarray(aucs).mean()}')