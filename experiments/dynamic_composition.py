import os

from keras.models import load_model
from keras.utils import to_categorical

from util.common import load_combos, getStackedLeNet, trainDynamicInterface
from util.data_util import load_data_by_name, combine_for_reuse
import numpy as np

num_sample_test = -1
num_sample_train = -1
logOutput = False
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
start_index = 0
end_index = 199
featureCnn=True

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
frequency_dict={}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

    unique_values, counts = np.unique(data[_d][1], return_counts=True)

    frequency_dict[_d] = dict(zip(unique_values, counts))
    print(_d, min(counts), max(counts))

comboList = load_combos(start=start_index, end=end_index)
# comboList = load_smallest_comobs(bottom=1)

ratio={}
for _cmb in range(len(comboList)):
    mnc=None
    mxc=None
    for (_d, _c, _m) in comboList[_cmb]:
        nc=frequency_dict[_d][_c]
        if mnc is None:
            mnc=nc
            mxc=nc
        else:
            if mnc>nc:
                mnc=nc
            if mxc < nc:
                mxc = nc
    ratio[_cmb]=mnc/mxc
sorted_ratio = dict(sorted(ratio.items(), key=lambda item: item[1]))

newComboList=[]
itr=0
for key, value in sorted_ratio.items():
    if itr>20:
        break
    newComboList.append(comboList[key])
    itr+=1

comboList=newComboList

if logOutput:
    out = open(os.path.join(base_path, "result", "dynamic_composition" + ".csv"), "w")
    out.write(
        'Combination ID,Modularized Accuracy,Training Time,Inference Time\n')

    out.close()

cPattern = {}
precisions=[]
recalls=[]
f1s=[]
aucs=[]
accuracis=[]
cmbId = start_index
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "dynamic_composition" + ".csv"), "a")

    modules = {}
    # print('Trying combo: ' + str(_cmb))

    for (_d, _c, _m) in comboList[_cmb]:

        if _d not in modules:
            modules[_d] = {}

        if _c not in modules[_d]:
            modules[_d][_c] = {}

        module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                         'module' + str(_c) + '.h5'))

        modules[_d][_c][_m] = module

    cModel, numMod = getStackedLeNet(modules, featureCnn=featureCnn)

    xT, yT, xt, yt, labels, num_classes = combine_for_reuse(modules, data, num_sample_train=num_sample_train,
                                                            num_sample_test=num_sample_test)
    yT = to_categorical(yT)

    score, train_time, eval_time,precision, recall, f1, auc = trainDynamicInterface(cModel, numMod, xT, yT, xt, yt,nb_classes=num_classes,featureCnn=featureCnn)

    accuracis.append(score)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc)


    if logOutput:
        print(score,train_time,eval_time)
        out.write(str(cmbId)
                  + ',' + str(score) + ',' + str(train_time) + ',' + str(eval_time)
                  + '\n')

        out.close()
    cmbId += 1

print(f'Accuracy: {np.asarray(accuracis).mean()}')
print(f'Precision: {np.asarray(precisions).mean()}')
print(f'Recall: {np.asarray(recalls).mean()}')
print(f'F1: {np.asarray(f1s).mean()}')
print(f'AUC: {np.asarray(aucs).mean()}')