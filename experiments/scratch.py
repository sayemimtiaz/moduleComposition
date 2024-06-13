import os
from keras.models import load_model
import numpy as np
from update_concern.ewc_util import evaluate_scratch
from util.common import load_combos, load_smallest_comobs
from util.data_util import load_data_by_name

num_sample_test = -1
num_sample_train = 0.5
logOutput = False
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
start_index = 0
end_index = 199
is_train_rate = True

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
frequency_dict={}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

    unique_values, counts = np.unique(data[_d][1], return_counts=True)

    # frequency_dict[_d] = dict(zip(unique_values, counts))
    # print(_d, min(counts), max(counts))

# comboList = load_combos(start=start_index, end=end_index)
comboList = load_smallest_comobs(bottom=5)

# ratio={}
# for _cmb in range(len(comboList)):
#     mnc=None
#     mxc=None
#     for (_d, _c, _m) in comboList[_cmb]:
#         nc=frequency_dict[_d][_c]
#         if mnc is None:
#             mnc=nc
#             mxc=nc
#         else:
#             if mnc>nc:
#                 mnc=nc
#             if mxc < nc:
#                 mxc = nc
#     ratio[_cmb]=mnc/mxc
# sorted_ratio = dict(sorted(ratio.items(), key=lambda item: item[1]))
#
# newComboList=[]
# itr=0
# for key, value in sorted_ratio.items():
#     if itr>20:
#         break
#     newComboList.append(comboList[key])
#     itr+=1
#
# comboList=newComboList

if logOutput:
    out = open(os.path.join(base_path, "result", "scratch_time.csv"), "w")
    out.write(
        'Combination ID,Model Accuracy,Training Time,Eval Time\n')

    out.close()

cmbId = start_index
precisions=[]
recalls=[]
f1s=[]
aucs=[]
accuracis=[]
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "scratch_time.csv"), "a")

    unique_modules = {}
    tmp_update_time = []

    for (_d, _c, _m) in comboList[_cmb]:
        # print(_d, _c, _m)
        module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                         'module' + str(_c) + '.h5'))

        if _d not in unique_modules:
            unique_modules[_d] = {}
        unique_modules[_d][_c] = 1

    score, trainTime, inferTime, precision, recall, f1, auc = evaluate_scratch(unique_modules, data, num_sample_train=num_sample_train,
                                                   num_sample_test=num_sample_test, is_train_rate=is_train_rate)

    accuracis.append(score)
    precisions.append(precision)
    recalls.append(recall)
    f1s.append(f1)
    aucs.append(auc)

    if logOutput:
        out.write(str(cmbId) + ',' +
                  str(score) + ',' + str(trainTime)
                  + ',' + str(inferTime)
                  + '\n')

        out.close()
    cmbId += 1

print(f'Accuracy: {np.asarray(accuracis).mean()}')
print(f'Precision: {np.asarray(precisions).mean()}')
print(f'Recall: {np.asarray(recalls).mean()}')
print(f'F1: {np.asarray(f1s).mean()}')
print(f'AUC: {np.asarray(aucs).mean()}')