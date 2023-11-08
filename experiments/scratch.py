import os
from keras.models import load_model

from update_concern.ewc_util import evaluate_scratch
from util.common import load_combos
from util.data_util import load_data_by_name

num_sample_test = 100
num_sample_train = -1
logOutput = True
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
start_index = 0
end_index = 199

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

comboList = load_combos(start=start_index, end=end_index)

if logOutput:
    out = open(os.path.join(base_path, "result", "scratch.csv"), "w")
    out.write(
        'Combination ID,Model Accuracy,Training Time,Eval Time\n')

    out.close()

cmbId = start_index
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "scratch.csv"), "a")

    unique_modules = {}
    tmp_update_time = []

    for (_d, _c, _m) in comboList[_cmb]:
        print(_d, _c, _m)
        module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                         'module' + str(_c) + '.h5'))

        if _d not in unique_modules:
            unique_modules[_d] = {}
        unique_modules[_d][_c] = 1

    score, trainTime, inferTime = evaluate_scratch(unique_modules, data, num_sample_train=num_sample_train,
                                                   num_sample_test=num_sample_test)

    if logOutput:
        out.write(str(cmbId) + ',' +
                  str(score) + ',' + str(trainTime)
                  + ',' + str(inferTime)
                  + '\n')

        out.close()
    cmbId += 1
