import os

from keras.models import load_model
from keras.utils import to_categorical

from util.common import load_combos, getStackedLeNet, trainDynamicInterface
from util.data_util import load_data_by_name, combine_for_reuse

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
    out = open(os.path.join(base_path, "result", "dynamic_composition" + ".csv"), "w")
    out.write(
        'Combination ID,Modularized Accuracy,Training Time,Inference Time\n')

    out.close()

cPattern = {}
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "dynamic_composition" + ".csv"), "w")

    modules = {}
    tmp_update_time = []
    print('Trying combo: ' + str(_cmb))

    for (_d, _c, _m) in comboList[_cmb]:

        if _d not in modules:
            modules[_d] = {}

        if _c not in modules[_d]:
            modules[_d][_c] = {}

        module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                         'module' + str(_c) + '.h5'))

        modules[_d][_c][_m] = module

    cModel, numMod =getStackedLeNet(modules)

    xT, yT, xt, yt, labels, num_classes = combine_for_reuse(modules, data, num_sample_train=num_sample_train, num_sample_test=num_sample_test)
    yT = to_categorical(yT)

    score, train_time, eval_time = trainDynamicInterface(cModel,numMod, xT, yT, xt, yt, nb_classes=num_classes)

    if logOutput:
        out.write(str(_cmb)
                  + ',' + str(score) + ',' + str(train_time) + ',' + str(eval_time)
                  + '\n')

        out.close()
