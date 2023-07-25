import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

from update_concern.ewc_trainer import evaluate
from update_concern.ewc_util import get_combos, update2, evaluate_composition2, load_combos
from util.common import compose_dynamically_and_train
from util.data_util import load_data_by_name, \
    sample_and_combine_train_positive, sample_and_combine_test_positive, sample, unarize, combine_for_reuse

mode = 'dynamic'
total_combination = 100
total_repeat = 1
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
model_suffix = '4'
num_sample = 100

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

result = {}

combos, _, _, _, _ = load_combos(name='update_model1')

comboList = []
for _cmb in combos.keys():
    tList = []
    for _d in combos[_cmb].keys():
        for _c in combos[_cmb][_d]:
            tList.append((_d, _c))
    comboList.append(tList)

for rpi in range(total_repeat):
    out = open(os.path.join(base_path, "result", mode + "_repeat_" + str(rpi) + ".csv"), "w")
    out.write('Combination,Accuracy,Time\n')

    out.close()
    mod_time = {}
    for _cmb in range(len(comboList)):
        out = open(os.path.join(base_path, "result", mode + "_repeat_" + str(rpi) + ".csv"), "a")

        modules = {}

        for (_d, _c) in comboList[_cmb]:
            module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + model_suffix,
                                             'module' + str(_c) + '.h5'))
            if _d not in modules:
                modules[_d] = {}

            modules[_d][_c] = module

        xT, yT, xt, yt, comboKey, labels, num_classes = combine_for_reuse(modules, data, num_sample=100)

        yT = to_categorical(yT)
        modScore, modularTime = compose_dynamically_and_train(modules, xT, yT, xt, yt, nb_classes=num_classes)

        if comboKey not in result:
            result[comboKey] = 0
        result[comboKey] += modScore

        avgMod = result[comboKey] / (rpi + 1)

        out.write(str(comboKey) + ',' + str(avgMod) + ',' + str(modularTime) + '\n')

        out.close()
