import os

from keras.models import load_model

from util.sc_util import get_activation_pattern, evaluate_sc
from util.common import load_combos, load_smallest_comobs
from util.data_util import load_data_by_name, \
    sample

num_sample_test = 100
num_sample_train = 10000
logOutput = True
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
start_index = 0
end_index = 199
featureCnn = True

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

comboList = load_combos(start=start_index, end=end_index)
# comboList = load_smallest_comobs(bottom=3)
if logOutput:
    out = open(os.path.join(base_path, "result", "sc_composition" + ".csv"), "w")
    out.write(
        'Combination ID,Modularized Accuracy,Setup Time,Inference Time\n')
    out.close()

cPattern = {}
moduleCount = {}
cmbId = start_index
for _cmb in range(len(comboList)):
    if logOutput:
        out = open(os.path.join(base_path, "result", "sc_composition" + ".csv"), "a")

    modules = {}
    moduleCount[_cmb] = 0
    setupTime = 0
    for (_d, _c, _m) in comboList[_cmb]:
        print(_d, _c, _m)
        moduleCount[_cmb] += 1
        if _d not in modules:
            modules[_d] = {}
            cPattern[_d] = {}

        if _c not in modules[_d]:
            modules[_d][_c] = {}
            cPattern[_d][_c] = {}

        module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                         'module' + str(_c) + '.h5'))

        if _m not in cPattern[_d][_c]:
            x, y = sample((data[_d][0], data[_d][1]), sample_only_classes=[_c],
                          balance=True, num_sample=num_sample_train, seed=29)
            mean, std, curSetupTime = get_activation_pattern(module, x)
            cPattern[_d][_c][_m] = (mean, std)
            setupTime += curSetupTime

        modules[_d][_c][_m] = module

    setupTime /= moduleCount[_cmb]
    score, eval_time = evaluate_sc(modules, data, cPattern, num_sample=num_sample_test, num_module=moduleCount[_cmb])
    if logOutput:
        out.write(str(cmbId)
                  + ',' + str(score) + ',' + str(setupTime) + ',' + str(eval_time)
                  + '\n')

        out.close()
    cmbId += 1
