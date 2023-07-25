import os
import numpy as np
from keras.models import load_model
from keras.utils import to_categorical

from update_concern.ewc_trainer import evaluate, compute_ewc_penalty_terms
from update_concern.ewc_util import get_combos, update2, evaluate_composition2, load_combos, update_for_ablation
from util.common import trainModelAndPredictInBinary
from util.data_util import load_data_by_name, \
    sample_and_combine_train_positive, sample_and_combine_test_positive, sample, unarize, \
    sample_and_combine_train_positive_for_ablation

model_suffix = ''
datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

data = {}
for _d in datasets:
    data[_d] = load_data_by_name(_d, hot=False)

out = open(os.path.join(base_path, "result", "model"+model_suffix+"_fisher_calculation_time.csv"), "w")

out.write('Dataset,Mean Time\n')

num_repeat = 5
result = {}
grand = 0
for _d in datasets:
    elaps = 0
    for i in range(num_repeat):
        module = load_model(os.path.join(base_path, 'h5', 'model_' + _d + model_suffix + '.h5'))

        tmp = compute_ewc_penalty_terms(module, (data[_d][0], data[_d][1]))
        grand += tmp
        elaps += tmp

    elaps = elaps / num_repeat

    out.write(_d + ',' + str(elaps) + '\n')

grand = grand / (len(datasets) * num_repeat)
print('Grand average time: '+str(grand))

out.close()


