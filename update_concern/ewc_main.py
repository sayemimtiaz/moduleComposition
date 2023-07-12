import os

from keras.models import load_model
from update_concern.ewc_util import update, evaluate_composition
from util.data_util import get_mnist_data, get_fmnist_data

total_run=10

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

mnist_mod = None
fmnist_mod = None
dataset_A = 'mnist'
dataset_B = 'fmnist'

if dataset_A == 'mnist':
    A_train_x, A_train_y, A_test_x, A_test_y, A_num_classes = get_mnist_data(hot=False)
if dataset_A == 'fmnist':
    A_train_x, A_train_y, A_test_x, A_test_y, A_num_classes = get_fmnist_data(hot=False)
if dataset_B == 'fmnist':
    B_train_x, B_train_y, B_test_x, B_test_y, B_num_classes = get_fmnist_data(hot=False)
if dataset_B == 'mnist':
    B_train_x, B_train_y, B_test_x, B_test_y, B_num_classes = get_mnist_data(hot=False)

result={}
for rn in range(total_run):
    A_updated_modules = {}
    for A_cN in range(A_num_classes):

        if mnist_mod is not None and mnist_mod != A_cN:
            continue

        module_A = load_model(os.path.join(base_path, 'modules', 'model_' + dataset_A,
                                           'module' + str(A_cN) + '.h5'))
        positiveModule = A_cN
        negativeModule = 0
        if A_cN == 0:
            negativeModule = 1

        update(module_A, A_train_x, A_train_y, B_train_x, B_train_y, positiveModule, negativeModule,
               dataset_A, B_num_classes)

        A_updated_modules[A_cN] = module_A

    B_updated_modules = {}
    for B_cN in range(B_num_classes):

        if fmnist_mod is not None and fmnist_mod != B_cN:
            continue

        module_B = load_model(os.path.join(base_path, 'modules', 'model_' + dataset_B,
                                           'module' + str(B_cN) + '.h5'))
        positiveModule = B_cN
        negativeModule = 0
        if B_cN == 0:
            negativeModule = 1

        update(module_B, B_train_x, B_train_y, A_train_x, A_train_y, positiveModule, negativeModule,
               dataset_B, A_num_classes)

        B_updated_modules[B_cN] = module_B

    rs_rn=evaluate_composition(dataset_A, dataset_B, A_updated_modules, B_updated_modules
                         , (A_train_x, A_train_y, A_test_x, A_test_y, A_num_classes),
                         (B_train_x, B_train_y, B_test_x, B_test_y, B_num_classes),
                         updated=True)
    for c1 in rs_rn.keys():
        for c2 in rs_rn[c1].keys():
            if c1 not in result:
                result[c1]={}
            if c2 not in result[c1]:
                result[c1][c2]=0
            result[c1][c2]+=rs_rn[c1][c2]

for c1 in result.keys():
    for c2 in result[c1].keys():
        result[c1][c2]/=total_run

out = open(os.path.join(base_path, "result", "update.csv"), "w")
out.write(
    'Class 1,Class 2,Modularized Accuracy,Trained Model Accuracy\n')
for c1 in result.keys():
    for c2 in result[c1].keys():
        out.write(str(c1) + ',' + str(c2) + ',' + str(result[c1][c2]) + ',' + str(0) + '\n')

out.close()