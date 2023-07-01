import os
import pickle

from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
from sklearn.metrics import accuracy_score

from update_concern.ewc_trainer import train
from util.data_util import get_mnist_data, get_fmnist_data, unarize, sample

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

DO_UPDATE = True
old_dataset = 'mnist'
new_dataset = 'fmnist'
positiveModule = 1
negativeModule = 0

module = load_model(os.path.join(base_path, 'modules', 'model_' + old_dataset,
                                 'module' + str(positiveModule) + '.h5'))

if old_dataset == 'mnist':
    old_train_x, old_train_y, old_test_x, old_test_y, old_num_classes = get_mnist_data(hot=False)
    # _, _, test_x1, test_y1 = unarize(old_train_x, old_train_y, old_test_x, old_test_y,
    #                                        positiveModule, positiveModule)

    temp_y = []
    for i in range(len(old_test_x)):
        if old_test_y[i] == positiveModule:
            temp_y.append(positiveModule)
        else:
            temp_y.append(negativeModule)

    old_test_y = temp_y

    # sample_only_classes = []
    # for i in range(old_num_classes):
    #     if i == positiveModule:
    #         sample_only_classes.append(i)
    # old_train_x, old_train_y = sample((old_train_x, old_train_y),
    #                                   balance=True, num_sample=1500, sample_only_classes=sample_only_classes)
    # temp_y = []
    # for i in range(len(old_train_y)):
    #     temp_y.append(negativeModule)
    # old_train_y = temp_y

if new_dataset == 'fmnist':
    new_train_x, new_train_y, new_test_x, new_test_y, new_num_classes = get_fmnist_data(hot=False)
    new_train_x, new_train_y = sample((new_train_x, new_train_y), num_classes=new_num_classes,
                                      balance=True, num_sample=500)

    temp_y = []
    for i in range(len(new_train_x)):
        temp_y.append(negativeModule)
    new_train_y = to_categorical(temp_y, new_num_classes)

    temp_y = []
    for i in range(len(new_test_x)):
        temp_y.append(negativeModule)
    new_test_y = temp_y

if DO_UPDATE:
    with open(os.path.join(base_path, 'modules', 'model_' + old_dataset,
                           'mask' + str(positiveModule) + '.pickle'), 'rb') as handle:
        mask = pickle.load(handle)
        # mask=None
        use_fim = True
        use_ewc = False
    train(module, (new_train_x, new_train_y), (old_train_x, old_train_y),
          use_fim=use_fim, use_ewc=use_ewc, ewc_samples=500, prior_mask=mask,
          fim_samples=500)

test_x = np.concatenate((old_test_x, new_test_x))
test_y = np.concatenate((old_test_y, new_test_y))

finalPred = []
p = module.predict(test_x[:len(test_y)])
for i in range(0, len(test_y)):
    if p[i][positiveModule] > p[i][negativeModule]:
        finalPred.append(positiveModule)
    else:
        finalPred.append(negativeModule)

from sklearn.metrics import accuracy_score

score = accuracy_score(finalPred, test_y)

print(score)
