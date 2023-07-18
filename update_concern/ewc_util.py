import csv
import os
import pickle
import random

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from keras.utils import to_categorical

from data_type.constants import Constants, DEBUG
from update_concern.ewc_trainer import train
from util.common import trainModelAndPredictInBinary
from util.data_util import sample, unarize, make_reuse_dataset, combine_for_reuse

base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def update(module, old_train_x, old_train_y, new_train_x, new_train_y, positiveModule, negativeModule, old_dataset,
           new_num_classes, old_num_classes, new_class=None):
    if new_class is not None:
        new_train_x, new_train_y = sample((new_train_x, new_train_y), sample_only_classes=[new_class],
                                          num_sample=500)
    else:
        new_train_x, new_train_y = sample((new_train_x, new_train_y), num_classes=new_num_classes,
                                          balance=True, num_sample=500)
    # sample_only_classes = []
    # for i in range(10):
    #     if i != positiveModule:
    #         sample_only_classes.append(i)
    # old_train_x, old_train_y = sample((old_train_x, old_train_y),
    #                                   balance=True, num_sample=500, sample_only_classes=sample_only_classes)
    temp_y = []
    for i in range(len(new_train_x)):
        temp_y.append(negativeModule)
    new_train_y = to_categorical(temp_y, old_num_classes)

    with open(os.path.join(base_path, 'modules', 'model_' + old_dataset,
                           'mask' + str(positiveModule) + '.pickle'), 'rb') as handle:
        mask = pickle.load(handle)
        # mask=None
        use_fim = True

        use_ewc = False

        adjusted = False
        train(module, (new_train_x, new_train_y), (old_train_x, old_train_y),
              use_fim=use_fim, use_ewc=use_ewc, ewc_samples=500, prior_mask=mask,
              fim_samples=500, fim_threshold=1e-3)

        # for fmt in [1e-3, 1e-5, 1e-7, 1e-15]:
        #     print('FMT: ', fmt)
        #     adjust = train(module, (new_train_x, new_train_y), (old_train_x, old_train_y),
        #                    use_fim=use_fim, use_ewc=use_ewc, ewc_samples=100, prior_mask=mask,
        #                    fim_samples=100, fim_threshold=1e-3)
        #     if adjust > 0:
        #         adjusted=True
        #         print('Adjusted')
        #         break
        # if not adjusted:
        #     print('Not adjusted')
        #     adjust = train(module, (new_train_x, new_train_y), (old_train_x, old_train_y),
        #                    use_fim=False, use_ewc=use_ewc, ewc_samples=100, prior_mask=mask,
        #                    fim_samples=100, fim_threshold=1e-3)
        #     if adjust > 0:
        #         adjusted = True
        #         print('Adjusted')


def update2(module, old_train_x, old_train_y, new_train_x, new_train_y, positiveModule, old_dataset, val_data=None,
            algorithm='EWC'):
    with open(os.path.join(base_path, 'modules', 'model_' + old_dataset,
                           'mask' + str(positiveModule) + '.pickle'), 'rb') as handle:
        use_incdet = False
        use_fim = False
        use_ewc = True
        if algorithm == 'EWC':
            mask = None
        else:
            mask = pickle.load(handle)
            use_incdet = True
            use_ewc = False

        return train(module, (new_train_x, new_train_y), (old_train_x, old_train_y),
                     use_fim=use_fim, use_ewc=use_ewc, ewc_samples=500, prior_mask=mask,
                     fim_samples=500, fim_threshold=1e-3, val_data=val_data,
                     use_incdet=use_incdet, incdet_thres=1e-6, ewc_lambda=1.0)


def evaluate_composition(dataset1, dataset2, modules1, modules2, data1, data2, updated=False):
    Constants.disableUnrollMode()

    outFileName = "static_" + dataset1 + "_" + dataset2
    if updated:
        outFileName += "_updated"
    outFileName += ".csv"

    out = open(os.path.join(base_path, "result", outFileName), "w")

    scratch_model_path = os.path.join(base_path, 'h5', 'model_scratch.h5')

    x_train1, y_train1, x_test1, y_test1, nb_classes1 = data1
    x_train2, y_train2, x_test2, y_test2, nb_classes2 = data2

    diff = 0.0
    no_pairs = 0

    out.write(
        'Class 1,Class 2,Modularized Accuracy,Trained Model Accuracy\n')

    result = {}
    for class1 in modules1.keys():

        for class2 in modules2.keys():

            moduleClass1 = modules1[class1]
            moduleClass2 = modules2[class2]

            xT1, yT1, xt1, yt1 = unarize(x_train1, y_train1, x_test1, y_test1, class1, 0)
            xT2, yT2, xt2, yt2 = unarize(x_train2, y_train2, x_test2, y_test2, class2, 1)

            xT = np.concatenate((xT1, xT2))
            yT = np.concatenate((yT1, yT2))
            xt = np.concatenate((xt1, xt2))
            yt = np.concatenate((yt1, yt2))
            xT, yT = shuffle(xT, yT, random_state=0)
            xt, yt = shuffle(xt, yt, random_state=0)

            predClass1 = moduleClass1.predict(xt)

            predClass2 = moduleClass2.predict(xt)

            finalPred = []
            for i in range(0, len(yt)):

                if predClass1[i][class1] >= predClass2[i][class2]:
                    finalPred.append(0)
                else:
                    finalPred.append(1)

            finalPred = np.asarray(finalPred)
            finalPred = finalPred.flatten()
            scoreModular = accuracy_score(finalPred, np.asarray(yt).flatten())
            if DEBUG:
                print("Modularized Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(
                    scoreModular))

            modelAccuracy = 0.0
            # yT = to_categorical(yT)
            # modelAccuracy, _ = trainModelAndPredictInBinary(scratch_model_path,
            #                                                 xT, yT, xt, yt)
            # print(
            #     "Trained Model Accuracy (Class " + str(class1) + " - Class " + str(class2) + "): " + str(modelAccuracy))

            # diff += (modelAccuracy - scoreModular)
            # no_pairs += 1

            out.write(str(class1) + ',' + str(class2) + ',' + str(scoreModular) + ',' + str(modelAccuracy) + '\n')
            if class1 not in result:
                result[class1] = {}
            result[class1][class2] = scoreModular
    out.close()

    # diff = diff / no_pairs
    # print('Average loss of accuracy: ' + str(diff))
    return result


def evaluate_composition2(modules, data, scratchDict, scratch_time, modular_dict, mode='update', model_suffix=''):
    Constants.disableUnrollMode()
    scratch_model_path = os.path.join(base_path, 'h5', 'model_scratch' + model_suffix + '.h5')

    xT, yT, xt, yt, combo_str, labels, num_classes = combine_for_reuse(modules, data)

    print('Evaluating ' + combo_str)

    if combo_str not in scratchDict:
        yT = to_categorical(yT)
        monScore, elpased = trainModelAndPredictInBinary(scratch_model_path,
                                                         xT, yT, xt, yt,
                                                         nb_classes=num_classes)
        scratchDict[combo_str] = monScore
        scratch_time[combo_str] = elpased

    if combo_str not in modular_dict:
        preds = {}
        for _d in modules:
            preds[_d] = {}
            for _c in modules[_d]:
                preds[_d][_c] = modules[_d][_c].predict(xt, verbose=0)

        predLabels = []
        for i in range(0, len(yt)):
            ks = None
            maxp = None
            minp = None
            winModule = []
            allNegs = []
            for _d in modules:
                for _c in modules[_d]:
                    if _c == 0:
                        nc = 1
                    else:
                        nc = 0
                    if mode == 'positive max':
                        if maxp is None or maxp < preds[_d][_c][i][_c]:
                            maxp = preds[_d][_c][i][_c]
                            ks = (_d, _c)
                    elif mode == 'negative min':
                        if minp is None or minp > preds[_d][_c][i][nc]:
                            minp = preds[_d][_c][i][nc]
                            ks = (_d, _c)
                    elif 'module win' in mode:
                        if preds[_d][_c][i][_c] > preds[_d][_c][i][nc]:
                            winModule.append((_d, _c, nc))
                        allNegs.append((_d, _c, nc))
                    elif mode == 'margin':
                        tmr = preds[_d][_c][i][_c] - preds[_d][_c][i][nc]
                        if maxp is None or maxp < tmr:
                            maxp = tmr
                            ks = (_d, _c)
                    elif mode == 'rate':
                        tmr = preds[_d][_c][i][_c] / preds[_d][_c][i][nc]
                        if maxp is None or maxp < tmr:
                            maxp = tmr
                            ks = (_d, _c)

            if 'module win' in mode:
                minp = None
                maxp = None
                tmpMod = winModule
                if len(tmpMod) == 0:
                    tmpMod = allNegs
                if 'negative min' in mode:
                    for (_d, _c, nc) in tmpMod:
                        if minp is None or preds[_d][_c][i][nc] < minp:
                            minp = preds[_d][_c][i][nc]
                            ks = (_d, _c)
                elif 'positive max' in mode:
                    for (_d, _c, nc) in tmpMod:
                        if maxp is None or preds[_d][_c][i][_c] > maxp:
                            maxp = preds[_d][_c][i][_c]
                            ks = (_d, _c)
                elif 'rate' in mode:
                    for (_d, _c, nc) in tmpMod:
                        tmr = preds[_d][_c][i][_c] / preds[_d][_c][i][nc]
                        if maxp is None or tmr > maxp:
                            maxp = tmr
                            ks = (_d, _c)
            predLabels.append(labels[ks[0]][ks[1]])

        predLabels = np.asarray(predLabels)
        predLabels = predLabels.flatten()
        modScore = accuracy_score(predLabels, np.asarray(yt).flatten())
    else:
        modScore = modular_dict[combo_str]

    print('Evaluating ' + combo_str)
    print("Modularized Accuracy (" + mode + "): " + str(modScore))
    print("Trained Accuracy: " + str(scratchDict[combo_str]))

    return combo_str, modScore, scratchDict[combo_str]


def find_modules(ds, _c, mc, combos, totalModuleCount, data):
    rs = len(ds)
    for _i, _d in enumerate(ds):
        if mc > totalModuleCount:
            return False
        if _i + 1 == len(ds):
            tc = min(mc - rs + 1, data[_d][4])
        else:
            tc = random.randint(1, min(mc - rs + 1, data[_d][4]))

        combos[_c][_d] = sorted(np.random.choice(range(data[_d][4]),
                                                 tc, replace=False))

        mc -= tc
        rs -= 1
        totalModuleCount -= data[_d][4]

    return mc == 0


def get_combos(data, datasets, total_combination=10, _seed=19):
    combos = {}

    random.seed(_seed)
    np.random.seed(_seed)

    for _c in range(total_combination):
        combos[_c] = {}
        ds = random.randint(2, len(datasets))
        ds = np.random.choice(datasets, ds, replace=False)
        totalModuleCount = 0
        for _d in ds:
            totalModuleCount += data[_d][4]

        mc = random.randint(len(ds), totalModuleCount)

        while True:
            if find_modules(ds, _c, mc, combos, totalModuleCount, data):
                break
            if DEBUG:
                print('Retrying combo matching: ' + str(_c))
    # print(combos)
    return combos


def load_combos(name='update_model4'):
    combos = {}
    scratchDict = {}
    scratchTime = {}
    modular_time = {}
    modular_dict = {}

    fileName = os.path.join(base_path, "result", name+".csv")
    with open(fileName) as csv_file:
        csv_reader = csv.reader(csv_file)
        next(csv_reader)
        _i = 0
        for row in csv_reader:
            combos[_i] = {}
            tcmb = row[0].strip()
            scratchDict[tcmb] = float(row[2])
            scratchTime[tcmb] = float(row[5])

            modular_dict[tcmb] = float(row[1])
            modular_time[tcmb] = float(row[4])

            tcmb = tcmb.replace('(', '')
            tcmb = tcmb.split(')')

            for _c in tcmb[:-1]:
                tc = _c.split(':')
                _d = tc[0].strip()
                tc = tc[1].split('-')[:-1]

                rt = []
                for r in tc:
                    rt.append(int(r))
                rt = sorted(rt)
                combos[_i][_d] = rt
            _i += 1

    return combos, scratchDict, scratchTime, modular_dict, modular_time
