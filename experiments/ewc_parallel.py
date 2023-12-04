import concurrent
import copy
import multiprocessing
from concurrent.futures._base import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from multiprocessing import Manager
import os
import time
from keras.models import load_model

from update_concern.ewc_util import update_module, evaluate_ewc
from util.common import load_smallest_comobs, load_combos
from util.data_util import load_data_by_name, sample_train_ewc, sample_test_ewc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    num_sample_test = 100
    num_sample_train = 500
    logOutput = True
    datasets = ['mnist', 'fmnist', 'kmnist', 'emnist']
    start_index = 0
    end_index = 199
    numMemorySample = 500
    positiveRatioInValid = 1.0
    max_workers = multiprocessing.cpu_count()

    data = {}
    for _d in datasets:
        data[_d] = load_data_by_name(_d, hot=False)

    comboList = load_combos(start=start_index, end=end_index)
    # comboList = load_smallest_comobs(bottom=3)
    if logOutput:
        out = open(os.path.join(base_path, "result", "ewc_composition" + ".csv"), "w")
        out.write(
            'Combination ID,Modularized Accuracy,Setup Time,Update Time,Inference Time\n')
        out.close()

    cPattern = {}
    moduleCount = {}
    cmbId = start_index
    for _cmb in range(len(comboList)):
        if logOutput:
            out = open(os.path.join(base_path, "result", "ewc_composition" + ".csv"), "a")

        modules = {}
        updated_modules = {}
        moduleCount[_cmb] = 0
        setupTime = 0
        updateTime = 0
        start = time.time()
        for (_d, _c, _m) in comboList[_cmb]:
            print(_d, _c, _m)

            moduleCount[_cmb] += 1
            if _d not in modules:
                modules[_d] = {}
                updated_modules[_d] = {}
            if _c not in modules[_d]:
                modules[_d][_c] = {}
                updated_modules[_d][_c] = {}

            # module = load_model(os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
            #                                  'module' + str(_c) + '.h5'))

            modules[_d][_c][_m] = os.path.join(base_path, 'modules', 'model_' + _d + str(_m),
                                               'module' + str(_c) + '.h5')

        with Manager() as manager:

            futures = []

            with ProcessPoolExecutor(max_workers=max_workers) as executor:

                for (_d, _c, _m) in comboList[_cmb]:
                    positiveModule = _c
                    negativeModule = 0
                    if _c == 0:
                        negativeModule = 1
                    nx, ny = sample_train_ewc(data, (_d, _c, _m), comboList[_cmb],
                                              negativeModule, positiveModule,
                                              num_sample=num_sample_train,
                                              includePositive=True,
                                              numMemorySample=numMemorySample)
                    val_data = sample_test_ewc(data, (_d, _c, _m), comboList[_cmb],
                                               negativeModule,
                                               positiveModule,
                                               positiveRatio=positiveRatioInValid)
                    nx_copy = copy.deepcopy(nx)
                    ny_copy = copy.deepcopy(ny)
                    val_data_copy = copy.deepcopy(val_data)

                    futures.append(executor.submit(update_module, module=modules[_d][_c][_m],
                                                   old_train_x=copy.deepcopy(data[_d][0]),
                                                   old_train_y=copy.deepcopy(data[_d][1]),
                                                   new_train_x=nx_copy,
                                                   new_train_y=ny_copy,
                                                   val_data=val_data_copy))

                # futures, _ = concurrent.futures.wait(futures)

                for future in as_completed(futures):
                    curSetupTime, curUpdateTime = future.result()
                    setupTime += curSetupTime
                    updateTime += curUpdateTime
                    # updated_modules[_d][_c][_m] = model

        end = time.time()
        print('Overall time: ', (end - start))
        setupTime /= moduleCount[_cmb]
        updateTime /= moduleCount[_cmb]

        for (_d, _c, _m) in comboList[_cmb]:
            module = load_model(os.path.join(base_path, 'modules', 'updated', 'model_' + _d + str(_m),
                                             'module' + str(_c) + '.h5'), compile=False)

            updated_modules[_d][_c][_m] = module

        score, eval_time = evaluate_ewc(updated_modules, data,
                                        num_sample=num_sample_test,
                                        num_module=moduleCount[_cmb])
        if logOutput:
            out.write(str(cmbId)
                      + ',' + str(score) + ',' + str(setupTime) + ',' + str(updateTime) + ',' +
                      str(eval_time)
                      + '\n')

            out.close()
        cmbId += 1
