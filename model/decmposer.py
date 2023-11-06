import pickle
from datetime import datetime
import tensorflow as tf
from modularization.channeling import channel
from modularization.concern.concern_identification import *

from keras.models import load_model

from dynamic_composer.composer import evaluate_rolled
from util.data_util import get_fmnist_data, get_mnist_data, loadTensorFlowDataset
from util.sampling_util import sample_for_one_output

# initially set unrolled mode , then disable
Constants.disableUnrollMode()

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_name = os.path.join(root, 'h5', 'model_emnist1.h5')
module_path = os.path.join(root, 'modules', extract_model_name(model_name))

firstModel = load_model(model_name)
concernIdentifier = ConcernIdentification()

if 'fmnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = get_fmnist_data(hot=False)
elif 'emnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = loadTensorFlowDataset(datasetName='emnist', hot=False)
elif 'kmnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = loadTensorFlowDataset(datasetName='kmnist', hot=False)
else:
    x_train, y_train, x_test, y_test, nb_classes = get_mnist_data(hot=False)

labs = range(0, nb_classes)
print("Start Time:" + datetime.now().strftime("%H:%M:%S"))

numPosSample = 500
for j in labs:

    model = load_model(model_name)
    positiveConcern = initModularLayers(model.layers)
    negativeConcern = initModularLayers(model.layers)

    print("#Module " + str(j) + " in progress....")
    # sx, sy = sample_for_one_output(x_train, y_train, j, numPosSample)
    #
    # hidden_values_pos = {}
    # for x in sx:
    #     x_t = x
    #
    #     for layerNo, _layer in enumerate(positiveConcern):
    #         x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)
    #
    #         if _layer.type == LayerType.Dense and not _layer.last_layer:
    #             if layerNo not in hidden_values_pos:
    #                 hidden_values_pos[layerNo] = []
    #             hidden_values_pos[layerNo].append(x_t)
    #
    # sx, sy = sample_for_one_output(x_train, y_train, j, numPosSample / (nb_classes - 1), positiveSample=False)
    #
    # hidden_values_neg = {}
    # for x in sx:
    #     x_t = x
    #
    #     for layerNo, _layer in enumerate(negativeConcern):
    #         x_t = concernIdentifier.propagateThroughLayer(_layer, x_t, apply_activation=True)
    #
    #         if _layer.type == LayerType.Dense and not _layer.last_layer:
    #             if layerNo not in hidden_values_neg:
    #                 hidden_values_neg[layerNo] = []
    #             hidden_values_neg[layerNo].append(x_t)

    # for layerNo, _layer in enumerate(positiveConcern):
    #     if _layer.type == LayerType.Dense and not _layer.last_layer:
    #         calculate_active_rate_rolled(hidden_values_pos[layerNo], _layer)
    #         calculate_active_rate_rolled(hidden_values_neg[layerNo], negativeConcern[layerNo])
    #
    # masks = []
    # maxRemove = 0.05
    # prevNumNode = None

    # for layerNo, _layer in enumerate(positiveConcern):
    #     if shouldRemove(_layer):
    #         if _layer.type == LayerType.Dense and not _layer.last_layer:
    #             layerMask = removeAndTangleConcernBasedActiveStat(positiveConcern[layerNo],
    #                                                               negativeConcern[layerNo], maxRemove=maxRemove,
    #                                                               tangleThreshold=-0.5)
    #             layerMask = list(layerMask)
    #
    #             tmp_msk = []
    #             for i in range(prevNumNode):
    #                 tmp_msk.append(layerMask)
    #             masks.append(tf.convert_to_tensor(tmp_msk))
    #             masks.append(tf.convert_to_tensor(layerMask))
    #     elif _layer.last_layer:
    #         layerMask = np.zeros(_layer.num_node, dtype=bool)
    #         layerMask[0] = True
    #         if j == 0:
    #             layerMask[1] = True
    #         else:
    #             layerMask[j] = True
    #             # layerMask[0] = True
    #
    #         tmp_msk = []
    #         for i in range(prevNumNode):
    #             tmp_msk.append(layerMask)
    #         masks.append(tf.convert_to_tensor(tmp_msk))
    #         masks.append(tf.convert_to_tensor(layerMask))
    #
    #     prevNumNode = _layer.num_node

    for layerNo, _layer in enumerate(positiveConcern):
        if _layer.type == LayerType.RepeatVector or _layer.type == LayerType.Flatten \
                or _layer.type == LayerType.Dropout \
                or _layer.type == LayerType.Activation \
                or _layer.type == LayerType.AveragePooling2D:
            continue

        if _layer.type == LayerType.Conv2D or (_layer.type == LayerType.Dense and not _layer.last_layer):
            # model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            model.layers[layerNo].set_weights([_layer.W, _layer.B])
            # getDeadNodePercent(_layer)

        elif _layer.type == LayerType.Embedding:
            model.layers[layerNo].set_weights([_layer.W])
        else:
            channel(_layer, labs, positiveIntent=j)
            model.layers[layerNo].set_weights([_layer.DW, _layer.DB])
            # getDeadNodePercent(_layer)

    model.save(os.path.join(module_path, 'module' + str(j) + '.h5'))

    maskFileName = os.path.join(module_path, 'mask' + str(j) + '.pickle')
    # with open(maskFileName, 'wb') as handle:
    #     pickle.dump(masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

Constants.disableUnrollMode()
evaluate_rolled(model_name)
