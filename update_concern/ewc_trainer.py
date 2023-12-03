import time

import numpy as np
import tensorflow as tf
from keras import regularizers
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.optimizers import Adam

from data_type.constants import DEBUG
from update_concern import ewc

# model = None
# train_step_fun = None
# gradient_mask = None
# incdet_threshold = None


def evaluate(model, val_data):
    """
    Print information about training progress. A separate accuracy figure is
    reported for each partition of the validation dataset.
    :param model: Model to evaluate.
    :param epoch: Index of the current epoch.
    :param validation_datasets: List of NumPy tuples (inputs, labels).
    :param batch_size: Number of inputs to be processed simultaneously.
    """
    _, accuracy = model.evaluate(val_data[0], val_data[1], verbose=0)

    return accuracy


def compile_model(model, learning_rate, extra_losses=None):
    def custom_loss(y_true, y_pred):
        # loss = sparse_categorical_crossentropy(y_true, y_pred)
        loss = categorical_crossentropy(y_true, y_pred)
        if extra_losses is not None:
            for fn in extra_losses:
                loss += fn(model)

        return loss

    model.compile(
        loss=custom_loss,
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )
    return model


# @tf.function
def train_step(model, inputs, labels, incdet_threshold=None, gradient_mask=None):
    with tf.GradientTape() as tape:
        outputs = model(inputs)
        loss = model.compiled_loss(labels, outputs)

    gradients = tape.gradient(loss, model.trainable_weights)
    # # Don't allow gradients to propagate to weights which are important.
    if gradient_mask is not None:
        gradients = ewc.apply_mask(gradients, gradient_mask)

    # Squash some of the very large gradients which EWC can create.
    if incdet_threshold is not None:
        gradients = ewc.clip_gradients(gradients, incdet_threshold)

    model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


def train_epoch(model, train_data, batch_size,
                gradient_mask=None, incdet_threshold=None, train_step_fun=None):
    # global train_step_fun
    """Need a custom training loop for when we modify the gradients."""
    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(len(train_data[0])).batch(batch_size)
    # print('train_epoch called')

    # overall_loss = tf.keras.metrics.Mean()
    for inputs, labels in dataset:
        loss = train_step_fun(model, inputs, labels, incdet_threshold=incdet_threshold, gradient_mask=gradient_mask)
        # overall_loss.update_state(loss)
        # for variable in model.trainable_variables:
        #     if tf.reduce_any(tf.math.is_nan(variable)):
        #         print('Nan weights detected')
        #         break

    # return overall_loss.result().numpy()
    return 0


def compute_ewc_penalty_terms(model, old_data, ewc_samples=500, ewc_lambda=0.1, learning_rate=1e-3):
    start = time.time()
    regularisers = []

    loss_fn = ewc.ewc_loss(ewc_lambda, model, old_data,
                           ewc_samples)
    regularisers.append(loss_fn)
    compile_model(model, learning_rate, extra_losses=regularisers)
    end = time.time()
    return end - start


def train(_model, new_data, old_data, val_data=None, epochs=100, batch_size=32, learning_rate=1e-3,
          use_ewc=False, ewc_lambda=1, ewc_samples=100, prior_mask=None,
          use_fim=False, fim_threshold=1e-3, fim_samples=100,
          use_incdet=False, incdet_thres=1e-9, patience=3):
    """
    Train a model using a complete dataset.
    :param model: Model to be trained.
    :param train_data: Training dataset.
    :param epochs: Number of epochs to train for.
    :param batch_size: Number of inputs to process simultaneously.
    :param learning_rate: Initial learning rate for Adam optimiser.
    :param use_ewc: Should EWC be used?
    :param ewc_lambda: Relative weighting of EWC loss vs normal loss.
    :param ewc_samples: Samples of dataset to take when initialising EWC.
    :param use_fim: Should Fisher information masking be used?
    :param fim_threshold: How important a parameter must be to stop training.
    :param fim_samples: Samples of dataset to take when initialising FIM.
    :param use_incdet: Should IncDet (incremental detection) be used?
    :param incdet_threshold: Threshold for IncDet gradient clipping.
    """
    # global model, train_step_fun, gradient_mask, incdet_threshold
    incdet_threshold = incdet_thres

    model = compile_model(_model, learning_rate)

    regularisers = []
    gradient_mask = None
    if not use_incdet:
        incdet_threshold = None

    start = time.time()
    if use_ewc:
        loss_fn = ewc.ewc_loss(ewc_lambda, model, old_data,
                               ewc_samples)
        regularisers.append(loss_fn)
        compile_model(model, learning_rate, extra_losses=regularisers)

    # If using FIM, determine which weights must be frozen to preserve
    # performance on the current dataset.
    if use_fim:
        new_mask = ewc.fim_mask(model, old_data, fim_samples,
                                fim_threshold)
        gradient_mask = ewc.combine_masks(gradient_mask, new_mask)

    if prior_mask is not None:
        gradient_mask = ewc.combine_masks(gradient_mask, prior_mask)
        adjust = 0
        freeze = 0
        for _m1 in gradient_mask:
            na = _m1.numpy()
            adjust += na.sum()
            if len(na.shape) == 2:
                freeze += (na.shape[0] * na.shape[1]) - na.sum()
            else:
                freeze += na.shape[0] - na.sum()
        if DEBUG:
            print(adjust, freeze)

        if DEBUG:
            print('Adjustment rate: ' + str((adjust / (adjust + freeze)) * 100.0) + '%')

    end = time.time()

    setupTime=end-start
    wait = 0
    priorAccuracy = evaluate(model, val_data)
    if DEBUG:
        print('Accuracy before training: ' + str(priorAccuracy))
    actualEpoch = 0
    best_weights = model.get_weights()
    improved = False

    train_step_fun = tf.function(train_step)

    start = time.time()

    for epoch in range(epochs):
        _ = train_epoch(model, new_data, batch_size,
                        gradient_mask=gradient_mask,
                        incdet_threshold=incdet_threshold, train_step_fun=train_step_fun)
        currentAccuracy = evaluate(model, val_data)
        wait += 1
        if currentAccuracy > priorAccuracy:
            wait = 0
            improved = True
            best_weights = model.get_weights()

        if wait >= patience and epoch > 0:
            if improved:
                model.set_weights(best_weights)
            break

        priorAccuracy = currentAccuracy
        actualEpoch += 1

        # report(model, epoch, valid_sets, batch_size)
    end = time.time()

    del train_step_fun
    # train_step_fun = None
    if DEBUG:
        print("Took: " + str(end - start) + " sec")
        print('Actual epoch: ' + str(actualEpoch))
        print('Accuracy after training: ' + str(evaluate(model, val_data)))

    return setupTime, end - start
