import time

import numpy as np
import tensorflow as tf
from keras.losses import sparse_categorical_crossentropy, categorical_crossentropy
from keras.optimizers import Adam

from update_concern import ewc


def report(model, epoch, validation_datasets, batch_size):
    """
    Print information about training progress. A separate accuracy figure is
    reported for each partition of the validation dataset.
    :param model: Model to evaluate.
    :param epoch: Index of the current epoch.
    :param validation_datasets: List of NumPy tuples (inputs, labels).
    :param batch_size: Number of inputs to be processed simultaneously.
    """
    result = []
    for inputs, labels in validation_datasets:
        _, accuracy = model.evaluate(inputs, labels, verbose=0,
                                     batch_size=batch_size)
        result.append("{:.2f}".format(accuracy * 100))

    # Add 1: assuming that we report after training has finished for this epoch.
    print(epoch + 1, "\t", "\t".join(result))


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


def train_epoch(model, train_data, batch_size,
                gradient_mask=None, incdet_threshold=None):
    """Need a custom training loop for when we modify the gradients."""
    dataset = tf.data.Dataset.from_tensor_slices(train_data)
    dataset = dataset.shuffle(len(train_data[0])).batch(batch_size)

    for inputs, labels in dataset:
        with tf.GradientTape() as tape:
            outputs = model(inputs)
            loss = model.compiled_loss(labels, outputs)

        gradients = tape.gradient(loss, model.trainable_weights)

        # Don't allow gradients to propagate to weights which are important.
        if gradient_mask is not None:
            # start = time.time()
            gradients = ewc.apply_mask(gradients, gradient_mask)
            # end = time.time()
            # print(end - start)

        # Squash some of the very large gradients which EWC can create.
        if incdet_threshold is not None:
            gradients = ewc.clip_gradients(gradients, incdet_threshold)

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))


def train(model, new_data, old_data, epochs=5, batch_size=32, learning_rate=1e-3,
          use_ewc=False, ewc_lambda=1, ewc_samples=100, prior_mask = None,
          use_fim=False, fim_threshold=1e-3, fim_samples=100,
          use_incdet=False, incdet_threshold=None):
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

    start = time.time()

    compile_model(model, learning_rate)

    regularisers = []
    gradient_mask = None
    if not use_incdet:
        incdet_threshold = None

    if use_ewc:
        loss_fn = ewc.ewc_loss(ewc_lambda, model, old_data,
                               ewc_samples)
        regularisers.append(loss_fn)
        compile_model(model, learning_rate, extra_losses=regularisers)
    # If using FIM, determine which weights must be frozen to preserve
    # performance on the current dataset.
    elif use_fim:
        new_mask = ewc.fim_mask(model, old_data, fim_samples,
                                fim_threshold)
        gradient_mask = ewc.combine_masks(gradient_mask, new_mask)

    if prior_mask is not None:
        gradient_mask = ewc.combine_masks(gradient_mask, prior_mask)

    for epoch in range(epochs):
        train_epoch(model, new_data, batch_size,
                    gradient_mask=gradient_mask,
                    incdet_threshold=incdet_threshold)

        # report(model, epoch, valid_sets, batch_size)
    end = time.time()
    print("Took: "+ str(end - start)+" sec")

