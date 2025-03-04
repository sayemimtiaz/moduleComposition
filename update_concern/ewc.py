from copy import deepcopy
import numpy as np
import tensorflow as tf


def fisher_matrix(model, dataset, samples):
    """
    Compute the Fisher matrix, representing the importance of each weight in the
    model. This is approximated using the variance of the gradient of each
    weight, for some number of samples from the dataset.
    :param model: Model whose Fisher matrix is to be computed.
    :param dataset: Dataset which the model has been trained on, but which will
                    not be seen in the future. Formatted as (inputs, labels).
    :param samples: Number of samples to take from the dataset. More samples
                    gives a better approximation of the true variance.
    :return: The main diagonal of the Fisher matrix, shaped to match the weights
             returned by `model.trainable_weights`.
    """
    inputs, labels = dataset
    weights = model.trainable_weights
    variance = [tf.zeros_like(tensor) for tensor in weights]

    for _ in range(samples):
        # Select a random element from the dataset.
        index = np.random.randint(len(inputs))
        data = inputs[index]

        # When extracting from the array we lost a dimension so put it back.
        data = tf.expand_dims(data, axis=0)

        # Collect gradients.
        with tf.GradientTape() as tape:
            output = model(data)
            log_likelihood = tf.math.log(output + 1e-9)

        gradients = tape.gradient(log_likelihood,
                                  weights)  # it computes for a small change in weight, how does output changes?
        # Larger the gradient, higher the impact on output, making it kore important

        # If the model has converged, we can assume that the current weights
        # are the mean, and each gradient we see is a deviation. The variance is
        # the average of the square of this deviation.
        variance = [var + (grad ** 2) for var, grad in zip(variance, gradients)]

    fisher_diagonal = [tensor / samples for tensor in variance]
    return fisher_diagonal


def ewc_loss(lam, model, dataset, samples):
    """
    Generate a loss function which will penalise divergence from the current
    state. It is assumed that the model achieves good accuracy on `dataset`,
    and we want to preserve this behaviour.
    The penalty is scaled according to how important each weight is for the
    given dataset, and `lam` (lambda) applies equally to all weights.
    :param lam: Weight of this cost function compared to the other losses.
    :param model: Model optimised for the given dataset.
    :param dataset: NumPy arrays (inputs, labels).
    :param samples: Number of samples of dataset to take when estimating
                    importance of weights. More samples improves estimates.
    :return: A loss function.
    """
    optimal_weights = deepcopy(model.trainable_weights)
    fisher_diagonal = fisher_matrix(model, dataset, samples)

    def loss_fn(new_model):
        # We're computing:
        # sum [(lambda / 2) * F * (current weights - optimal weights)^2]
        current_weights = new_model.trainable_weights
        loss = sum([tf.reduce_sum(f * tf.square(tf.subtract(c, o))) for f, c, o in
                    zip(fisher_diagonal, current_weights, optimal_weights)])
        return loss * (lam / 2)

    return loss_fn


def fim_mask(model, dataset, samples, threshold):
    """
    Generate a mask for a model. Where the mask is 1, the model's weight may
    be modified.
    :param model: Model optimised for the given dataset.
    :param dataset: NumPy arrays (inputs, labels).
    :param samples: Number of samples of dataset to take when estimating
                    importance of weights. More samples improves estimates.
    :param threshold: Weights with importance > threshold may not be trained
                      further.
    :return: Boolean mask indicating which weights can continue training.
    """
    fisher_diagonal = fisher_matrix(model, dataset, samples)
    mask = [tensor < threshold for tensor in fisher_diagonal]
    return mask


def combine_masks(mask1, mask2):
    """
    :param mask1: A mask generated using `fim_mask`.
    :param mask2: A mask generated using `fim_mask`.
    :return: A new mask, the composition of the two inputs.
    """
    if mask1 is None:
        return mask2
    elif mask2 is None:
        return mask1
    else:
        return [tf.logical_and(tensor1, tensor2)
                for tensor1, tensor2 in zip(mask1, mask2)]


def apply_mask(gradients, mask):
    """
    Apply a gradient mask to a model's gradients.
    :param gradients: Gradients for weights.
    :param mask: Mask indicating which weights are allowed to be updated.
    :return: Gradients with same shape, but some values set to 0.
    """
    if mask is not None:
        gradients = [grad * tf.cast(mask, tf.float32)
                     for grad, mask in zip(gradients, mask)]
        # gradients = [grad * mask
        #              for grad, mask in zip(gradients, mask)]
    return gradients


def clip_gradients(gradients, threshold):
    """
    IncDet's gradient clipping method. EWC by default uses a quadratic loss,
    which can grow very large and lead to unstable training. This method clips
    quadratic terms to linear ones after the given threshold.
    https://ieeexplore.ieee.org/document/9127478
    :param gradients: Gradients of weights.
    :param threshold: Boundary between quadratic and linear gradients.
    :return: Gradients with same shape, but some values clipped.
    """
    # We scale each gradient g by: b / (max(b, |g|))
    result = []
    for tensor in gradients:
        scale = threshold / (tf.math.maximum(threshold, tf.math.abs(tensor)))
        result.append(scale * tensor)
    return result
