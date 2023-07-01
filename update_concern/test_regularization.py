import tensorflow as tf
from keras.layers import Flatten
from tensorflow import keras

from util.data_util import get_mnist_data

x_train, y_train, x_test, y_test, nb_classes = get_mnist_data()

# Define your importance metrics as a dictionary mapping layer names to importance matrices
importance_metrics = { "dense_2": tf.constant([1.0, 0.0])}

# Define your model
model1 = keras.models.Sequential([
    Flatten(input_shape=x_train.shape[1:]),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(2, activation='relu'),
    keras.layers.Dense(nb_classes, activation='softmax')
])


# class ImportanceRegularizationCallback(keras.callbacks.Callback):
#     def __init__(self, importance_metrics, initial_weights, reg_lambda=0.1):
#         super(ImportanceRegularizationCallback, self).__init__()
#         self.importance_metrics = importance_metrics
#         self.reg_lambda = reg_lambda
#         self.initial_weights=initial_weights
#
#     def on_epoch_end(self, epoch, logs=None):
#         # Apply the importance-based regularization to all layers in the model
#         # for layer_name in self.importance_metrics:
#         #     layer = self.model.get_layer(layer_name)
#         #     importance = self.importance_metrics[layer_name]
#         #     weights, biases = layer.get_weights()
#         #
#         #     reg_term_weights = tf.reduce_sum(importance * (weights - self.initial_weights[layer_name][0]))
#         #     reg_term_biases = tf.reduce_sum(importance * (biases - self.initial_weights[layer_name][1]))
#         #     if reg_term_biases.numpy()>0.0:
#         #         print(True)
#         #     reg_term_weights *= self.reg_lambda
#         #     reg_term_biases *= self.reg_lambda
#         #     layer.add_loss(lambda: reg_term_weights)
#         #     layer.add_loss(lambda: reg_term_biases)
#
#         for layer_name in self.importance_metrics:
#             layer = self.model.get_layer(layer_name)
#             importance = self.importance_metrics[layer_name]
#             weights, biases = layer.get_weights()
#             for neuron_idx in range(weights.shape[1]):
#                 reg_term_weights = importance[neuron_idx] * (
#                             weights[:, neuron_idx] - self.initial_weights[layer_name][0][:, neuron_idx])
#                 reg_term_biases = importance[neuron_idx] * (
#                             biases[neuron_idx] - self.initial_weights[layer_name][1][neuron_idx])
#                 layer.add_loss(lambda: self.reg_lambda * tf.reduce_sum(tf.square(reg_term_weights)))
#                 layer.add_loss(lambda: self.reg_lambda * tf.square(reg_term_biases))

class ImportanceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, reg_lambda, initial_weights, importance_metrics):
        self.reg_lambda = reg_lambda
        self.initial_weights = initial_weights
        self.importance_metrics = importance_metrics

    def __call__(self, x, model=None):
        regularization = 0.0
        for layer_name in self.importance_metrics:
            layer = model.get_layer(layer_name)
            importance = self.importance_metrics[layer_name]
            weights, biases = layer.get_weights()
            reg_term_weights = importance * (weights - self.initial_weights[layer_name][0])
            reg_term_biases = importance * (biases - self.initial_weights[layer_name][1])
            regularization += self.reg_lambda * tf.reduce_sum(reg_term_weights) + \
                              self.reg_lambda * tf.reduce_sum(reg_term_biases)
        return regularization


initial_weights = {"dense_2": model1.layers[2].get_weights()}
print(model1.layers[2].get_weights()[1])
# before_w = {}
# for layer in model1.layers:
#     if layer.name in importance_metrics:
#         initial_weights[layer.name] = layer.get_weights()
#
#         before_w[layer.name] = layer.get_weights()[1]
#         print(before_w[layer.name])


model = keras.models.Sequential([
    Flatten(input_shape=x_train.shape[1:]),
    keras.layers.Dense(2, activation='relu', name='dense_1'),
    keras.layers.Dense(2, activation='relu', name='dense_2'),
    keras.layers.Dense(nb_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'],
              loss_regularizer=ImportanceRegularizer(reg_lambda=0.01, initial_weights=initial_weights,
                                                     importance_metrics=importance_metrics)(model=model)
              )

model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=0)

scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

for layer in model.layers:
    if layer.name in importance_metrics:
        after_w = layer.get_weights()[1]
        print(after_w)
