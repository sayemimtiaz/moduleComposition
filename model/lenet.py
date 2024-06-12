import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, AveragePooling2D
from sklearn.metrics import precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

from util.data_util import get_mnist_data, get_fmnist_data, loadTensorFlowDataset
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_name = os.path.join(root, 'h5', 'model_mnist1.h5')

if 'fmnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = get_fmnist_data(hot=True)
elif 'emnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = loadTensorFlowDataset(datasetName='emnist', hot=True)
elif 'kmnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = loadTensorFlowDataset(datasetName='kmnist', hot=True)
elif 'mnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = get_mnist_data(hot=True)


def build_lenet5_model(input_shape=(28, 28, 1), num_classes=10, neurons=None):
    if neurons is None:
        neurons = [6, 16, 120, 84]
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(neurons[0], kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(neurons[1], kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Flatten before fully connected layers
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(neurons[2], activation='relu'))
    model.add(Dense(neurons[3], activation='relu'))

    # Output layer
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    return model


if '1' in model_name:
    neurons = [6, 16, 120, 84] #model1
if '2' in model_name:
    neurons = [12, 22, 120, 84] #model2
if '3' in model_name:
    neurons = [18, 28, 120, 84] #model3
if '4' in model_name:
    neurons = [24, 34, 120, 84] #model4
model = build_lenet5_model(neurons=neurons, num_classes=nb_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

if 'scratch' not in model_name:
    epochs = 5

    history = model.fit(x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=32,
                        verbose=2)
    scores = model.evaluate(x_test, y_test, verbose=2)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

# scores = model.predict(x_test)
# model.save(model_name)

# Make predictions
pred = model.predict(x_test[:len(y_test)])
pred = pred.argmax(axis=-1)

# Check if Y_test is one-hot encoded
if len(y_test.shape) > 1:
    true_labels = y_test.argmax(axis=-1)
else:
    true_labels = y_test

# Calculate precision
precision = precision_score(true_labels, pred, average='weighted')

# Calculate recall
recall = recall_score(true_labels, pred, average='weighted')

# Calculate AUC
if len(y_test.shape) > 1:
    prob_pred = model.predict(x_test[:len(y_test)])
    auc = roc_auc_score(y_test, prob_pred, multi_class='ovr')
else:
    prob_pred = model.predict_proba(x_test[:len(y_test)])
    auc = roc_auc_score(y_test, prob_pred[:, 1])

# Print the results
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'AUC: {auc}')

# Print confusion matrix
cm = confusion_matrix(true_labels, pred)
print("Confusion Matrix:\n", cm)

# Print class distribution
unique, counts = np.unique(true_labels, return_counts=True)
class_distribution = dict(zip(unique, counts))
print("Class Distribution:\n", class_distribution)

report = classification_report(true_labels, pred)
print("\nClassification Report:\n", report)