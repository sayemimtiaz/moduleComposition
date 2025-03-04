import os

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation

from util.data_util import get_mnist_data, get_fmnist_data, loadTensorFlowDataset

root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
model_name = os.path.join(root, 'h5', 'model_scratch1.h5')

if 'fmnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = get_fmnist_data(hot=True)
elif 'emnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = loadTensorFlowDataset(datasetName='emnist', hot=True)
elif 'kmnist' in model_name:
    x_train, y_train, x_test, y_test, nb_classes = loadTensorFlowDataset(datasetName='kmnist', hot=True)
else:
    x_train, y_train, x_test, y_test, nb_classes = get_mnist_data(hot=True)

model = Sequential()

rnn_model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))

model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(66))
model.add(Activation('relu'))
model.add(Dropout(0.4))
# model.add(Dense(300))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
# model.add(Dense(400))
# model.add(Activation('relu'))
# model.add(Dropout(0.4))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

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

scores = model.predict(x_test)
model.save(model_name)
