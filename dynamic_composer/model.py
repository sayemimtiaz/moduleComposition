from keras.models import Sequential
from keras.layers import Dense, Flatten

from util.data_util import get_mnist_data, get_fmnist_data

# x_train, y_train, x_test, y_test, nb_classes = get_mnist_data()
x_train, y_train, x_test, y_test, nb_classes = get_fmnist_data()


model = Sequential()

rnn_model = Sequential()

model.add(Flatten(input_shape=x_train.shape[1:]))

model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

epochs = 5

history = model.fit(x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=32,
                    verbose=2)
scores = model.evaluate(x_test, y_test, verbose=2)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

scores = model.predict(x_test)
model.save('h5/model_fmnist.h5')
# model.save('h5/model_scratch.h5')
