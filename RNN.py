from keras.layers.recurrent import LSTM

from keras.models import Model
from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

import numpy as np

import sklearn.metrics as metrics

class lstm():
    def __init__(self, max_words, w2v_length):
        self.model = self.build_model(max_words, w2v_length)
    def train(self, X, y, n_epochs, optimizer='adam', criterion='categorical_crossentropy',
              batch_size=64):
        self.model.compile(optimizer=optimizer, loss=criterion)

        self.model.fit(X, y, nb_epoch=n_epochs, batch_size=batch_size, shuffle=True)

    def test(self, X, y, print_output=False):
        truth = []
        predictions = self.predict_classes(X)

        for i in range(y.shape[0]):
            if y[i, 1] == 1:
                truth.append(1)
            else:
                truth.append(0)

        if print_output:
            print('Ground truth: {}'.format(truth))
            print('Predicted: {}'.format(predictions))

        accuracy = metrics.accuracy_score(truth, predictions)
        f1_score = metrics.f1_score(truth, predictions)
        precision = metrics.precision_score(truth, predictions)
        auc = metrics.roc_auc_score(truth, predictions)
        recall = metrics.recall_score(truth, predictions)

        return accuracy, f1_score, precision, auc, recall

    def build_model(self, max_words, w2v_length):
        model = Sequential()

        model.add(LSTM(200, input_shape=(max_words, w2v_length)))
     #   model.add(Dense(200))
      #  model.add(Activation('relu'))
      #  model.add(Dropout(.5))
        model.add(Dense(2))
        model.add(Activation('softmax'))

        return model

    def predict_classes(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

