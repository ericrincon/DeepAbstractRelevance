from keras.layers import Input,Embedding, merge, Dense

from keras.models import Model

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Activation, Dropout

from keras.optimizers import Adam, SGD

from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping

import sklearn.metrics as metrics
import numpy as np


# Implementation of Convolutional Neural Networks for Sentence Classification
# Paper: http://arxiv.org/abs/1408.5882
class CNN:
    def __init__(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_filters,
                 dense_layer_sizes, name, activation_function, dropout_p):
        self.model = self.build_model(n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes,
                                      n_filters, dense_layer_sizes, activation_function, dropout_p)
        self.model_name = name

    def build_model(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_filters,
                    dense_layer_sizes, activation_function, dropout_p):

        # From the paper http://arxiv.org/pdf/1511.07289v1.pdf
        # Supposed to perform better but lets see about that
        if activation_function == 'elu':
            activation_function = ELU(alpha=1.0)


        if use_embedding:
            w2v_input = Input(shape=(max_words, ))
            conv_input = Embedding(output_dim=w2v_size, input_dim=vocab_size, input_length=max_words)(w2v_input)
        else:
            w2v_input = Input(shape=(1, max_words, w2v_size))
            conv_input = w2v_input
        conv_layers = []

        for filter_size in filter_sizes:
            convolution_layer = Convolution2D(n_filters, filter_size, w2v_size,
                                              input_shape=(1, max_words, w2v_size))(conv_input)

            max_layer = MaxPooling2D(pool_size=(max_words - filter_size + 1, 1))(convolution_layer)
            activation_layer = Activation(activation_function)(max_layer)
            flattened_layer = Flatten()(activation_layer)

            conv_layers.append(flattened_layer)

        merge_layer = merge(conv_layers, mode='concat')

        dense_layers = []
        first_dense_layer = Dense(dense_layer_sizes.pop(0))(merge_layer)
        dense_layers.append(first_dense_layer)

        if len(dense_layer_sizes) > 1:
            i = 1

            for dense_layer_size in dense_layer_sizes:
                dense_layer = Dense(dense_layer_size)(dense_layer_sizes[i-1])
                dense_activation_layer = Activation(activation_function)(dense_layer)
                dropout_layer = Dropout(dropout_p)(dense_activation_layer)
                dense_layers.append(dropout_layer)
                i += 1

        # Add last layer
        softmax_dense_layer = Dense(n_classes)(dense_layers[-1])
        softmax_layer = Activation('softmax')(softmax_dense_layer)

        model = Model(input=w2v_input, output=softmax_layer)

        return model

    def train(self, x, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy', save_model=True,
              verbose=2):

        if optim_algo == 'adam':
            optim_algo = Adam()
        else:
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

        self.model.compile(optimizer=optim_algo, loss=criterion)

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto')

        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        self.model.fit(x, y, nb_epoch=n_epochs, callbacks=[early_stopping], validation_split=0.2,
                       verbose=verbose, batch_size=16)

        if save_model:
            self.model.save_weights(self.model_name + '.h5', overwrite=True)

    def test(self, x, y):
        truth = []
        predictions = self.predict_classes(x)

        for i in range(y.shape[0]):
            if y[i, 1] == 1:
                truth.append(1)
            else:
                truth.append(0)

        accuracy = metrics.accuracy_score(truth, predictions)
        f1_score = metrics.f1_score(truth, predictions)
        precision = metrics.precision_score(truth, predictions)
        auc = metrics.roc_auc_score(truth, predictions)
        recall = metrics.recall_score(truth, predictions)

        return accuracy, f1_score, precision, auc, recall

    def predict_classes(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def save(self):
        self.model.save_weights(self.model_name)
# Implementation of Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network
