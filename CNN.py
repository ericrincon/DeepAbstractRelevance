from keras.layers import Input,Embedding, merge, Dense

from keras.models import Model

from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Flatten, Activation, Dropout

from keras.optimizers import Adam, SGD, Adagrad

from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping

from keras.constraints import maxnorm

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
            w2v_input = Input(shape=(max_words, w2v_size))
            conv_input = w2v_input
        conv_layers = []

        for filter_size in filter_sizes:
            convolution_layer = Convolution1D(nb_filter=n_filters, input_dim=max_words, filter_length=filter_size,
                                              input_shape=(max_words, w2v_size))(conv_input)

            max_pooling = MaxPooling1D(pool_length=(max_words - filter_size + 1))(convolution_layer)
            activation_layer = Activation(activation_function)(max_pooling)
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
              verbose=2, plot=True, tensorboard_path='./logs', patience=20):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')

        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        history = self.model.fit(x, y, nb_epoch=n_epochs, callbacks=[early_stopping], validation_split=0.2,
                       verbose=verbose, batch_size=32, shuffle=True)



        if save_model:
            self.model.save_weights(self.model_name + '.h5', overwrite=True)

    def test(self, x, y, print_output=False):
        truth = []
        predictions = self.predict_classes(x)

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

    def predict_classes(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes


class CICNN:
    def __init__(self, n_classes, max_words, w2v_size, vocab_size, filter_sizes, n_filters,
                 dense_layer_sizes, name, activation_function, dropout_p):
        self.model = self.build_model(n_classes, max_words, w2v_size, vocab_size, filter_sizes,
                                      n_filters, dense_layer_sizes, activation_function, dropout_p)
        self.model_name = name


    def _build_conv_node(self, activation_function, max_words, w2v_size, vocab_size, n_filters, filter_sizes):
        # From the paper http://arxiv.org/pdf/1511.07289v1.pdf
        # Supposed to perform better but lets see about that
        if activation_function == 'elu':
            activation_function = ELU(alpha=1.0)

        w2v_input = Input(shape=(max_words, ))
        conv_input = Embedding(output_dim=w2v_size, input_dim=vocab_size, input_length=max_words)(w2v_input)
        conv_layers = []

        for filter_size in filter_sizes:
            convolution_layer = Convolution2D(n_filters, filter_size, w2v_size,
                                              input_shape=(1, max_words, w2v_size))(conv_input)

            max_layer = MaxPooling2D(pool_size=(max_words - filter_size + 1, 1))(convolution_layer)
            activation_layer = Activation(activation_function)(max_layer)
            flattened_layer = Flatten()(activation_layer)

            conv_layers.append(flattened_layer)

        merge_layer = merge(conv_layers, mode='concat')

        return merge_layer

    def build_model(self, n_classes, max_words, w2v_size, vocab_size, filter_sizes, n_filters,
                    dense_layer_sizes, activation_function, dropout_p):

        positive_node, positive_input = self._build_conv_node(activation_function, max_words, w2v_size, vocab_size, n_filters, filter_sizes)
        negative_node, negative_input = self._build_conv_node(activation_function, max_words, w2v_size, vocab_size, n_filters, filter_sizes)

        merged_dense_layer = merge([[positive_node, negative_node], 'concat'])

        dense_layers = []
        first_dense_layer = Dense(dense_layer_sizes.pop(0))(merged_dense_layer)
        dense_layers.append(first_dense_layer)

        if len(dense_layer_sizes) > 1:
            i = 1

            for dense_layer_size in dense_layer_sizes:
                dense_layer = Dense(dense_layer_size, W_constraint=maxnorm(2))(dense_layer_sizes[i-1])
                dense_activation_layer = Activation(activation_function)(dense_layer)
                dropout_layer = Dropout(dropout_p)(dense_activation_layer)
                dense_layers.append(dropout_layer)
                i += 1

        # Add last layer
        softmax_dense_layer = Dense(n_classes)(dense_layers[-1])
        softmax_layer = Activation('softmax')(softmax_dense_layer)

        model = Model(input=[positive_input, negative_input], output=softmax_layer)

        return model

    def train(self, X_positive, X_negative, y_positive, y_negative, n_epochs, optim_algo='adam',
              criterion='categorical_crossentropy', save_model=True,
              verbose=2, use_tensorboard=False):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        callbacks = []
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')
        callbacks.append(early_stopping)

        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        # @TODO this is not correct! Fix it!!!!
        self.model.fit([X_positive, X_negative], [y_positive, y_negative], nb_epoch=n_epochs, callbacks=[early_stopping], validation_split=0.2,
                       verbose=verbose, batch_size=16)

        if save_model:
            self.model.save_weights(self.model_name + '.h5', overwrite=True)

    def test(self, x, y, print_output=False):
        truth = []
        predictions = self.predict_classes(x)

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

    def predict_classes(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def save(self):
        self.model.save_weights(self.model_name)


class AbstractCNN:
    def __init__(self, n_classes, w2v_size, vocab_size, use_embedding, name, activation_function, dropout_p,
                 dense_layer_sizes, max_words=None, n_feature_maps=None, filter_sizes=None, embedding=None):

        if max_words is None:
            max_words = {'text': 220, 'mesh': 40, 'title': 14}

        if filter_sizes is None:
            filter_sizes = {'text': [2, 3, 5],
                            'mesh': [2, 3, 5],
                            'title': [2, 3, 5]}
        if n_feature_maps is None:
            n_feature_maps = {'text': 100, 'mesh': 50, 'title': 50}

        self.model = self.build_model(n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes,
                                      n_feature_maps, dense_layer_sizes, activation_function, dropout_p,
                                      embedding=embedding)
        self.model_name = name

    def build_conv_node(self, n_feature_maps, max_words, w2v_size, activation_function, filter_sizes, vocab_size,
                        use_embedding, name, embedding=None):

        conv_input = Input(shape=(max_words, w2v_size), name=name)

        """
        if use_embedding:
            assert embedding is not None, 'Make sure you pass the embedding weights!'

            w2v_input = Input(shape=(max_words, ), name=name)
            conv_input = Embedding(output_dim=w2v_size, input_dim=vocab_size, input_length=max_words,
                                   weights=[embedding])(w2v_input)
        else:
            w2v_input = Input(shape=(1, max_words, w2v_size))
            conv_input = w2v_input

        """
        conv_layers = []

        for filter_size in filter_sizes:

            if use_embedding:
                convolution_layer = Convolution1D(n_feature_maps, filter_size, input_shape=(max_words, w2v_size))(conv_input)
                max_layer = MaxPooling1D(pool_length=max_words - filter_size + 1)(convolution_layer)
            else:
                convolution_layer = Convolution1D(n_feature_maps, filter_size, input_shape=(max_words, w2v_size))(conv_input)

                max_layer = MaxPooling1D(pool_length=(max_words - filter_size + 1))(convolution_layer)

            activation_layer = Activation(activation_function)(max_layer)
            flattened_layer = Flatten()(activation_layer)

            conv_layers.append(flattened_layer)

        merge_layer = merge(conv_layers, mode='concat')

        return merge_layer, conv_input

    def build_model(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_feature_maps,
                    dense_layer_sizes, activation_function, dropout_p, embedding=None):

        # From the paper http://arxiv.org/pdf/1511.07289v1.pdf
        # Supposed to perform better but lets see about that
        if activation_function == 'elu':
            activation_function = ELU(alpha=1.0)

        abstract_node, abstract_input = self.build_conv_node(n_feature_maps['text'], max_words['text'], w2v_size,
                                                             activation_function, filter_sizes['text'], vocab_size,
                                                             use_embedding, 'abstract_input', embedding=embedding)
        title_node, title_input = self.build_conv_node(n_feature_maps['title'], max_words['title'], w2v_size,
                                                       activation_function, filter_sizes['title'], vocab_size,
                                                       use_embedding, 'title_input', embedding=embedding)
        mesh_node, mesh_input = self.build_conv_node(n_feature_maps['mesh'], max_words['mesh'], w2v_size,
                                                     activation_function, filter_sizes['mesh'], vocab_size,
                                                     use_embedding, 'mesh_terms_input', embedding=embedding)

        merge_layer = merge([abstract_node, title_node, mesh_node], mode='concat')

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

        model = Model(input=[abstract_input, title_input, mesh_input], output=softmax_layer)

        return model

    def train(self, X_abstract, X_titles, X_mesh, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy',
              save_model=True, verbose=2, plot=True, tensorBoard_path='', patience=20, use_tensorboard=False):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        callbacks = []

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, mode='auto')

        callbacks.append(early_stopping)

        # verbose: 0 for no logging to stdout, 1 for progress bar logging, 2 for one log line per epoch.
        self.model.fit([X_abstract, X_titles, X_mesh], y, nb_epoch=n_epochs, callbacks=callbacks, validation_split=0.2,
                       verbose=verbose, batch_size=32, shuffle=True)

        if save_model:
            self.model.save_weights(self.model_name + '.h5', overwrite=True)

    def test(self, X_abstract, X_titles, X_mesh, y, print_output=False):
        truth = []
        predictions = self.predict_classes([X_abstract, X_titles, X_mesh])

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

    def predict_classes(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def save(self):
        self.model.save_weights(self.model_name)


# Implementation of Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network
