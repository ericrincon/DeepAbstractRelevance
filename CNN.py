from keras.layers import Input, Embedding, merge, Dense

from keras.models import Model, Sequential

from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution1D, MaxPooling1D
from keras.layers.core import Flatten, Activation, Dropout, Reshape

from keras.optimizers import Adam, SGD, Adagrad

from keras.layers.advanced_activations import ELU
from keras.callbacks import EarlyStopping

from keras.constraints import maxnorm

from BatchGenerator import ImbalancedBBG, StanderedBG, DomainBG

import sklearn.metrics as metrics
import numpy as np
import DataLoader as dl




class NLPCNN:
    def __init__(self):
        self.model = build_model()
    def train(self, X, y, model, batch_generator, n_epochs=50, optim_algo='adam',
              criterion='categorical_crossentropy', save_model=True, verbose=2,
              plot=True, batch_size=64,):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        loss_train_history = []
        loss_val_history = []
        batch_history = {'f1': [], 'recall': [], 'precision': []}

        for epoch in range(1, n_epochs + 1):
            batch_f1_history = []
            batch_precision_history = []
            batch_recall_history = []

            for X, y in batch_generator.next_batch():
                history = self.model.fit(X, y, nb_epoch=1, batch_size=batch_size,
                                         validation_split=0.2, verbose=0)

                val_loss, loss = history.history['val_loss'][0], history.history['loss'][0]

                loss_train_history.append(loss)
                loss_val_history.append(val_loss)

                truth = self.model.validation_data[3]
                truth = dl.onehot2list(truth)
                batch_prediction = self.predict_classes(self.model.validation_data[0:3])

                batch_f1 = metrics.f1_score(truth, batch_prediction)
                batch_recall = metrics.recall_score(truth, batch_prediction)
                batch_precision = metrics.precision_score(truth, batch_prediction)

                batch_f1_history.append(batch_f1)
                batch_recall_history.append(batch_recall)
                batch_precision_history.append(batch_precision)

            batch_history['f1'].append(batch_f1_history)
            batch_history['recall'].append(batch_recall_history)
            batch_history['precision'].append(batch_precision_history)

            print('Epoch: {} | Train loss: {} | Valid loss: {}'.format(epoch, loss, val_loss))
            print("Epoch Metrics | F1: {} | Recall {} | Precision: {}".format(np.mean(batch_history['f1'][epoch - 1]),
                                                                              np.mean(batch_history['recall'][epoch - 1]),
                                                                              np.mean(batch_history['precision'][epoch - 1])))
            a_max = np.argmax(batch_history['f1'][epoch - 1])
            print("Best F1 at Epoch {} Minibatch {}: {}\n".format(epoch, a_max, batch_history['f1'][epoch-1][a_max]))


        if save_model:
            self.model.save_weights(self.model_name + '.h5', overwrite=True)



class DomainCNN:
    def __init__(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_filters,
                 dense_layer_sizes, name, activation_function, dropout_p):
        self.model = self.build_model(n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes,
                                      n_filters, dense_layer_sizes, activation_function, dropout_p)
        self.model_name = name
        self.conv_layer = None

    def build_model(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_filters,
                    dense_layer_sizes, activation_function, dropout_p):

        # From the paper http://arxiv.org/pdf/1511.07289v1.pdf
        # Supposed to perform better but lets see about that
        if activation_function == 'elu':
            activation_function = ELU(alpha=1.0)
        else:
            activation_function = Activation(activation_function)

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
            activation_layer = Activation(activation_function)(convolution_layer)

            max_pooling = MaxPooling1D(pool_length=(max_words - filter_size + 1))(activation_layer)
            flattened_layer = Flatten()(max_pooling)

            conv_layers.append(flattened_layer)

        # Domain embedding
        n_domains = 9
        domain_input = Input(shape=(1, ), dtype='int32')
        domain_emebedding = Embedding(n_domains, 50, input_length=1)(domain_input)
        domain_node = Flatten()(domain_emebedding)

        merge_layer = merge(conv_layers, mode='concat')

      #  concat_layer = merge([merge_layer, domain_node], mode='concat')


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

        model = Model(input=[w2v_input, domain_input], output=softmax_layer)

        return model

    def train(self, X, y, n_epochs, optim_algo='adagrad', criterion='categorical_crossentropy', save_model=True,
              verbose=2, plot=True, batch_size=64, fold_idxs=None):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        batch_generator = DomainBG(X, y, batch_size, pos_p=.7, fold_indices=fold_idxs)


        loss_train_history = []
        loss_val_history = []
        batch_history = {'f1': [], 'recall': [], 'precision': []}

        for epoch in range(1, n_epochs + 1):
            batch_f1_history = []
            batch_precision_history = []
            batch_recall_history = []

            for X, y in batch_generator.next_batch():
                history = self.model.fit(X, y, nb_epoch=1, batch_size=batch_size,
                                         validation_split=0.2, verbose=0)

                val_loss, loss = history.history['val_loss'][0], history.history['loss'][0]

                loss_train_history.append(loss)
                loss_val_history.append(val_loss)

                truth = self.model.validation_data[2]
                truth = dl.onehot2list(truth)
                batch_prediction = self.predict_classes(self.model.validation_data[0:2])

                batch_f1 = metrics.f1_score(truth, batch_prediction)
                batch_recall = metrics.recall_score(truth, batch_prediction)
                batch_precision = metrics.precision_score(truth, batch_prediction)

                batch_f1_history.append(batch_f1)
                batch_recall_history.append(batch_recall)
                batch_precision_history.append(batch_precision)

            batch_history['f1'].append(batch_f1_history)
            batch_history['recall'].append(batch_recall_history)
            batch_history['precision'].append(batch_precision_history)

            print('Epoch: {} | Train loss: {} | Valid loss: {}'.format(epoch, loss, val_loss))
            print("Epoch Metrics | F1: {} | Recall {} | Precision: {}".format(np.mean(batch_history['f1'][epoch - 1]),
                                                                              np.mean(batch_history['recall'][epoch - 1]),
                                                                              np.mean(batch_history['precision'][epoch - 1])))
            a_max = np.argmax(batch_history['f1'][epoch - 1])
            print("Best F1 at Epoch {} Minibatch {}: {}\n".format(epoch, a_max, batch_history['f1'][epoch-1][a_max]))
        if save_model:
            self.model.save_weights(self.model_name + '.h5', overwrite=True)

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

    def predict_classes(self, x):
        predictions = self.model.predict(x)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

# Implementation of Convolutional Neural Networks for Sentence Classification
# Paper: http://arxiv.org/abs/1408.5882

class SentenceCNN:
    def __init__(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_filters,
                 dense_layer_sizes, name, activation_function, dropout_p):
        self.model = self.build_model(n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes,
                                      n_filters, dense_layer_sizes, activation_function, dropout_p)
        self.model_name = name
        self.conv_layer = None

    def build_model(self, n_classes, max_words, w2v_size, vocab_size, use_embedding, filter_sizes, n_filters,
                    dense_layer_sizes, activation_function, dropout_p):

        # From the paper http://arxiv.org/pdf/1511.07289v1.pdf
        # Supposed to perform better but lets see about that
        if activation_function == 'elu':
            activation_function = ELU(alpha=1.0)
        else:
            activation_function = Activation(activation_function)

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
            activation_layer = Activation(activation_function)(convolution_layer)

            max_pooling = MaxPooling1D(pool_length=(max_words - filter_size + 1))(activation_layer)
            flattened_layer = Flatten()(max_pooling)

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

    def train(self, X, y, n_epochs, optim_algo='adam', criterion='categorical_crossentropy', save_model=True,
              verbose=2, plot=True, batch_size=64, fold_idxs=None):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        if fold_idxs is not None:
            batch_generator = StanderedBG(X, y, batch_size=batch_size, fold_indices=fold_idxs)
        else:
            batch_generator = StanderedBG(X, y, batch_size=batch_size)


        loss_train_history = []
        loss_val_history = []
        batch_history = {'accuracy': []}

        for epoch in range(1, n_epochs + 1):
            batch_accuracy_history = []

            for X, y in batch_generator.next_batch():
                history = self.model.fit(X, y, nb_epoch=1, batch_size=batch_size,
                                         validation_split=0.2, verbose=0)

                val_loss, loss = history.history['val_loss'][0], history.history['loss'][0]

                loss_train_history.append(loss)
                loss_val_history.append(val_loss)

                truth = self.model.validation_data[1]
                truth = dl.onehot2list(truth)
                batch_prediction = self.predict_classes(self.model.validation_data[0])
                accuracy = metrics.accuracy_score(truth, batch_prediction)
                batch_accuracy_history.append(accuracy)

            batch_history['accuracy'].append(batch_accuracy_history)

            print('Epoch: {} | Train loss: {} | Valid loss: {}'.format(epoch, loss, val_loss))
            print("Epoch Metrics | Accuracy: {}".format(np.mean(batch_history['accuracy'][epoch-1])))

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
        self.conv_output = None
        self.input = None
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

            convolution_layer = Convolution1D(n_feature_maps, filter_size, input_shape=(max_words, w2v_size))(conv_input)
            activation_layer = Activation(activation_function)(convolution_layer)
            max_layer = MaxPooling1D(pool_length=max_words - filter_size + 1)(activation_layer)

            flattened_layer = Flatten()(max_layer)

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

        self.input = [abstract_input, title_input, mesh_input]

        merge_layer = merge([abstract_node, title_node, mesh_node], mode='concat')

        self.conv_output = merge_layer

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

        model = Model(input=self.input, output=softmax_layer)

        return model

    def train(self, X_abstract, X_titles, X_mesh, y, n_epochs, optim_algo='adam',
              criterion='categorical_crossentropy', save_model=True, verbose=2,
              plot=True, tensorBoard_path='', patience=20, use_tensorboard=False,
              batch_size=64):

        if optim_algo == 'adam':
            optim_algo = Adam()
        elif optim_algo == 'sgd':
            optim_algo = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        elif optim_algo == 'adagrad':
            optim_algo = Adagrad()

        self.model.compile(optimizer=optim_algo, loss=criterion)

        batch_generator = ImbalancedBBG([X_abstract, X_titles, X_mesh], y, batch_size, pos_p=.8)


        loss_train_history = []
        loss_val_history = []
        batch_history = {'f1': [], 'recall': [], 'precision': []}

        for epoch in range(1, n_epochs + 1):
            batch_f1_history = []
            batch_precision_history = []
            batch_recall_history = []

            for X, y in batch_generator.next_batch():
                history = self.model.fit(X, y, nb_epoch=1, batch_size=batch_size,
                                         validation_split=0.2, verbose=0)

                val_loss, loss = history.history['val_loss'][0], history.history['loss'][0]

                loss_train_history.append(loss)
                loss_val_history.append(val_loss)

                truth = self.model.validation_data[3]
                truth = dl.onehot2list(truth)
                batch_prediction = self.predict_classes(self.model.validation_data[0:3])

                batch_f1 = metrics.f1_score(truth, batch_prediction)
                batch_recall = metrics.recall_score(truth, batch_prediction)
                batch_precision = metrics.precision_score(truth, batch_prediction)

                batch_f1_history.append(batch_f1)
                batch_recall_history.append(batch_recall)
                batch_precision_history.append(batch_precision)

            batch_history['f1'].append(batch_f1_history)
            batch_history['recall'].append(batch_recall_history)
            batch_history['precision'].append(batch_precision_history)

            print('Epoch: {} | Train loss: {} | Valid loss: {}'.format(epoch, loss, val_loss))
            print("Epoch Metrics | F1: {} | Recall {} | Precision: {}".format(np.mean(batch_history['f1'][epoch - 1]),
                                                                              np.mean(batch_history['recall'][epoch - 1]),
                                                                              np.mean(batch_history['precision'][epoch - 1])))
            a_max = np.argmax(batch_history['f1'][epoch - 1])
            print("Best F1 at Epoch {} Minibatch {}: {}\n".format(epoch, a_max, batch_history['f1'][epoch-1][a_max]))


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

    def output_learned_features(self, X):
        assert (self.conv_output and self.input) is not None, 'Build the model first!'

        conv_layer_model = Model(input=self.input, output=self.conv_output)
        learned_features = conv_layer_model.predict(X)

        return learned_features


# Implementation of Modelling, Visualising and Summarising Documents with a Single Convolutional Neural Network
class DCNN:
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

        conv_layers = []

        for filter_size in filter_sizes:

            if use_embedding:
                convolution_layer = Convolution1D(n_feature_maps, filter_size, input_shape=(max_words, w2v_size))(conv_input)
                max_layer = MaxPooling1D(pool_length=max_words - filter_size + 1)(convolution_layer)
            else:
                convolution_layer = Convolution1D(n_feature_maps, filter_size, input_shape=(max_words, w2v_size))(conv_input)

                max_layer = MaxPooling1D(pool_length=(max_words - filter_size + 1))(convolution_layer)

            activation_layer = Activation(activation_function)(max_layer)

            doc_convolution_layer = Convolution1D(n_feature_maps, filter_size, )

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

        sentence_inputs = []

        for n_sentence in range(max_sentences):
            sentence_node, sentence_input = self.build_conv_node(n_feature_maps['text'], max_words['text'], w2v_size,
                                                             activation_function, filter_sizes['text'], vocab_size,
            sentence_inputs.append((abstract_node, abstract_input)))



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

    def predict_classes(self, X):
        predictions = self.model.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)

        return predicted_classes

    def save(self):
        self.model.save_weights(self.model_name)


