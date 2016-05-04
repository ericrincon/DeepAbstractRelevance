import getopt
import sys
import os
import pandas as pd
import nltk
import numpy as np
from CNN import CNN

from sklearn.cross_validation import KFold
from gensim.models import Word2Vec

from nltk.corpus import stopwords


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['n_feature_maps=', 'epochs=', 'max_words', 'dropout_p='
                                                      'undersample=', 'n_feature_maps=', 'criterion=',
                                                      'optimizer=', 'max_words=', 'layers=',
                                                      'hyperopt=', 'model_name=', 'w2v_path=', 'tacc=', 'use_all_date='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    w2v_path = '/Users/ericrincon/PycharmProjects/Deep-PICO/wikipedia-pubmed-and-PMC-w2v.bin'
    n_feature_maps = 100
    epochs = 50
    criterion = 'categorical_crossentropy'
    optimizer = 'adagrad'
    model_name = 'model'
    w2v_size = 200
    activation = 'elu'
    dense_sizes = [300]
    filter_sizes = [3, 5, 7, 9, 11, 15, 20]
    max_words = 800
    word_vector_size = 200
    using_tacc = False
    undersample = True
    use_all_date = False
    p = .8

    for opt, arg in opts:
        if opt == '--window_size':
            window_size = int(arg)
        elif opt == '--wiki':
            if arg == 0:
                wiki = False
        elif opt == '--dropout_p':
            p = int(arg)
        elif opt == '--epochs':
            epochs = int(arg)
        elif opt == '--layers':
            layer_sizes = arg.split(',')
        elif opt == '--n_feature_maps':
            n_feature_maps = int(arg)
        elif opt == '--n_feature_maps':
            n_feature_maps = int(arg)
        elif opt == '--criterion':
            criterion = arg
        elif opt == '--optimizer':
            optimizer = arg
        elif opt == '--tacc':
            if int(arg) == 1:
                using_tacc = True
        elif opt == '--hyperopt':
            if int(arg) == 1:
                hyperopt = True
        elif opt == '--model_name':
            model_name = arg
        elif opt == 'max_words ':
            max_words = int(arg)
        elif opt == '--w2v_path':
            w2v_path = arg
        elif opt == '--word_vector_size':
            word_vector_size = int(arg)
        elif opt == '--tacc':
            if int(arg) == 1:
                using_tacc = True
        elif opt == '--use_all_data':
            if int(arg) == 1:
                use_all_date = True
        else:
            print("Option {} is not valid!".format(opt))
    if using_tacc:
        nltk.data.path.append('/work/03186/ericr/nltk_data/')
    print('Loading Word2Vec...')
    w2v = Word2Vec.load_word2vec_format(w2v_path, binary=True)
    print('Loaded Word2Vec...')
    X_list = []
    y_list = []

    print('Loading data...')
    if use_all_date:
        X, y = get_data(max_words, word_vector_size, w2v)

        X_list.append(X)
        y_list.append(y)
    else:
        X_list, y_list = get_data_separately(max_words, word_vector_size, w2v)
    print('Loaded data...')

    run_on_all_data(X_list, y_list, model_name, max_words, w2v_size, n_feature_maps, dense_sizes, optimizer, criterion,
                    epochs, filter_sizes, activation, undersample, p)

def run_on_all_data(X_list, y_list, model_name, max_words, w2v_size, n_feature_maps, dense_sizes, optimizer, criterion, epochs,
                  filter_sizes, activation, undersample, p):
    for X, y in zip(X_list, y_list):
        n = X.shape[0]
        kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

        for fold_idx, (train, test) in enumerate(kf):
            X_train, y_train = X[train, :, :, :], y[train, :]
            X_test, y_test = X[test, :, :, :], y[test, :]

            if undersample:
                # Get all the targets that are not relevant i.e., y = 0
                idx_undersample = np.where(y_train[:, 0] == 1)[0]

                # Get all the targets that are relevant i.e., y = 1
                idx_postive = np.where(y_train[:, 1] == 1)[0]

                # Now sample from the no relevant targets
                random_negative_sample = np.random.choice(idx_undersample, idx_postive.shape[0])

                X_train_postive = X_train[idx_postive, :, :, :]

                X_train_negative = X_train[random_negative_sample, :, :, :]

                y_train_postive = y_train[idx_postive, :]
                y_train_negative = y_train[random_negative_sample, :]

                X_train = np.vstack((X_train_postive, X_train_negative))
                y_train = np.vstack((y_train_postive, y_train_negative))


                print("N y = 0: {}".format(random_negative_sample.shape[0]))
                print("N y = 1: {}".format(idx_postive.shape[0]))

            _X = [X_train, X_test]
            _y = [y_train, y_test]

            run_model(_X, _y, model_name, fold_idx, max_words, w2v_size, n_feature_maps, dense_sizes, optimizer, criterion, epochs,
                      filter_sizes, activation, p)


def run_model(X, y, model_name, fold_idx, max_words, w2v_size, n_feature_maps, dense_sizes, optimizer, criterion, epochs,
              filter_sizes, activation, p):
    X_train, X_test = X
    y_train, y_test = y
    print("X_train shape: {}".format(X_train.shape))
    temp_model_name = model_name + '_fold_{}.h5'.format(fold_idx + 1)
    cnn = CNN(n_classes=2, max_words=max_words, w2v_size=w2v_size, vocab_size=1000, use_embedding=False,
              filter_sizes=filter_sizes, n_filters=n_feature_maps, dense_layer_sizes=dense_sizes.copy(),
              name=temp_model_name, activation_function=activation, dropout_p=p)

    cnn.train(X_train, y_train, n_epochs=epochs, optim_algo=optimizer, criterion=criterion, verbose=1)
    accuracy, f1_score, precision, auc, recall = cnn.test(X_test, y_test, print_output=True)

    print("Accuracy: {}".format(accuracy))
    print("F1: {}".format(f1_score))
    print("Precision: {}".format(precision))
    print("AUC: {}".format(auc))
    print("Recall: {}".format(recall))

    cnn.save()


def get_all_files(path):
    file_paths = []

    for path, subdirs, files in os.walk(path):
        for name in files:

            # Make sure hidden files do not make into the list
            if name[0] == '.':
                continue
            file_paths.append(os.path.join(path, name))
    return file_paths

def get_data_separately(max_words, word_vector_size, w2v):
    file_paths = get_all_files('Data')
    X_list, y_list = [], []
    total_words = 0.0
    n_abstracts = 0.0

    for file in file_paths:
        abstract_text, abstract_labels = extract_abstract_and_labels(file)

        abstracts_as_words = []
        labels = []
        for i in range(abstract_text.shape[0]):
            abstract = abstract_text.iloc[i]

            if abstract == 'MISSING':
                continue
            else:
                abstract_as_words = nltk.word_tokenize(abstract)

                total_words += len(abstracts_as_words)
                n_abstracts += 1.0

                abstracts_as_words.append(abstract_as_words)
                labels.append(abstract_labels.iloc[i])

        X = np.zeros((len(abstracts_as_words), 1, max_words, word_vector_size))
        y = np.zeros((len(labels), 2))

        for i, (abstract, label) in enumerate(zip(abstracts_as_words, labels)):
            word_matrix = abstract_to_w2v(abstract, max_words, w2v, word_vector_size)
            X[i, 0, :, :] = word_matrix
            if label == -1:
                label = 0
            y[i, label] = 1

        X_list.append(X)
        y_list.append(y)
    average_word_per_abstract = float(total_words)/float(n_abstracts)

    print('Average word per abstract: {}'.format(average_word_per_abstract))

    return X_list, y_list


def extract_abstract_and_labels(file):
    data_frame = pd.read_csv(file)

    abstract_text = data_frame.iloc[:, 4]
    labels = data_frame.iloc[:, 6]

    return abstract_text, labels


def get_data(max_words, word_vector_size, w2v):

    file_paths = get_all_files('Data')

    abstract_text_df = pd.DataFrame()
    labels_df = pd.DataFrame()

    for file in file_paths:
        abstract_text, labels = extract_abstract_and_labels(file)

        abstract_text_df = pd.concat((abstract_text_df, abstract_text))
        labels_df = pd.concat((labels_df, labels))

    abstracts_as_words = []
    labels = []
    total_words = 0

    for i in range(abstract_text_df.shape[0]):
        abstract = abstract_text_df.iloc[i, :][0]

        if abstract == 'MISSING':
            continue
        else:

            abstract_as_words = nltk.word_tokenize(abstract)
            abstracts_as_words.append(abstract_as_words)
            labels.append(labels_df.iloc[i])

            total_words += len(abstracts_as_words)
    X = np.zeros((len(abstracts_as_words), 1, max_words , word_vector_size))
    y = np.zeros((len(labels), 2))

    print(total_words/len(abstracts_as_words))

    for i, (abstract, label) in enumerate(zip(abstracts_as_words, labels)):
        word_matrix = abstract_to_w2v(abstract, max_words, w2v, word_vector_size)
        X[i, 0, :, :] = word_matrix
        label = label[0]
        if label == -1:
            label = 0
        y[i, label] = 1

    return X, y


def abstract_to_w2v(abstract_as_words, max_words, w2v, word_vector_size, remove_stop_words=True):
    if remove_stop_words:
        stop = stopwords.words('english')
    i = 0
    word_matrix = np.zeros((max_words, word_vector_size))

    for word in abstract_as_words:
        if word in stop and remove_stop_words:
            continue

        if i == max_words - 1:
            break
        try:
            word_vector = w2v[word]
            word_matrix[i, word_vector] = word_vector
            i += 1
        except:
            continue



    return word_matrix


if __name__ == '__main__':
    main()
