import getopt
import sys
import nltk
import numpy as np
import DataLoader

from CNN import AbstractCNN

from sklearn.cross_validation import KFold
from gensim.models import Word2Vec

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['n_feature_maps=', 'epochs=', 'max_words=', 'dropout_p=',
                                                      'undersample=', 'n_feature_maps=', 'criterion=',
                                                      'optimizer=', 'max_words=', 'layers=',
                                                      'hyperopt=', 'model_name=', 'w2v_path=', 'tacc=', 'use_all_date=',
                                                      'patience=', 'filter_sizes=', 'model_type='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    w2v_path = '/Users/ericrincon/PycharmProjects/Deep-PICO/wikipedia-pubmed-and-PMC-w2v.bin'
    epochs = 50
    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    model_name = 'model'
    w2v_size = 200
    activation = 'relu'
    dense_sizes = [400, 400]
    max_words = {'text': 220, 'mesh': 40, 'title': 14}

    filter_sizes = {'text': [2, 3, 5, 7, 9, 12, 13, 17],
                    'mesh': [2, 3, 5],
                    'title': [2, 3, 5, 7, 9]}
    n_feature_maps = {'text': 100, 'mesh': 50, 'title': 50}
    word_vector_size = 200
    using_tacc = False
    undersample = True
    use_all_date = False
    patience = 20
    p = .7

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
        elif opt == '--max_words':
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
        elif opt == '--patience':
            patience = int(arg)

        elif opt == '--undersample':
            if int(arg) == 0:
                undersample = False
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
        X, y = DataLoader.get_data(max_words, word_vector_size, w2v)

        X_list.append(X)
        y_list.append(y)
    else:
        X_list, y_list = DataLoader.get_data_separately(max_words, word_vector_size, w2v, use_abstract_cnn=True)

    print('Loaded data...')

    for X, y in zip(X_list, y_list):
        X_abstract, X_titles, X_mesh = X
        n = X_abstract.shape[0]
        kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

        for fold_idx, (train, test) in enumerate(kf):
            X_abstract_train = X_abstract[train, :, :, :]
            X_titles_train = X_titles[train, :, :, :]
            X_mesh_train = X_mesh[train, :, :, :]
            y_train = y[train, :]

            X_abstract_test = X_abstract[test, :, :, :]
            X_titles_test = X_titles[test, :, :, :]
            X_mesh_test = X_mesh[test, :, :, :]
            y_test = y[test, :]

            if undersample:
                # Get all the targets that are not relevant i.e., y = 0
                idx_undersample = np.where(y_train[:, 0] == 1)[0]

                # Get all the targets that are relevant i.e., y = 1
                idx_postive = np.where(y_train[:, 1] == 1)[0]

                # Now sample from the no relevant targets
                random_negative_sample = np.random.choice(idx_undersample, idx_postive.shape[0])

                X_abstract_train_positive = X_abstract_train[idx_postive, :, :, :]
                X_titles_train_positive = X_titles_train[idx_postive, :, :, :]
                X_mesh_train_positive = X_mesh_train[idx_postive, :, :, :]

                X_abstract_train_negative = X_abstract_train[random_negative_sample, :, :, :]
                X_titles_train_negative = X_titles_train[random_negative_sample, :, :, :]
                X_mesh_train_negative = X_mesh_train[random_negative_sample, :, :, :]

                X_abstract_train = np.vstack((X_abstract_train_positive, X_abstract_train_negative))
                X_titles_train = np.vstack((X_titles_train_positive, X_titles_train_negative))
                X_mesh_train = np.vstack((X_mesh_train_positive, X_mesh_train_negative))

                y_train_positive = y_train[idx_postive, :]
                y_train_negative = y_train[random_negative_sample, :]
                y_train = np.vstack((y_train_positive, y_train_negative))

                print("N y = 0: {}".format(random_negative_sample.shape[0]))
                print("N y = 1: {}".format(idx_postive.shape[0]))

            temp_model_name = model_name + '_fold_{}.h5'.format(fold_idx + 1)

            cnn = AbstractCNN(n_classes=2,  max_words=max_words, w2v_size=w2v_size, vocab_size=1000, use_embedding=False,
                              filter_sizes=filter_sizes, n_feature_maps=n_feature_maps, dense_layer_sizes=dense_sizes.copy(),
                              name=temp_model_name, activation_function=activation, dropout_p=p)
            cnn.train(X_abstract_train, X_titles_train, X_mesh_train, y_train, n_epochs=epochs, optim_algo=optimizer,
                      criterion=criterion, verbose=1, patience=patience)
            accuracy, f1_score, precision, auc, recall = cnn.test(X_abstract_test, X_titles_test, X_mesh_test, y_test,
                                                                  print_output=True)

            print("Accuracy: {}".format(accuracy))
            print("F1: {}".format(f1_score))
            print("Precision: {}".format(precision))
            print("AUC: {}".format(auc))
            print("Recall: {}".format(recall))

            cnn.save()
if __name__ == '__main__':
    main()
