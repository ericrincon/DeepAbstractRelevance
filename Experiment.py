import getopt
import sys
import nltk
import numpy as np
import DataLoader

from CNN import CNN

from sklearn.cross_validation import KFold
from gensim.models import Word2Vec



def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['n_feature_maps=', 'epochs=', 'max_words=', 'dropout_p=',
                                                      'undersample=', 'n_feature_maps=', 'criterion=',
                                                      'optimizer=', 'max_words=', 'layers=',
                                                      'hyperopt=', 'experiment_name=', 'w2v_path=', 'tacc=', 'use_all_date=',
                                                      'patience=', 'filter_sizes=', 'model_type=', 'verbose='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    w2v_path = '/Users/ericrincon/PycharmProjects/Deep-PICO/wikipedia-pubmed-and-PMC-w2v.bin'

    n_feature_maps = 30
    epochs = 50
    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    experiment_name = 'cnn'
    w2v_size = 200
    activation = 'relu'
    dense_sizes = [400, 400]
    filter_sizes = [2, 3, 4, 5]
    max_words = 280
    word_vector_size = 200
    using_tacc = False
    undersample = True
    use_all_date = False
    patience = 20
    p = .5
    model_type = 'cnn'
    verbose = 1

    for opt, arg in opts:
        if opt == '--window_size':
            window_size = int(arg)
        elif opt == '--verbose':
            verbose = int(arg)
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
        elif opt == '--model_type':
            model_type = arg
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
        X_list, y_list = DataLoader.get_data_separately({'text': max_words}, word_vector_size, w2v,
                                                          use_abstract_cnn=False, preprocess_text=False)

    print('Loaded data...')

    dataset_names = DataLoader.get_all_files('Data')
    dataset_names = [name.split('/')[1].split('.')[0] for name in dataset_names]

    for i, (X, y) in enumerate(zip(X_list, y_list)):
        print("Dataset: {}".format(dataset_names[i]))

        model_name = dataset_names[i]

        n = X.shape[0]
        kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

        fold_accuracies = []
        fold_recalls = []
        fold_precisions =[]
        fold_aucs = []
        fold_f1s = []

        for fold_idx, (train, test) in enumerate(kf):
            X_train = X[train]
            y_train = y[train, :]

            X_test = X[test]
            y_test = y[test, :]

            if undersample:
                # Get all the targets that are not relevant i.e., y = 0
                idx_undersample = np.where(y_train[:, 0] == 1)[0]

                # Get all the targets that are relevant i.e., y = 1
                idx_positive = np.where(y_train[:, 1] == 1)[0]

                print("idx_undersample length: {}".format(idx_undersample.shape[0]))

                # Now sample from the no relevant targets
                random_negative_sample = np.random.choice(idx_undersample, idx_positive.shape[0])

                X_train_positive = X_train[idx_positive, :, :]
                X_train_negative = X_train[random_negative_sample, :, :]

                X_train = np.vstack((X_train_positive, X_train_negative))

                y_train_positive = y_train[idx_positive, :]
                y_train_negative = y_train[random_negative_sample, :]
                y_train = np.vstack((y_train_positive, y_train_negative))

                print("N y = 0: {}".format(random_negative_sample.shape[0]))
                print("N y = 1: {}".format(idx_positive.shape[0]))

            temp_model_name = experiment_name + '_'+ model_name + '_fold_{}'.format(fold_idx + 1)

            print("X_train shape: {}".format(len(X_train)))

            cnn = CNN(n_classes=2, max_words=max_words, w2v_size=w2v_size, vocab_size=1000, use_embedding=False,
                      filter_sizes=filter_sizes, n_filters=n_feature_maps, dense_layer_sizes=dense_sizes.copy(),
                      name=temp_model_name, activation_function=activation, dropout_p=p)

            cnn.train(X_train, y_train, n_epochs=epochs, optim_algo=optimizer, criterion=criterion, verbose=verbose,
                      patience=patience, save_model=True)
            accuracy, f1_score, precision, auc, recall = cnn.test(X_test, y_test, print_output=True)


            print("Accuracy: {}".format(accuracy))
            print("F1: {}".format(f1_score))
            print("Precision: {}".format(precision))
            print("AUC: {}".format(auc))
            print("Recall: {}".format(recall))
            print('\n')

            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_aucs.append(auc)
            fold_f1s.append(f1_score)

        average_accuracy = np.mean(fold_accuracies)
        average_precision = np.mean(fold_precisions)
        average_recall = np.mean(fold_recalls)
        average_auc = np.mean(fold_aucs)
        average_f1 = np.mean(fold_f1s)

        print("Fold Average Accuracy: {}".format(average_accuracy))
        print("Fold Average F1: {}".format(average_f1))
        print("Fold Average Precision: {}".format(average_precision))
        print("Fold Average AUC: {}".format(average_auc))
        print("Fold Average Recall: {}".format(average_recall))
        print('\n')


if __name__ == '__main__':
    main()
