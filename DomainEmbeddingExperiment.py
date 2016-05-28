import getopt
import sys
import nltk
import h5py
import pickle
import DataLoader
import numpy as np

from CNN import DomainCNN
from sklearn.cross_validation import KFold


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['n_feature_maps=', 'epochs=', 'max_words=', 'dropout_p=',
                                                      'criterion=', 'optimizer=', 'max_words=', 'layers=',
                                                      'experiment_name=', 'w2v_path=', 'tacc=',
                                                      'baseline=', 'save_model='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    epochs = 50

    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    experiment_name = 'abstractCNN'
    activation = 'relu'
    word_vector_size = 200
    using_tacc = False
    p = .5
    save_model = True
    print_output = False
    dense_sizes = [400, 400]
    max_words = {'text': 270, 'mesh': 50, 'title': 17}

    filter_sizes = {'text': [2, 3, 4, 5],
                    'mesh': [2, 3, 4, 5],
                    'title': [2, 3, 4, 5]}
    n_feature_maps = {'text': 100, 'mesh': 50, 'title': 50}
    for opt, arg in opts:
        if opt == '--save_model':
            if int(arg) == 0:
                save_model = False
            elif int(arg) ==  1:
                save_model = True

        elif opt == '--verbose':
            verbose = int(arg)
        elif opt == '--use_embedding':
            if int(arg) == 0:
                use_embedding = False
        elif opt == '--dropout_p':
            p = float(arg)
        elif opt == '--epochs':
            epochs = int(arg)
        elif opt == '--layers':
            layer_sizes = arg.split(',')
        elif opt == '--n_feature_maps':
            n_feature_maps = int(arg)
        elif opt == '--criterion':
            criterion = arg
        elif opt == '--optimizer':
            optimizer = arg
        elif opt == '--tacc':
            if int(arg) == 1:
                using_tacc = True
        elif opt == '--experiment_name':
            experiment_name = arg
        elif opt == '--max_words':
            max_words = int(arg)
        elif opt == '--word_vector_size':
            word_vector_size = int(arg)
        elif opt == '--patience':
            patience = int(arg)
        elif opt == '--tacc':
            if int(arg) == 1:
                using_tacc = True

        else:
            print("Option {} is not valid!".format(opt))


    if using_tacc:
        nltk.data.path.append('/work/03186/ericr/nltk_data/')
    print('Loading data...')

    data = h5py.File('all_domains.hdf5')
    X_text = data['X_text']
    X_title = data['X_title']
    X_mesh = data['X_mesh']
    X_embedding = data['de']
    X = [X_text, X_title, X_mesh, X_embedding]
    y = data['y']
    domain2idxs = pickle.load(open('domain2idxs.p', 'rb'))
    print('Loaded data...')
    print(X_text.shape)
    domain_folds = []
    domain_names = DataLoader.get_all_files('Data')
    domain_names = [domain_name.split('/')[-1].split('.')[0] for domain_name in domain_names]

    folds = [[[], []] for i in range(9)]

    X_list, y_list = DataLoader.load_datasets_from_h5py('DataProcessed', load_mesh_title=True, load_as_np=False)

    for i in range(9):
        kf = KFold(X_list[i]['text'].shape[0], random_state=1337, shuffle=True, n_folds=5)
        domain_split = []

        for fold_i, (train, test) in enumerate(kf):
            train_fold = []
            test_fold = []

            for train_idx in train:
                train_fold.append(domain2idxs[i][train_idx])

            for test_idx in test:
                test_fold.append(domain2idxs[i][test_idx])

            domain_split.append((train, test))
            folds[fold_i][0].extend(train_fold)
            folds[fold_i][1].extend(test_fold)

        domain_folds.append(domain_split)

    for fold_idx, (train, test) in enumerate(folds):
        model_name = experiment_name + str(fold_idx)
        cnn = DomainCNN(n_classes=2,  max_words=max_words, w2v_size=200, vocab_size=1000, use_embedding=False,
                        filter_sizes=filter_sizes, n_filters=n_feature_maps, dense_layer_sizes=dense_sizes,
                        name=model_name, activation_function=activation, dropout_p=p)

        cnn.train(X, y, n_epochs=epochs, optim_algo=optimizer, criterion=criterion,
                  save_model=save_model, fold_idxs=train)

        accuracy, f1_score, precision, auc, recall = cnn.test(X, y, print_output, indices=test)

        print('Performance on all data')
        print("Accuracy: {}".format(accuracy))
        print("F1 score: {}".format(f1_score))
        print("AUC: {}".format(auc))
        print("Recall: {}\n".format(recall))

        for domain_i, (domain_name, domain_fold, _X, _y) in enumerate(zip(domain_names, domain_folds, X_list, y_list)):
            _, test_domain = domain_fold[fold_idx]
            X_domain = np.empty((_X['text'].shape[0], 1))
            X_domain[:, 0] = domain_i

            X_test = [_X['text'], _X['title'], _X['mesh'], X_domain]
            accuracy, f1_score, precision, auc, recall = cnn.test(X_test, _y, print_output, indices=test_domain)

            print('Performance on {}'.format(domain_name))
            print("Accuracy: {}".format(accuracy))
            print("F1 score: {}".format(f1_score))
            print("AUC: {}".format(auc))
            print("Recall: {}\n".format(recall))

        print('-------------------------------------------------------')



if __name__ == '__main__':
    main()
