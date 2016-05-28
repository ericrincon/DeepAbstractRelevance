import getopt
import sys
import nltk
import h5py

from CNN import DomainCNN
from sklearn.cross_validation import KFold


def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['n_feature_maps=', 'epochs=', 'max_words=', 'dropout_p=',
                                                      'criterion=', 'optimizer=', 'max_words=', 'layers=',
                                                      'hyperopt=', 'experiment_name=', 'w2v_path=', 'tacc=',
                                                      'tacc=', 'save_model='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    epochs = 50
    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    experiment_name = 'abstractCNN'
    activation = 'relu'
    dense_sizes = [400]
    max_words = 270
    filter_sizes = [2, 3, 4, 5]
    n_feature_maps = 100
    word_vector_size = 200
    using_tacc = False
    patience = 50
    p = .5
    verbose = 0
    save_model = False
    print_output = True

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
    X_text = data['X']
    X_embedding = data['de']
    y = data['y']
    domain_embeddings = []

    print('Loaded data...')
    print(X_text.shape)
    n = X_text.shape[0]
    kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

    for fold_idx, (train, test) in enumerate(kf):
        cnn = DomainCNN(n_classes=2,  max_words=max_words, w2v_size=200, vocab_size=1000, use_embedding=False,
                              filter_sizes=filter_sizes, n_filters=n_feature_maps, dense_layer_sizes=dense_sizes.copy(),
                              name='model_all', activation_function=activation, dropout_p=p)

        cnn.train([X_text, X_embedding], y, n_epochs=epochs, optim_algo=optimizer, criterion=criterion,
                  verbose=verbose, save_model=save_model, fold_idxs=train)
        X_test, y_test = [X_text[test, :, :], X_embedding[test, :]], y[test, :]
        accuracy, f1_score, precision, auc, recall = cnn.test(X_test, y_test, print_output)

        print("Accuracy: {}".format(accuracy))
        print("F1 score: {}".format(f1_score))
        print("AUC: {}".format(auc))
        print("Recall: {}".format(recall))

if __name__ == '__main__':
    main()
