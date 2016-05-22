import getopt
import sys
import nltk
import numpy as np
import DataLoader
import seaborn as sns

from CNN import AbstractCNN

from SVM import SVM

from sklearn.cross_validation import KFold
from gensim.models import Word2Vec

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['n_feature_maps=', 'epochs=', 'max_words=', 'dropout_p=',
                                                      'undersample=', 'n_feature_maps=', 'criterion=',
                                                      'optimizer=', 'max_words=', 'layers=',
                                                      'hyperopt=', 'experiment_name=', 'w2v_path=', 'tacc=', 'use_all_date=',
                                                      'patience=', 'filter_sizes=', 'model_type=', 'use_embedding=',
                                                      'verbose=', 'tacc=', 'pretrain=', 'undersample_all=', 'save_model=',
                                                      'transfer_learning='])
    except getopt.GetoptError as error:
        print(error)
        sys.exit(2)

    w2v_path = '/Users/ericrincon/PycharmProjects/Deep-PICO/wikipedia-pubmed-and-PMC-w2v.bin'
    epochs = 1
    criterion = 'categorical_crossentropy'
    optimizer = 'adam'
    experiment_name = 'abstractCNN'
    w2v_size = 200
    activation = 'relu'
    dense_sizes = [400, 400]
    max_words = {'text': 270, 'mesh': 50, 'title': 17}

    filter_sizes = {'text': [2, 3, 4, 5],
                    'mesh': [2, 3, 4, 5],
                    'title': [2, 3, 4, 5]}
    n_feature_maps = {'text': 30, 'mesh': 30, 'title': 30}
    word_vector_size = 200
    using_tacc = False
    undersample = True
    use_embedding = False
    embedding = None
    use_all_date = False
    patience = 50
    p = 0
    verbose = 1
    pretrain = False
    undersample_all = True
    filter_small_data = True
    save_model = False
    load_data_from_scratch = False
    print_output = True
    transfer_learning = True

    for opt, arg in opts:
        if opt == '--save_model':
            if int(arg) == 0:
                save_model = False
            elif int(arg) ==  1:
                save_model = True
        elif opt == '--transfer_learning':
            if int(arg) == 1:
                transfer_learning = True
            elif int(arg) == 0:
                transfer_learning = False
        elif opt == '--undersample_all':
            if int(arg) == 0:
                undersample_all = False
            elif int(arg) == 1:
                undersample_all = True
        elif opt == '--pretrain':
            if int(arg) == 0:
                pretrain = False
            elif int(arg) == 1:
                pretrain = True
            else:
                print("Invalid input")

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
        elif opt == '--experiment_name':
            experiment_name = arg
        elif opt == '--max_words':
            max_words = int(arg)
        elif opt == '--w2v_path':
            w2v_path = arg
        elif opt == '--word_vector_size':
            word_vector_size = int(arg)
        elif opt == '--use_all_data':
            if int(arg) == 1:
                use_all_date = True
        elif opt == '--patience':
            patience = int(arg)

        elif opt == '--undersample':
            if int(arg) == 0:
                undersample = False
            elif int(arg) == 1:
                undersample = True
        elif opt == '--tacc':
            if int(arg) == 1:
                using_tacc = True

        else:
            print("Option {} is not valid!".format(opt))


    if using_tacc:
        nltk.data.path.append('/work/03186/ericr/nltk_data/')
    print('Loading data...')

    if load_data_from_scratch:

        print('Loading Word2Vec...')
        w2v = Word2Vec.load_word2vec_format(w2v_path, binary=True)
        print('Loaded Word2Vec...')
        X_list = []
        y_list = []

        if use_embedding:

            X_list, y_list, embedding_list = DataLoader.get_data_as_seq(w2v, w2v_size, max_words)

        else:
            X_list, y_list = DataLoader.get_data_separately(max_words, word_vector_size,
                                                            w2v, use_abstract_cnn=True,
                                                            preprocess_text=False,
                                                            filter_small_data=filter_small_data)
    else:
        X_list, y_list = DataLoader.load_datasets_from_h5py('DataProcessed', True)


    print('Loaded data...')
    dataset_names = DataLoader.get_all_files('DataProcessed')
    dataset_names = [x.split('/')[-1].split('.')[0] for x in dataset_names]

    results_file = open(experiment_name + "_results.txt", "w+")

    for dataset_i, (X, y) in enumerate(zip(X_list, y_list)):
        if use_embedding:
            embedding = embedding_list[dataset_i]

        model_name = dataset_names[dataset_i]

        print("Dataset: {}".format(model_name))

        results_file.write(model_name)
        results_file.write("Dataset: {}".format(model_name))

        X_abstract, X_title, X_mesh = X['text'], X['title'], X['mesh']
        n = X_abstract.shape[0]
        kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

        if pretrain:
            pretrain_fold_accuracies = []
            pretrain_fold_recalls = []
            pretrain_fold_precisions =[]
            pretrain_fold_aucs = []
            pretrain_fold_f1s = []

        if transfer_learning:
            svm_fold_accuracies = []
            svm_fold_recalls = []
            svm_fold_precisions =[]
            svm_fold_aucs = []
            svm_fold_f1s = []

        fold_accuracies = []
        fold_recalls = []
        fold_precisions =[]
        fold_aucs = []
        fold_f1s = []

        for fold_idx, (train, test) in enumerate(kf):
            temp_model_name = experiment_name + '_' + model_name + '_fold_{}'.format(fold_idx + 1)


            cnn = AbstractCNN(n_classes=2,  max_words=max_words, w2v_size=w2v_size, vocab_size=1000, use_embedding=use_embedding,
                              filter_sizes=filter_sizes, n_feature_maps=n_feature_maps, dense_layer_sizes=dense_sizes.copy(),
                              name=temp_model_name, activation_function=activation, dropout_p=p, embedding=embedding)

            if pretrain:
                X_abstract_train = X_abstract[train, :, :]
                X_title_train = X_title[train, :, :]
                X_mesh_train = X_mesh[train, :, :]

                y_train = y[train, :]

                X_abstract_test = X_abstract[test, :, :]
                X_title_test = X_title[test, :, :]
                X_mesh_test = X_mesh[test, :, :]

                y_test = y[test, :]

                for i, (_x, _y) in enumerate(zip(X_list, y_list)):
                    if not i == dataset_i:
                        X_abstract_train = np.vstack((X_abstract_train, _x['text'][()]))
                        X_title_train = np.vstack((X_title_train, _x['title'][()]))
                        X_mesh_train = np.vstack((X_mesh_train, _x['mesh'][()]))
                        y_train = np.vstack((y_train, _y[()]))

                if undersample_all:
                    X_abstract_train, X_title_train, X_mesh_train, y_train = \
                        DataLoader.undersample_acnn(X_abstract_train, X_title_train, X_mesh_train, y_train)

                print(X_abstract_train.shape)

                cnn.train(X_abstract_train, X_title_train, X_mesh_train, y_train, n_epochs=epochs,
                          optim_algo=optimizer, criterion=criterion, verbose=verbose, patience=patience,
                          save_model=save_model)


                accuracy, f1_score, precision, auc, recall = cnn.test(X_abstract_test, X_title_test, X_mesh_test, y_test,
                                                                  print_output=True)

                print("Results from training on all data only")

                print("Accuracy: {}".format(accuracy))
                print("F1: {}".format(f1_score))
                print("Precision: {}".format(precision))
                print("AUC: {}".format(auc))
                print("Recall: {}".format(recall))
                print("\n")

                pretrain_fold_accuracies.append(accuracy)
                pretrain_fold_precisions.append(precision)
                pretrain_fold_recalls.append(recall)
                pretrain_fold_aucs.append(auc)
                pretrain_fold_f1s.append(f1_score)

            if not use_embedding:
                X_abstract_train = X_abstract[train, :, :]
                X_title_train = X_title[train, :, :]
                X_mesh_train = X_mesh[train, :, :]
                y_train = y[train, :]

                X_abstract_test = X_abstract[test, :, :]
                X_titles_test = X_title[test, :, :]
                X_mesh_test = X_mesh[test, :, :]
                y_test = y[test, :]


                if undersample:
                    X_abstract_train, X_title_train, X_mesh_train, y_train =\
                        DataLoader.undersample_acnn(X_abstract_train, X_title_train, X_mesh_train, y_train)
            elif use_embedding:
                X_abstract_train = X_abstract[train]
                X_title_train = X_title[train]
                X_mesh_train = X_mesh[train]
                y_train = y[train, :]

                X_abstract_test = X_abstract[test]
                X_titles_test = X_title[test]
                X_mesh_test = X_mesh[test]
                y_test = y[test, :]

                if undersample:
                    X_abstract_train, X_title_train, X_mesh_train, y_train = \
                        DataLoader.undersample_seq(X_abstract_train, X_title_train, X_mesh_train, y_train)

            cnn.train(X_abstract_train, X_title_train, X_mesh_train, y_train, n_epochs=epochs, optim_algo=optimizer,
                      criterion=criterion, verbose=verbose, patience=patience,
                      save_model=save_model)
            accuracy, f1_score, precision, auc, recall = cnn.test(X_abstract_test, X_titles_test, X_mesh_test, y_test,
                                                                  print_output)
            if transfer_learning:
                svm = SVM()

                # Transfer weights
                X_transfer_train = cnn.output_learned_features([X_abstract_train, X_title_train, X_mesh_train])
                X_transfer_test = cnn.output_learned_features([X_abstract_test, X_titles_test, X_mesh_test])

                svm.train(X_transfer_train, DataLoader.onehot2list(y_train))
                svm.test(X_transfer_test, DataLoader.onehot2list(y_test))

                print("\nSVM results")
                print(svm)
                print('\n')

                svm_fold_accuracies.append(svm.metrics['Accuracy'])
                svm_fold_precisions.append(svm.metrics['Precision'])
                svm_fold_aucs.append(svm.metrics['AUC'])
                svm_fold_recalls.append(svm.metrics['Recall'])
                svm_fold_f1s.append(svm.metrics['F1'])

            print('CNN results')
            print("Accuracy: {}".format(accuracy))
            print("F1: {}".format(f1_score))
            print("Precision: {}".format(precision))
            print("AUC: {}".format(auc))
            print("Recall: {}".format(recall))

            fold_accuracies.append(accuracy)
            fold_precisions.append(precision)
            fold_recalls.append(recall)
            fold_aucs.append(auc)
            fold_f1s.append(f1_score)



        if pretrain:
            pretrain_average_accuracy = np.mean(pretrain_fold_accuracies)
            pretrain_average_precision = np.mean(pretrain_fold_precisions)
            pretrain_average_recall = np.mean(pretrain_fold_recalls)
            pretrain_average_auc = np.mean(pretrain_fold_aucs)
            pretrain_average_f1 = np.mean(pretrain_fold_f1s)

            print("\nAverage results from using all data")
            print("Fold Average Accuracy: {}".format(pretrain_average_accuracy))
            print("Fold Average F1: {}".format(pretrain_average_f1))
            print("Fold Average Precision: {}".format(pretrain_average_precision))
            print("Fold Average AUC: {}".format(pretrain_average_auc))
            print("Fold Average Recall: {}".format(pretrain_average_recall))
            print('\n')



        average_accuracy = np.mean(fold_accuracies)
        average_precision = np.mean(fold_precisions)
        average_recall = np.mean(fold_recalls)
        average_auc = np.mean(fold_aucs)
        average_f1 = np.mean(fold_f1s)


        print('CNN Results')
        print("Fold Average Accuracy: {}".format(average_accuracy))
        print("Fold Average F1: {}".format(average_f1))
        print("Fold Average Precision: {}".format(average_precision))
        print("Fold Average AUC: {}".format(average_auc))
        print("Fold Average Recall: {}".format(average_recall))
        print('\n')

        results_file.write("CNN results")
        results_file.write("Fold Average Accuracy: {}\n".format(average_accuracy))
        results_file.write("Fold Average F1: {}\n".format(average_f1))
        results_file.write("Fold Average Precision: {}\n".format(average_precision))
        results_file.write("Fold Average AUC: {}\n".format(average_auc))
        results_file.write("Fold Average Recall: {}\n".format(average_recall))
        results_file.write('\n')

        if transfer_learning:
            average_accuracy = np.mean(svm_fold_accuracies)
            average_precision = np.mean(svm_fold_precisions)
            average_recall = np.mean(svm_fold_recalls)
            average_auc = np.mean(svm_fold_aucs)
            average_f1 = np.mean(svm_fold_f1s)

            print("SVM with cnn features")
            print("Fold Average Accuracy: {}".format(average_accuracy))
            print("Fold Average F1: {}".format(average_f1))
            print("Fold Average Precision: {}".format(average_precision))
            print("Fold Average AUC: {}".format(average_auc))
            print("Fold Average Recall: {}".format(average_recall))
            print('\n')

            results_file.write("SVM with cnn features\n")
            results_file.write("Fold Average Accuracy: {}\n".format(average_accuracy))
            results_file.write("Fold Average F1: {}\n".format(average_f1))
            results_file.write("Fold Average Precision: {}\n".format(average_precision))
            results_file.write("Fold Average AUC: {}\n".format(average_auc))
            results_file.write("Fold Average Recall: {}\n".format(average_recall))
            results_file.write('\n')

if __name__ == '__main__':
    main()
