import pandas as pd
import numpy as np

from SVM import SVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold

import Experiment
import DataLoader

def get_data():
    file_paths = DataLoader.get_all_files('Data')

    X_list, y_list = [], []

    for file_path in file_paths:
        data_frame = pd.read_csv(file_path)

        abstract_text, abstract_labels = DataLoader.extract_abstract_and_labels(data_frame)
        mesh_terms, title = DataLoader.extract_mesh_and_title(data_frame)

        X = []
        y = []

        for i in range(abstract_text.shape[0]):
            abstract_str = abstract_text[i]
            mesh_str = mesh_terms[i]
            title_str = title[i]
            label = abstract_labels[i]

            text = "".join([abstract_str, " ", mesh_str, " ", title_str])

            X.append(text)
            y.append(label)
        X_list.append(X)
        y_list.append(y)

    return X_list, y_list

def main():



    print("Loading data...")
    X_list, y_list = get_data()

    print("Loaded data...")
    print('\n')
    dataset_names = DataLoader.get_all_files('Data')
    dataset_names = [name.split('/')[1].split('.')[0] for name in dataset_names]
    undersample = True

    for i, (X, y) in enumerate(zip(X_list, y_list)):
        print("Dataset: {}".format(dataset_names[i]))

        X = np.array(X)
        y = np.array(y)

        n = len(X)

        kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

        fold_accuracies = []
        fold_recalls = []
        fold_precisions =[]
        fold_aucs = []
        fold_f1s = []

        for fold_idx, (train, test) in enumerate(kf):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            if undersample:
                # Get all the targets that are not relevant i.e., y = 0
                idx_undersample = np.where(y_train == -1)[0]

                # Get all the targets that are relevant i.e., y = 1
                idx_positive = np.where(y_train == 1)[0]
                # Now sample from the no relevant targets
                random_negative_sample = np.random.choice(idx_undersample, idx_positive.shape[0])

                X_train_positive = X_train[idx_positive]

                X_train_negative = X_train[random_negative_sample]

                X_train_undersample = np.hstack((X_train_positive, X_train_negative))

                y_train_positive = y_train[idx_positive]
                y_train_negative = y_train[random_negative_sample]
                y_train_undersample = np.hstack((y_train_positive, y_train_negative))

            count_vec = CountVectorizer(ngram_range=(1, 3), max_features=50000)

            count_vec.fit(X_train)

            if undersample:
                X_train = X_train_undersample
                y_train = y_train_undersample

            X_train_undersample = count_vec.transform(X_train)
            X_test = count_vec.transform(X_test)

            svm = SVM()
            svm.train(X_train_undersample, y_train)
            svm.test(X_test, y_test)

            f1_score = svm.metrics["F1"]
            precision = svm.metrics["Precision"]
            recall = svm.metrics["Recall"]
            auc = svm.metrics["AUC"]
            accuracy = svm.metrics["Accuracy"]

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