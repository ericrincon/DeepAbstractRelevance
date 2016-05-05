import pandas as pd
import numpy as np

from SVM import SVM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import KFold

import Experiment

def get_data():
    file_paths = Experiment.get_all_files('Data')

    X_list, y_list = [], []

    for file_path in file_paths:
        data_frame = pd.read_csv(file_path)

        abstract_text, abstract_labels = Experiment.extract_abstract_and_labels(data_frame)
        mesh_terms, title = Experiment.extract_mesh_and_title(data_frame)

        X = []
        y = []

        for i in range(abstract_text.shape[0]):
            abstract_str = abstract_text[i]
            mesh_str = mesh_terms[i]
            title_str = title[i]
            label = abstract_labels[i]

            text = "".join([abstract_str, " ", mesh_str, " " ,title_str])

            X.append(text)
            y.append(label)
        X_list.append(X)
        y_list.append(y)

    return X_list, y_list

def main():
    X_list, y_list = get_data()

    for X, y in zip(X_list, y_list):
        X = np.array(X)
        y = np.array(y)

        n = len(X)

        kf = KFold(n, random_state=1337, shuffle=True, n_folds=5)

        for fold_idx, (train, test) in enumerate(kf):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]

            count_vec = CountVectorizer()

            X_train = count_vec.fit_transform(X_train)
            X_test = count_vec.transform(X_test)

            svm = SVM()
            svm.train(X_train, y_train)
            svm.test(X_test, y_test)

            print(svm)


if __name__ == '__main__':
    main()