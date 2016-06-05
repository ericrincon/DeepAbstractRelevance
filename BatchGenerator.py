import numpy as np
import math

"""
    Mini batch generator for a binary imbalanced dataset.
"""
class ImbalancedBBG:

    def __init__(self, X, y, batch_size, shuffle=True, pos_p=.5, idxs=None):

        if idxs is None:
            self.indices = [i for i in range(X[0].shape[0])]
        else :
            self.indices = idxs
        pos_idx, neg_idx = self._get_pos_ned_idx(y)

        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.pos_p = pos_p
        self.X_abstract = X[0]
        self.X_title = X[1]
        self.X_mesh = X[2]
        self.y = y

    def next_batch(self):
        np.random.shuffle(self.pos_idx)
        np.random.shuffle(self.neg_idx)

        pos_size = math.floor(self.batch_size * self.pos_p)
        neg_size = self.batch_size - pos_size

        pos_start = 0
        pos_end = pos_size

        neg_start = 0
        neg_end = neg_size

        while True:
            if neg_end >= self.neg_idx.shape[0]:
                break
            else:
                pos_idxs = self.pos_idx[pos_start: pos_end]
                neg_idxs = self.neg_idx[neg_start: neg_end]

                pos_start = pos_end
                pos_end += pos_size

                neg_start = neg_end
                neg_end += neg_size

                yield self._create_batch(pos_idxs, neg_idxs)

    def _get_pos_ned_idx(self, y):
        pos_idx = np.where(y[self.indices, 0] == 1)[0]
        neg_idx = np.where(y[self.indices, 1] == 1)[0]

        return pos_idx, neg_idx

    def _create_batch(self, pos_idx, neg_idx):
        idxs = np.array([x for x in range(self.batch_size)])
        np.random.shuffle(idxs)

        pos_idx = np.sort(pos_idx)
        neg_idx = np.sort(neg_idx)

        X_abstract_batch = np.vstack((self.X_abstract[pos_idx, :, :], self.X_abstract[neg_idx, :, :]))
        X_title_batch = np.vstack((self.X_title[pos_idx, :, :], self.X_title[neg_idx, :, :]))
        X_mesh_batch = np.vstack((self.X_mesh[pos_idx, :, :], self.X_mesh[neg_idx, :, :]))
        y_batch = np.vstack((self.y[pos_idx, :], self.y[neg_idx, :]))
        return [X_abstract_batch[idxs], X_title_batch[idxs], X_mesh_batch[idxs]], y_batch[idxs]

class StanderedDomainBG:
    def __init__(self, X, y, batch_size, shuffle=True, fold_indices=None):
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.X_text = X[0]
        self.X_title = X[1]
        self.X_mesh = X[2]
        self.X_domain = X[3]
        self.y = y

        if fold_indices is None:
            self.indices = [i for i in range(X.shape[0])]
        else :
            self.indices = fold_indices

        self.n_examples = len(fold_indices)

        if shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        start = 0
        end = self.batch_size

        while True:
            if end >= self.n_examples:
                break
            else:
                idxs = self.indices[start: end]
                sorted_idxs = np.sort(idxs)
                X_text_batch = self.X_text[sorted_idxs, :, :]
                X_title_batch = self.X_title[sorted_idxs, :, :]
                X_mesh_batch = self.X_mesh[sorted_idxs, :, :]
                X_domain_batch = self.X_domain[sorted_idxs, :]

                y_batch = self.y[sorted_idxs, :]

                if self.shuffle_data:

                    X = [np.random.shuffle(x) for x in X]
                    np.random.shuffle(y_batch)

                start = end
                end += self.batch_size

                yield [X_text_batch, X_title_batch, X_mesh_batch, X_domain_batch], y_batch


class StanderedBG:
    def __init__(self, X, y, batch_size, shuffle=True, fold_indices=None):
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.X_text = X[0]
        self.X_title = X[1]
        self.X_mesh = X[2]
        self.X_domain = X[3]
        self.y = y

        if fold_indices is None:
            self.indices = [i for i in range(X.shape[0])]
        else :
            self.indices = fold_indices

        if shuffle:
            np.random.shuffle(self.indices)

    def next_batch(self):
        start = 0
        end = self.batch_size

        while True:
            if end >= self.X.shape[0]:
                break
            else:
                idxs = self.indices[start: end]
                sorted_idxs = np.sort(idxs)

                X = self.X[sorted_idxs, :, :]

                y = self.y[sorted_idxs, :]

                if self.shuffle_data:
                    np.random.shuffle(X)
                    np.random.shuffle(y)

                start = end
                end += self.batch_size

                yield X, y


"""
    Custom batch generator tailored for an imbalanced dataset and a dataset that contains
    multiple domains.
"""
class DomainBG:
    def __init__(self, X, y, batch_size, shuffle=True, pos_p=.5, fold_indices=None):

        if fold_indices is None:
            self.indices = [i for i in range(X[0].shape[0])]
        else :
            self.indices = fold_indices

        pos_idx, neg_idx = self._get_pos_ned_idx(y)

        self.pos_idx = pos_idx
        self.neg_idx = neg_idx
        self.batch_size = batch_size
        self.shuffle_data = shuffle
        self.pos_p = pos_p
        self.X_text = X[0]
        self.X_title = X[1]
        self.X_mesh = X[2]
        self.X_domain = X[3]
        self.y = y

    def next_batch(self):
        np.random.shuffle(self.pos_idx)
        np.random.shuffle(self.neg_idx)

        pos_size = math.floor(self.batch_size * self.pos_p)
        neg_size = self.batch_size - pos_size

        pos_start = 0
        pos_end = pos_size

        neg_start = 0
        neg_end = neg_size

        while True:
            if neg_end >= self.neg_idx.shape[0]:
                break
            else:

                pos_idxs = self.pos_idx[pos_start: pos_end]
                neg_idxs = self.neg_idx[neg_start: neg_end]

                pos_start = pos_end
                pos_end += pos_size

                neg_start = neg_end
                neg_end += neg_size

                yield self._create_batch(pos_idxs, neg_idxs)

    def _get_pos_ned_idx(self, y):
        pos_idx = np.where(y[self.indices, 0] == 1)[0]
        neg_idx = np.where(y[self.indices, 1] == 1)[0]

        return pos_idx, neg_idx

    def _create_batch(self, pos_idx, neg_idx):
        idxs = [x for x in range(self.batch_size)]

        pos_sorted_idxs = np.sort(pos_idx)
        neg_sorted_idxs = np.sort(neg_idx)

        X_text_batch = np.vstack((self.X_text[pos_sorted_idxs, :, :],
                                  self.X_text[neg_sorted_idxs, :, :]))
        X_title_batch = np.vstack((self.X_title[pos_sorted_idxs, :, :],
                                   self.X_title[neg_sorted_idxs, :, :]))
        X_mesh_batch = np.vstack((self.X_mesh[pos_sorted_idxs, :, :],
                                  self.X_mesh[neg_sorted_idxs, :, :]))

        X_domain_batch = np.vstack((self.X_domain[pos_sorted_idxs, :],
                                    self.X_domain[neg_sorted_idxs, :]))
        y_batch = np.vstack((self.y[pos_sorted_idxs, :],
                             self.y[neg_sorted_idxs, :]))
        np.random.shuffle(idxs)

        return [X_text_batch[idxs], X_title_batch[idxs], X_mesh_batch[idxs],
                X_domain_batch[idxs]], y_batch[idxs]
