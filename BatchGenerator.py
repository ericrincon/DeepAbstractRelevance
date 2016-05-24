import numpy as np
import math

"""
    Mini batch generator for a binary imbalanced dataset.
"""


class ImbalancedBBG:

    def __init__(self, X, y, batch_size, shuffle=True, pos_p=.5):
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
        pos_idx = np.where(y[:, 0] == 1)[0]
        neg_idx = np.where(y[:, 1] == 1)[0]

        return pos_idx, neg_idx

    def _create_batch(self, pos_idx, neg_idx):
        idxs = np.array([x for x in range(self.batch_size)])
        np.random.shuffle(idxs)

        X_abstract_batch = np.vstack((self.X_abstract[pos_idx], self.X_abstract[neg_idx]))
        X_title_batch = np.vstack((self.X_title[pos_idx], self.X_title[neg_idx]))
        X_mesh_batch = np.vstack((self.X_mesh[pos_idx], self.X_mesh[neg_idx]))
        y_batch = np.vstack((self.y[pos_idx], self.y[neg_idx]))

        return [X_abstract_batch[idxs], X_title_batch[idxs], X_mesh_batch[idxs]], y_batch[idxs]




