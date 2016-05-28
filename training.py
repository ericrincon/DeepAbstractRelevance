
class Trainer():
    def __init__(self, batch_generator, learner):
        self.batch_generator = batch_generator
        self.learner = learner

    def run(self, n_epochs, metrics):

        for epoch in range(1, n_epochs + 1):
            batch_f1_history = []
            batch_precision_history = []
            batch_recall_history = []

            for X, y in self.batch_generator.next_batch():
                history = self.learner.model.fit(X, y, nb_epoch=1, batch_size=self.batch_generator.batch_size,
                                         validation_split=0.2, verbose=0)

                val_loss, loss = history.history['val_loss'][0], history.history['loss'][0]

                loss_train_history.append(loss)
                loss_val_history.append(val_loss)

                truth = self.model.validation_data[2]
                truth = dl.onehot2list(truth)
                batch_prediction = self.predict_classes(self.model.validation_data[0:2])

                batch_f1 = metrics.f1_score(truth, batch_prediction)
                batch_recall = metrics.recall_score(truth, batch_prediction)
                batch_precision = metrics.precision_score(truth, batch_prediction)

                batch_f1_history.append(batch_f1)
                batch_recall_history.append(batch_recall)
                batch_precision_history.append(batch_precision)

            batch_history['f1'].append(batch_f1_history)
            batch_history['recall'].append(batch_recall_history)
            batch_history['precision'].append(batch_precision_history)

            print('Epoch: {} | Train loss: {} | Valid loss: {}'.format(epoch, loss, val_loss))
            print("Epoch Metrics | F1: {} | Recall {} | Precision: {}".format(np.mean(batch_history['f1'][epoch - 1]),
                                                                              np.mean(batch_history['recall'][epoch - 1]),
                                                                              np.mean(batch_history['precision'][epoch - 1])))
            a_max = np.argmax(batch_history['f1'][epoch - 1])
            print("Best F1 at Epoch {} Minibatch {}: {}\n".format(epoch, a_max, batch_history['f1'][epoch-1][a_max]))