

class logger():
    def __init__(self, metrics, learner, order=None):
        self.metrics = []
        self.learner = learner

        if order is None:
            self.order = [key for key in metrics.keys]

    def log(self):
        results = self.learner.test()


class