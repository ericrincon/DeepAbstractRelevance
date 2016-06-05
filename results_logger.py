

class logger():
    def __init__(self, metrics, order=None):
        self.metrics = metrics

        if order is None:
            self.order = [key for key in metrics.keys]

        # Set up the metrics

    """
        results should be a dictatnory with the same keys as self.metrics
    """
    def log(self, results):
        for metric in self.order:
            result = results[metric]

            self.metrics[metric]


class