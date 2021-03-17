import logging


class ERPClassifier:

    def __init__(self, classifier):
        self._classifier = classifier
        self._logger = logging.getLogger(__name__)

    def train(self, x_train, y_train):
        result = self._classifier.fit(x_train, y_train)
        return result

    def test(self, x_test, y_test):
        score = self._classifier.score(x_test, y_test)
        return score
