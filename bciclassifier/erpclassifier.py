import logging


class ERPClassifier:

    def __init__(self, classifier):
        """
        Creates new ERPClassifier wrapper for scikit classifiers.

        :param classifier: Scikit classifier-like.
        """
        self._classifier = classifier
        self._logger = logging.getLogger(__name__)

    def train(self, x_train, y_train):
        """
        Trains classifier with given dataset.

        :param x_train: Feature vectors for training.
        :param y_train: Label vectors for training.
        :return: Result of the classifier's fit method.
        """
        result = self._classifier.fit(x_train, y_train)
        return result

    def test(self, x_test, y_test):
        """
        Tests classifier on given dataset.

        :param x_test: Feature vectors.
        :param y_test: Labels.
        :return: Score of the test.
        """
        score = self._classifier.score(x_test, y_test)
        return score
