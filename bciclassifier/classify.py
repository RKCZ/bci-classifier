import logging

from sklearn.metrics import precision_score, recall_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bciclassifier.data_manager import DataManager


def classify_target(data_manager, metrics, classifier):
    """
    Trains and tests classification of target vs. non-target epochs.

    :param metrics:
    :param data_manager: DataManager object which supplies data for classification.
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info("Classifying target vs. non-target epochs.")
    # get samples for training
    x_train, y_train = data_manager.get_target_split()
    # flatten feature vectors
    x_train = DataManager.flatten(x_train)
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(x_train, y_train)
    del x_train, y_train
    x_test, y_test = data_manager.get_target_split(test=True)
    x_test = DataManager.flatten(x_test)
    predictions = pipe.predict(x_test)
    result = evaluate(metrics, y_test, predictions)
    return result


def evaluate(metrics, y_true, y_pred):
    result = {}
    if 'confusion_matrix' in metrics:
        result['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    if 'recall' in metrics:
        result['recall'] = recall_score(y_true, y_pred, average=None)
    if 'precision' in metrics:
        result['precision'] = precision_score(y_true, y_pred, average=None)
    return result


def classify_audiovisual(data_manager, metrics, classifier):
    """
    Trains and tests classification of audio vs. visual vs. audiovisual epochs.
    :param metrics:
    :param data_manager: DataManager object which supplies data for classification.
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info("Classifying visual vs. audio vs. audiovisual epochs.")
    # get samples for training
    x_train, y_train = data_manager.get_experiment_split()
    # flatten feature vectors
    x_train = DataManager.flatten(x_train)
    pipe = make_pipeline(StandardScaler(), classifier)
    pipe.fit(x_train, y_train)
    del x_train, y_train
    x_test, y_test = data_manager.get_experiment_split(test=True)
    x_test = DataManager.flatten(x_test)
    predictions = pipe.predict(x_test)
    result = evaluate(metrics, y_test, predictions)
    return result
