import logging

from scikeras.wrappers import KerasClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix

from bciclassifier.data_manager import DataManager
from bciclassifier.erpclassifier import ERPClassifier
from bciclassifier.keras_model import keras_model_target, keras_model_audiovisual


def classify_target(datamanager, metrics):
    """
    Trains and tests classification of target vs. non-target epochs.

    :param metrics:
    :param datamanager: DataManager object which supplies data for classification.
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info("Classifying target vs. non-target epochs.")
    clf = KerasClassifier(
        model=keras_model_target,
        loss="sparse_categorical_crossentropy",
        name="model_target",
        optimizer='adam',
        epochs=20,
        batch_size=128
    )
    data = datamanager.get_target_split()
    result = _classify(data, clf, metrics)
    return result


def _classify(data, pipeline, metrics):
    # flatten samples before training
    x_train = DataManager.flatten(data[0])
    x_test = DataManager.flatten(data[1])
    # flatten labels
    y_train = data[2].ravel()
    y_test = data[3].ravel()
    erpclassifier = ERPClassifier(pipeline)
    train_result = erpclassifier.train(x_train, y_train)
    predictions = erpclassifier.predict(x_test)

    result = {}
    if 'confusion_matrix' in metrics:
        result['confusion_matrix'] = confusion_matrix(y_test, predictions)
    if 'recall' in metrics:
        result['recall'] = recall_score(y_test, predictions, average=None)
    if 'precision' in metrics:
        result['precision'] = precision_score(y_test, predictions, average=None)

    return result


def classify_audiovisual(datamanager, metrics):
    """
    Trains and tests classification of audio vs. visual vs. audiovisual epochs.
    :param metrics:
    :param datamanager: DataManager object which supplies data for classification.
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info("Classifying visual vs. audio vs. audiovisual epochs.")
    clf = KerasClassifier(
        model=keras_model_audiovisual,
        loss="sparse_categorical_crossentropy",
        name="model_audiovisual",
        optimizer='adam',
        epochs=20,
        batch_size=128
    )
    data = datamanager.get_experiment_style_split()
    result = _classify(data, clf, metrics)
    return result
