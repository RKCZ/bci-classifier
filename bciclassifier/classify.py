import logging

from scikeras.wrappers import KerasClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.losses import SparseCategoricalCrossentropy

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
        loss=SparseCategoricalCrossentropy(),
        name="model_target",
        optimizer=Adam(),
        init=GlorotUniform(),
        metrics=[SparseCategoricalAccuracy()],
        epochs=5,
        batch_size=128
    )
    # get samples for training
    x_train, y_train = datamanager.get_target_split()
    # flatten feature vectors
    x_train = DataManager.flatten(x_train)
    erpclassifier = ERPClassifier(clf)
    erpclassifier.train(x_train, y_train)
    del x_train, y_train
    x_test, y_test = datamanager.get_target_split(test=True)
    x_test = DataManager.flatten(x_test)
    predictions = erpclassifier.predict(x_test)
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
        loss=SparseCategoricalCrossentropy(),
        name="model_audiovisual",
        optimizer=Adam(),
        init=GlorotUniform(),
        metrics=[SparseCategoricalAccuracy()],
        epochs=5,
        batch_size=128
    )
    # get samples for training
    x_train, y_train = datamanager.get_experiment_split()
    # flatten feature vectors
    x_train = DataManager.flatten(x_train)
    erpclassifier = ERPClassifier(clf)
    erpclassifier.train(x_train, y_train)
    del x_train, y_train
    x_test, y_test = datamanager.get_experiment_split(test=True)
    x_test = DataManager.flatten(x_test)
    predictions = erpclassifier.predict(x_test)
    result = evaluate(metrics, y_test, predictions)
    return result
