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
    data = datamanager.get_target_split()
    result = _classify(data, clf, metrics)
    return result


def _classify(data, pipeline, metrics):
    # flatten samples before training
    x_train = DataManager.flatten(data[0])
    x_test = DataManager.flatten(data[1])
    y_train = data[2]
    y_test = data[3]
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
        loss=SparseCategoricalCrossentropy(),
        name="model_audiovisual",
        optimizer=Adam(),
        init=GlorotUniform(),
        metrics=[SparseCategoricalAccuracy()],
        epochs=5,
        batch_size=128
    )
    data = datamanager.get_experiment_style_split()
    result = _classify(data, clf, metrics)
    return result
