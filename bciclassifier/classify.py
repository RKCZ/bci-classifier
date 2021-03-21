from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from bciclassifier.erpclassifier import ERPClassifier
from bciclassifier.keras_model import keras_model_target, keras_model_audiovisual
from bciclassifier.data_manager import DataManager

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import logging


def classify_target(datamanager):
    """
    Trains and tests classification of target vs. non-target epochs.

    :param datamanager: DataManager object which supplies data for classification.
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info("Classifying target vs. non-target epochs.")
    normalizer = Normalizer()
    clf = KerasClassifier(build_fn=keras_model_target, name="model_target", optimizer='adam', epochs=20, batch_size=128)
    pipeline = make_pipeline(normalizer, clf, verbose=True)
    data = datamanager.get_target_split()
    _classify(data, pipeline)


def _classify(data, pipeline):
    # flatten samples before training
    x_train = DataManager.flatten(data[0])
    x_test = DataManager.flatten(data[1])
    # flatten labels
    y_train = data[2].ravel()
    y_test = data[3].ravel()
    erpclassifier = ERPClassifier(pipeline)
    train_result = erpclassifier.train(x_train, y_train)
    test_score = erpclassifier.test(x_test, y_test)
    print(test_score)


def classify_audiovisual(datamanager):
    """
    Trains and tests classification of audio vs. visual vs. audiovisual epochs.
    :param datamanager: DataManager object which supplies data for classification.
    :return: None
    """
    logger = logging.getLogger(__name__)
    logger.info("Classifying visual vs. audio vs. audiovisual epochs.")
    normalizer = Normalizer()
    clf = KerasClassifier(build_fn=keras_model_audiovisual, name="model_audiovisual", optimizer='adam', epochs=20,
                          batch_size=128)
    pipeline = make_pipeline(normalizer, clf, verbose=True)
    data = datamanager.get_experiment_style_split()
    # flatten samples before training
    _classify(data, pipeline)
