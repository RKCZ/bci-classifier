from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier

from bciclassifier.erpclassifier import ERPClassifier
from bciclassifier.keras_model import keras_model_target, keras_model_audiovisual
from bciclassifier.data_manager import DataManager

from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

import logging


def classify_target(datamanager):
    logger = logging.getLogger(__name__)
    logger.info("Classifying target vs. non-target epochs.")
    normalizer = Normalizer()
    clf = KerasClassifier(build_fn=keras_model_target, name="model_target", optimizer='adam', epochs=20, batch_size=128)
    pipeline = make_pipeline(normalizer, clf, verbose=True)
    data = datamanager.get_target_split()
    _classify(data, pipeline)


def _classify(data, pipeline):
    # flatten samples before training
    data = [DataManager.flatten(x) for x in data]
    erpclassifier = ERPClassifier(pipeline)
    train_result = erpclassifier.train(data[0], data[2])
    test_score = erpclassifier.test(data[1], data[3])
    print(test_score)


def classify_audiovisual(datamanager):
    logger = logging.getLogger(__name__)
    logger.info("Classifying visual vs. audio vs. audiovisual epochs.")
    normalizer = Normalizer()
    clf = KerasClassifier(build_fn=keras_model_audiovisual, name="model_audiovisual", optimizer='adam', epochs=20,
                          batch_size=128)
    pipeline = make_pipeline(normalizer, clf, verbose=True)
    data = datamanager.get_experiment_style_split()
    # flatten samples before training
    _classify(data, pipeline)
