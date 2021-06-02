import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import mne
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

from bciclassifier.classify import classify_target, classify_audiovisual
from bciclassifier.data_manager import DataManager
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay
import bciclassifier.constants as consts
from bciclassifier.keras_model import keras_model_target, keras_model_audiovisual


def plot_averaged_targets_non_targets(target_epochs=None, non_target_epochs=None):
    mne.viz.plot_compare_evokeds(dict(target=target_epochs.average(), non_target=non_target_epochs.average()),
                                 legend='upper left', show_sensors='upper right', title='Target vs. Non-target')


def plot_averaged_sessions(visual_epochs=None, audio_epochs=None, audiovisual_epochs=None):
    mne.viz.plot_compare_evokeds(dict(visual=visual_epochs.average(picks=('Pz',)),
                                      audio=audio_epochs.average(picks=('Pz',)),
                                      audiovisual=audiovisual_epochs.average(picks=('Pz',))),
                                 legend='upper left', show_sensors='upper right', title='Target epochs')


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to P300 datasets')
    parser.add_argument('-l', '--loglevel', choices=consts.LOG_LEVELS, help="logging level", default=logging.WARNING)
    parser.add_argument('-t', '--type', choices=consts.CLASSIFICATION_TYPES, help="type of classification",
                        default=consts.CLASSIFICATION_TYPES)
    parser.add_argument('-m', '--metrics', choices=consts.METRICS, help="which metrics should be evaluated",
                        default=consts.METRICS)
    parser.add_argument('--show', action='store_true', help="only show loaded data and do not classify")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig()
    logger = logging.getLogger('bciclassifier')
    logger.setLevel(args.loglevel)
    logger.debug(f"Command line arguments: {args}")

    # Set metrics
    metrics = args.metrics

    dm = DataManager(args.path)

    if args.show:
        times = np.arange(0.3, 0.6, 0.05)
        avg = 0.025
        target_epochs = dm.data[consts.TAG_TARGET]
        non_target_epochs = dm.data[consts.TAG_NON_TARGET]
        visual_epochs = dm.data[consts.TAG_VISUAL][consts.TAG_TARGET]
        audio_epochs = dm.data[consts.TAG_AUDIO][consts.TAG_TARGET]
        audiovisual_epochs = dm.data[consts.TAG_AUDIOVISUAL][consts.TAG_TARGET]
        plot_averaged_targets_non_targets(target_epochs=target_epochs, non_target_epochs=non_target_epochs)
        plot_averaged_sessions(visual_epochs=visual_epochs, audio_epochs=audio_epochs,
                               audiovisual_epochs=audiovisual_epochs)
        target_epochs.average().plot_topomap(times=times, average=avg, ncols=8, nrows='auto', title='Target epochs')
        non_target_epochs.average().plot_topomap(times=times, average=avg, ncols=8, nrows='auto',
                                                 title='Non-target epochs')
        audiovisual_epochs.average().plot_topomap(times=times, average=avg, ncols=8, nrows='auto',
                                                  title='AV target epochs')
        visual_epochs.average().plot_topomap(times=times, average=avg, ncols=8, nrows='auto',
                                             title='V target epochs')
        audio_epochs.average().plot_topomap(times=times, average=avg, ncols=8, nrows='auto',
                                            title='A target epochs')
    else:
        result = {}
        if consts.CLASSIFICATION_TYPES[0] in args.type:
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
            result[consts.CLASSIFICATION_TYPES[0]] = classify_target(dm, metrics, clf)
        if consts.CLASSIFICATION_TYPES[1] in args.type:
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
            result[consts.CLASSIFICATION_TYPES[1]] = classify_audiovisual(dm, metrics, clf)

        for res in result:
            print(f"{res}: {result[res]}")
            if "confusion_matrix" in result[res]:
                labels = [str(x) for (x, _) in enumerate(result[res]["confusion_matrix"])]
                ConfusionMatrixDisplay(result[res]["confusion_matrix"], display_labels=labels).plot()
            if "recall" in result[res] and "precision" in result[res]:
                PrecisionRecallDisplay(precision=result[res]["precision"], recall=result[res]["recall"]).plot()
            plt.show()


if __name__ == '__main__':
    main()
