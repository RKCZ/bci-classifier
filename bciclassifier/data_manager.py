import os
import re
import logging

import mne
import numpy as np
import scipy.io
from mne import EpochsArray, create_info
import bciclassifier.constants as consts


class DataManager:

    def __init__(self, data_path: str):
        """
        Initializes Data manager. If the given data_path is valid, data are extracted automatically.

        :param data_path: Path to data folder.
        :raises NotADirectoryError
        """
        self._logger = logging.getLogger(__name__)
        if data_path is not None and os.path.isdir(data_path):
            self._path = data_path
            self.data = self.extract_data()
        else:
            raise NotADirectoryError

    def get_target_split(self, test=False):
        """
        Returns dataset for classification of target vs. non-target epochs.

        :param test: A boolean switch. If true a test dataset is returned.
        :return: A tuple with array of features and array of labels.
        """
        classes = (consts.TAG_TARGET, consts.TAG_NON_TARGET)
        if test:
            dataset = consts.TAG_TEST
        else:
            dataset = consts.TAG_TRAIN

        features_list = []
        labels_list = []
        for index, class_ in enumerate(classes):
            feats = self.data[class_][dataset].get_data()
            features_list.append(feats)
            labels_list.append(np.full_like(feats[:, 0, 0], index))
        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    def get_experiment_split(self, test=False):
        """
        Returns a dataset for classification into audio, visual and audiovisual classes.

        :param test: A boolean switch. If true a test dataset is returned.
        :return: A tuple with feature vectors and labels for experiment style classification.
        """
        if test:
            dataset = consts.TAG_TEST
        else:
            dataset = consts.TAG_TRAIN

        features_list = []
        labels_list = []
        for index, class_ in enumerate(consts.EXPERIMENT_SESSIONS):
            feats = self.data[class_][dataset][consts.TAG_TARGET].get_data()
            features_list.append(feats)
            labels_list.append(np.full_like(feats[:, 0, 0], index))
        features = np.concatenate(features_list)
        labels = np.concatenate(labels_list)
        return features, labels

    @staticmethod
    def flatten(data):
        """
        Reshapes array of features into array of flat feature vectors.

        :param data: Array of features of arbitrary dimensionality. First dimension of the array is taken as number
        of samples.
        :return: Reshaped array with the same number of samples.
        """
        samples = data.shape[0]
        return data.reshape((samples, -1))

    @staticmethod
    def _read_mat_file(file):
        data = scipy.io.loadmat(file)
        return data

    @staticmethod
    def get_data_filename_regex(experiment_style='audiovisual', dataset='train'):
        """
        Generates a regular expression for matching data files.

        :param experiment_style:  Type of the queried experiment. Must be one of 'audio', 'visual', 'audiovisual'.
        :param dataset: Type of dataset. The value should be either 'train' or 'test'.
        :return: Raw string containing regular expression, which can be used as a pattern for matching data files.
        """
        experiments_dict = {'audio': 'A', 'audiovisual': 'AV', 'visual': 'V'}
        return fr's\d+_{experiments_dict[experiment_style]}_{dataset}\.dat_?\d*\.mat'

    def get_data_filenames(self, pattern):
        """
        Finds paths to all data files which match a given pattern. The search is executed in a directory specified by
        path and all its subdirectories.

        :param pattern: A regular expression pattern to match against filenames.
        :return: A list of paths to matched files in the
        given directory and its subdirectories.
        """
        result = []
        for path, dirs, files in os.walk(self._path):
            for f in files:
                if re.match(pattern, f):
                    result.append(os.path.join(path, f))
        return result

    def extract_data(self):
        """
        Extracts all data from files at given path and subdirectories. The data are stored into `mne.EpochsArray`
        object.

        :return: Extracted epochs.
        """
        event_id_iterator = 1
        epochs_list = []
        for experiment_style in consts.EXPERIMENT_SESSIONS:
            self._logger.debug(f"Extracting data for {experiment_style}.")
            for dataset in (consts.TAG_TEST, consts.TAG_TRAIN):
                self._logger.debug(f"Extracting {dataset} data.")
                pattern = DataManager.get_data_filename_regex(experiment_style, dataset)
                files = self.get_data_filenames(pattern)
                for index, file in enumerate(files):
                    subject = DataManager._read_mat_file(file)
                    # prepare info object and set sensor locations
                    n_samples = subject['tSCALE'].shape[-1]
                    t_min, t_max = subject['tSCALE'][0][[0, -1]]
                    duration = t_max - t_min
                    sampling_freq = (n_samples - 1) / duration
                    ch_names = [x[0] for x in subject['electrodes'].flat]
                    ch_types = ['eeg'] * len(ch_names)
                    info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sampling_freq)
                    info.set_montage('standard_1020')

                    # create epochs
                    target_epochs = self._create_epochs(
                        events_dict={f'{experiment_style}/{dataset}/{index}/{consts.TAG_TARGET}': event_id_iterator},
                        info=info, data=np.transpose(subject[consts.TAG_TARGET], (0, 2, 1)),
                        event_number=event_id_iterator, t_min=t_min
                    )
                    event_id_iterator = event_id_iterator + 1
                    n_target_epochs = self._create_epochs(
                        events_dict={
                            f'{experiment_style}/{dataset}/{index}/{consts.TAG_NON_TARGET}': event_id_iterator},
                        info=info, data=np.transpose(subject[consts.TAG_NON_TARGET], (0, 2, 1)),
                        event_number=event_id_iterator, t_min=t_min
                    )
                    event_id_iterator = event_id_iterator + 1
                    epochs_list.append(target_epochs)
                    epochs_list.append(n_target_epochs)
        result = mne.concatenate_epochs(epochs_list)
        return result

    @staticmethod
    def _create_epochs(events_dict=None, info: mne.Info = None, data=None, event_number=0, t_min=0):
        # Convert values to volts
        data = data * 1.0e-6
        n_epochs = data.shape[0]
        n_samples = data.shape[2]
        events = np.column_stack((
            np.arange(0, n_epochs * n_samples, n_samples),
            np.zeros(n_epochs, dtype=int),
            np.full((n_epochs,), event_number)
        ))
        epochs = EpochsArray(
            data,
            info,
            baseline=(0, -t_min),
            events=events,
            event_id=events_dict
        )
        epochs = epochs.shift_time(t_min, relative=False)
        return epochs
