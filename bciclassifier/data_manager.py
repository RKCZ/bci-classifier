import os
import re
import logging

import numpy as np
import scipy.io
from mne import EpochsArray, create_info

TAG_NON_TARGET = 'allNTARGETS'

TAG_TARGET = 'allTARGETS'

TAG_TRAIN = 'train'

TAG_TEST = 'test'


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
        experiment_styles = ('visual', 'audio', 'audiovisual')
        classes = (TAG_TARGET, TAG_NON_TARGET)
        if test:
            dataset = TAG_TEST
        else:
            dataset = TAG_TRAIN
        features = np.empty(shape=(0, 16, 614))
        labels = np.empty(shape=(0, 1))
        for index, class_ in enumerate(classes):
            for xs in experiment_styles:
                feats = self.extract_features(xs, dataset, class_)
                lbs = np.tile(index, (len(feats), 1))
                features = np.concatenate((features, feats), axis=0)
                labels = np.concatenate((labels, lbs), axis=0)
        return features, labels

    def get_experiment_split(self, test=False):
        """
        Returns a dataset for classification into audio, visual and audiovisual classes.

        :param test: A boolean switch. If true a test dataset is returned.
        :return: A tuple with feature vectors and labels for experiment style classification.
        """
        classes = ('visual', 'audio', 'audiovisual')
        targets = TAG_TARGET
        if test:
            dataset = TAG_TEST
        else:
            dataset = TAG_TRAIN
        features = np.empty(shape=(0, 16, 614))
        labels = np.empty(shape=(0, 1))
        for index, class_ in enumerate(classes):
            feats = self.extract_features(class_, dataset, targets)
            lbs = np.tile(index, (len(feats), 1))
            features = np.concatenate((features, feats), axis=0)
            labels = np.concatenate((labels, lbs), axis=0)
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
        Extracts all data from files at given path and subdirectories. The data are stored into dictionary object with
        keys 'audio', 'visual' and 'audiovisual'. Each of these values are another dictionaries with data divided by
        dataset type_ ('test', 'train').

        :return: A directory of data extracted from files at given location.
        """
        result = {}
        for experiment_style in ("visual", "audio", "audiovisual"):
            self._logger.debug(f"Extracting data for {experiment_style}.")
            experiment_dict = {}
            for dataset in (TAG_TEST, TAG_TRAIN):
                self._logger.debug(f"Extracting {dataset} data.")
                subjects = []
                pattern = DataManager.get_data_filename_regex(experiment_style, dataset)
                files = self.get_data_filenames(pattern)
                for f in files:
                    subject = DataManager._read_mat_file(f)
                    info = create_info(ch_names=[x[0] for x in subject['electrodes'].ravel()], ch_types='eeg',
                                       sfreq=512)
                    target_epochs = EpochsArray(np.transpose(subject['allTARGETS'], (0, 2, 1)), info)
                    ntarget_epochs = EpochsArray(np.transpose(subject['allNTARGETS'], (0, 2, 1)), info)
                    subjects.append({'allTARGETS': target_epochs, 'allNTARGETS': ntarget_epochs})
                experiment_dict[dataset] = subjects
            result[experiment_style] = experiment_dict
        return result

    def extract_features(self, experiment_style, dataset, target):
        """
        Extract selected feature vectors from data structure.

        :param experiment_style: Type of experiment data ('visual', 'audio', 'audiovisual').
        :param dataset: Dataset to load ('test', 'train').
        :param target:
        :return:
        """
        features = np.empty(shape=(0, 16, 614))
        subjects = self.data[experiment_style][dataset]
        for subject in subjects:
            features = np.concatenate((features, subject[target].get_data()))
        return features
