import os
import re
import logging

import numpy as np
import scipy.io

TAG_NON_TARGET = 'allNTARGETS'

TAG_TARGET = 'allTARGETS'

TAG_TRAIN = 'train'

TAG_TEST = 'test'


class DataManager:

    def __init__(self, data_path: str):
        self._logger = logging.getLogger(__name__)
        if data_path is not None:
            self._path = data_path
            self.data = self.extract_data()

    def get_target_split(self):
        experiment_styles = ('visual', 'audio', 'audiovisual')
        classes = (TAG_TARGET, TAG_NON_TARGET)

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for index, class_ in enumerate(classes):
            for xs in experiment_styles:
                # process train data
                feats = self.extract_features(xs, TAG_TRAIN, class_)
                labels = np.tile(index, (len(feats), 1))
                x_train.extend(feats)
                y_train.extend(labels)
                # process test data
                feats = self.extract_features(xs, TAG_TEST, class_)
                labels = np.tile(index, (len(feats), 1))
                x_test.extend(feats)
                y_test.extend(labels)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        return x_train, x_test, y_train, y_test

    def get_experiment_style_split(self):
        classes = ('visual', 'audio', 'audiovisual')
        targets = TAG_TARGET

        x_train = []
        y_train = []
        x_test = []
        y_test = []
        for index, class_ in enumerate(classes):
            # process train data
            feats = self.extract_features(class_, TAG_TRAIN, targets)
            labels = np.tile(index, (len(feats), 1))
            x_train.extend(feats)
            y_train.extend(labels)
            # process test data
            feats = self.extract_features(class_, TAG_TEST, targets)
            labels = np.tile(index, (len(feats), 1))
            x_test.extend(feats)
            y_test.extend(labels)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        return x_train, x_test, y_train, y_test

    @staticmethod
    def flatten(data):
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
                    subjects.append(DataManager._read_mat_file(f))
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
        features = []
        subjects = self.data[experiment_style][dataset]
        for subject in subjects:
            for epoch in subject[target]:
                features.append(epoch)
        return np.array(features)
