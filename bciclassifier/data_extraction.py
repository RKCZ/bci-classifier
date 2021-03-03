import scipy.io
import os
import re


def read_mat_file(path):
    """
    Loads data from a single .mat file.
    :param path: Path to a .mat file.
    :return: Contents of the specified .mat file.
    """
    data = scipy.io.loadmat(path)
    return data


def get_data_filename_regex(experiment_style='audiovisual', dataset='train'):
    """
    Generates a regular expression for matching data files.
    :param experiment_style:  Type of the queried experiment. Must be one of 'audio', 'visual', 'audiovisual'.
    :param dataset: Type of dataset. The value should be either 'train' or 'test'.
    :return: Raw string containing regular expression, which can be used as a pattern for matching data files.
    """
    experiments_dict = {'audio': 'A', 'audiovisual': 'AV', 'visual': 'V'}
    return fr's\d+_{experiments_dict[experiment_style]}_{dataset}\.dat_?\d*\.mat'


def get_data_filenames(path, pattern):
    """
    Finds paths to all data files which match a given pattern. The search is executed in a directory specified by path
    and all its subdirectories.
    :param pattern: A regular expression pattern to match against filenames.
    :param path: A path to a directory where files are searched.
    :return: A list of paths to matched files in the given directory and its subdirectories.
    """
    result = []
    for path, dirs, files in os.walk(path):
        for f in files:
            if re.match(pattern, f):
                result.append(os.path.join(path, f))
    return result


def extract_data(path):
    """
    Extracts all data from files at given path and subdirectories. The data are stored into dictionary object with
    keys 'audio', 'visual' and 'audiovisual'. Each of these values are another dictionaries with data divided by
    dataset type ('test', 'train').
    :param path: A path to root of data directory.
    :return: A directory of data extracted from files at given location.
    """
    result = {}
    for experiment_style in ("visual", "audio", "audiovisual"):
        experiment_dict = {}
        for dataset in ("test", "train"):
            subjects = []
            pattern = get_data_filename_regex(experiment_style, dataset)
            files = get_data_filenames(path, pattern)
            for f in files:
                subjects.append(read_mat_file(f))
            experiment_dict[dataset] = subjects
        result[experiment_style] = experiment_dict
    return result


def extract_features(data, experiment_style, dataset, target):
    """
    Extract selected feature vectors from data structure.
    :param data: Data structure loaded from external source.
    :param experiment_style: Type of experiment data ('visual', 'audio', 'audiovisual').
    :param dataset: Dataset to load ('test', 'train').
    :param target:
    :return:
    """
    features = []
    subjects = data[experiment_style][dataset]
    for subject in subjects:
        for epoch in subject[target]:
            features.append(epoch)
    return features
