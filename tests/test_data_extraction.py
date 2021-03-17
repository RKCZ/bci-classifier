from collections import Counter

import pytest
import scipy.io
import bciclassifier

pytestmark = pytest.mark.skip(reason="Tests need to be updated.")


def test_read_mock_file(mocker):
    mocker.patch('scipy.io.loadmat')
    bciclassifier.read_mat_file('path/to/file')
    scipy.io.loadmat.assert_called_once_with('path/to/file')


def test_read_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        bciclassifier.read_mat_file('nonexistent/file')


@pytest.mark.parametrize("experiment_style, dataset, expected", [
    ("visual", "test", r's\d+_V_test\.dat_?\d*\.mat'),
    ("audiovisual", "test", r's\d+_AV_test\.dat_?\d*\.mat'),
    ("audio", "train", r's\d+_A_train\.dat_?\d*\.mat')
])
def test_get_data_filename_regex(experiment_style, dataset, expected):
    assert bciclassifier.get_data_filename_regex(experiment_style, dataset) == expected


@pytest.fixture
def mock_data_dir(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    f1 = audio_dir / "s1_A_test.dat_1.mat"
    f1.touch()
    visual_dir = tmp_path / "visual"
    visual_dir.mkdir()
    f2 = visual_dir / "s1_V_train.dat.mat"
    f2.touch()
    audiovisual_dir = tmp_path / "audiovisual"
    audiovisual_dir.mkdir()
    f3 = audiovisual_dir / "s1_AV_test.dat_1.mat"
    f3.touch()
    f4 = audiovisual_dir / "s1_AV_train.dat.mat"
    f4.touch()
    f5 = audiovisual_dir / "s2_AV_test.dat_1.mat"
    f5.touch()
    f6 = audiovisual_dir / "s2_AV_test.dat_2.mat"
    f6.touch()
    files = (f1, f2, f3, f4, f5, f6)
    return {"path": tmp_path, "filenames": [str(x) for x in files]}


@pytest.mark.parametrize("experiment_style, dataset, expected", [
    ("visual", "test", ()),
    ("audio", "test", (0,)),
    ("audiovisual", "test", (2, 4, 5)),
    ("audiovisual", "train", (3,))
])
def test_get_data_filenames(mock_data_dir, experiment_style, dataset, expected):
    exp_files = [mock_data_dir["filenames"][i] for i in expected]
    path = mock_data_dir["path"]
    regex = bciclassifier.get_data_filename_regex(experiment_style, dataset)
    # Using Counter for comparison because order of the files seems to not be consistent on all OS platforms
    assert Counter(bciclassifier.get_data_filenames(path, regex)) == Counter(exp_files)
