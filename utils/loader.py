import os

from dataset import DataLoaderTrain, DataLoaderTrainOfficialWarped, DataLoaderVal, DataLoaderTest
def get_training_data(rgb_dir, img_options, color_space='rgb'):
    assert os.path.exists(rgb_dir), rgb_dir
    if 'official_warped' in rgb_dir:
        return DataLoaderTrainOfficialWarped(rgb_dir, img_options, None, color_space)
    else:
        return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_data(rgb_dir, color_space='rgb'):
    assert os.path.exists(rgb_dir), rgb_dir
    return DataLoaderVal(rgb_dir, None, color_space)

def get_test_data(rgb_dir, color_space):
    assert os.path.exists(rgb_dir), rgb_dir
    return DataLoaderTest(rgb_dir, None, color_space)
