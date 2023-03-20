import os

from dataset import DataLoaderTrain, DataLoaderTrainOfficialWarped, DataLoaderTrainOfficialWarpedJointLearning, DataLoaderVal, DataLoaderTest
def get_training_data(rgb_dir, img_options, color_space='rgb', mask_dir='mask', opt=None):
    assert os.path.exists(rgb_dir), rgb_dir
    if opt.joint_learning_alpha:
        return DataLoaderTrainOfficialWarpedJointLearning(rgb_dir, img_options, None, color_space, mask_dir, opt)
    if 'official_warped' in rgb_dir:
        return DataLoaderTrainOfficialWarped(rgb_dir, img_options, None, color_space, mask_dir, opt)
    else:
        return DataLoaderTrain(rgb_dir, img_options, None)

def get_validation_data(rgb_dir, color_space='rgb', mask_dir='mask', opt=None):
    assert os.path.exists(rgb_dir), rgb_dir
    return DataLoaderVal(rgb_dir, None, color_space, mask_dir, opt=opt)

def get_test_data(rgb_dir, color_space, mask_dir, opt=None):
    assert os.path.exists(rgb_dir), rgb_dir
    return DataLoaderTest(rgb_dir, None, color_space, mask_dir, opt=opt)
