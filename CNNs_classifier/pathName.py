import os
from globalParams import *

if environ == 'local':
    path_dcase2017 = '/Volumes/Rong Seagat/dcase/task1/TUT-acoustic-scenes-2017-development'
    # path_dcase2017 = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/dcase/task1/TUT-acoustic-scenes-2017-development'
    path_audio = os.path.join(path_dcase2017, 'audio')
    path_audio_left = os.path.join(path_dcase2017, 'audio_left')
    path_audio_right = os.path.join(path_dcase2017, 'audio_right')
    path_audio_average = os.path.join(path_dcase2017, 'audio_average')
    path_audio_difference = os.path.join(path_dcase2017, 'audio_difference')

else:
    path_dcase2017 = '/homedtic/rgong/DCASE'

path_audio_augmentation = os.path.join(path_dcase2017, 'audio_augmentation')
path_audio_eval = os.path.join(path_dcase2017, 'unseenEvalAudio')

path_feature = os.path.join(path_dcase2017, 'feature')

path_feature_CNNs = os.path.join(path_feature, 'CNNs')
path_feature_CNNs_logMel = os.path.join(path_feature_CNNs, 'log-mel')
path_feature_CNNs_logMel_96bands = os.path.join(path_feature_CNNs_logMel, '96bands')

path_feature_CNNs_logMel_96bands_augmentation = os.path.join(path_feature_CNNs_logMel_96bands, 'augmentation')

path_feature_CNNs_logMel_96bands_unseenEval = os.path.join(path_feature_CNNs_logMel_96bands, 'unseenEval')

path_feature_CNNs_logMel_96bands_left = os.path.join(path_feature_CNNs_logMel_96bands, 'left')
path_feature_CNNs_logMel_96bands_right = os.path.join(path_feature_CNNs_logMel_96bands, 'right')
path_feature_CNNs_logMel_96bands_difference = os.path.join(path_feature_CNNs_logMel_96bands, 'difference')
path_feature_CNNs_logMel_96bands_average = os.path.join(path_feature_CNNs_logMel_96bands, 'average')

path_feature_CNNs_logMel_96bands_preprocessing = os.path.join(path_feature_CNNs_logMel_96bands, 'preprocessing')
path_feature_CNNs_logMel_96bands_models = os.path.join(path_feature_CNNs_logMel_96bands, 'models')
path_feature_CNNs_logMel_96bands_log = os.path.join(path_feature_CNNs_logMel_96bands, 'log')

path_evaluation_setup = os.path.join(path_dcase2017, 'evaluation_setup')

path_fold1_train_txt = os.path.join(path_evaluation_setup, 'fold1_train.txt')
path_fold1_eval_txt = os.path.join(path_evaluation_setup, 'fold1_evaluate.txt')

path_fold2_train_txt = os.path.join(path_evaluation_setup, 'fold2_train.txt')
path_fold2_eval_txt = os.path.join(path_evaluation_setup, 'fold2_evaluate.txt')

path_fold3_train_txt = os.path.join(path_evaluation_setup, 'fold3_train.txt')
path_fold3_eval_txt = os.path.join(path_evaluation_setup, 'fold3_evaluate.txt')

path_fold4_train_txt = os.path.join(path_evaluation_setup, 'fold4_train.txt')
path_fold4_eval_txt = os.path.join(path_evaluation_setup, 'fold4_evaluate.txt')

path_test_txt = os.path.join(path_evaluation_setup, 'test.txt')

if feature_aug:
    path_results = os.path.join(path_feature_CNNs_logMel_96bands_augmentation, 'results')
else:
    path_results = os.path.join(path_feature_CNNs_logMel_96bands, 'results')