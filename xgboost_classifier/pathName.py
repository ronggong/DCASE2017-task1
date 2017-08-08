import os

path_dcase2017_origin = '/Volumes/Rong Seagat/dcase/task1/TUT-acoustic-scenes-2017-development'
# path_dcase2017_origin = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/dcase/task1/TUT-acoustic-scenes-2017-development'
path_dcase2017_augmentation = '/Volumes/Rong Seagat/dcase/'
# path_dcase2017_augmentation = '/media/gong/ec990efa-9ee0-4693-984b-29372dcea0d1/Data/dcase/'

path_dcase2017_evaluation = '/Users/gong/Documents/MTG document/dataset/dcase/TUT-acoustic-scenes-2017-evaluation'


path_audio_origin = os.path.join(path_dcase2017_origin, 'audio')

path_audio_eval = os.path.join(path_dcase2017_evaluation, 'audio')

path_audio_left = os.path.join(path_dcase2017_origin, 'audio_left')
path_audio_right = os.path.join(path_dcase2017_origin, 'audio_right')
path_audio_average = os.path.join(path_dcase2017_origin, 'audio_average')
path_audio_difference = os.path.join(path_dcase2017_origin, 'audio_difference')

path_audio_augmentation = os.path.join(path_dcase2017_augmentation, 'audio_augmentation')

path_feature_origin = os.path.join(path_dcase2017_origin, 'feature')
path_feature_augmentation = os.path.join(path_audio_augmentation, 'feature')

path_feature_freesound = os.path.join(path_feature_origin, 'freesound_extractor')
path_feature_freesound_augmentation = os.path.join(path_feature_augmentation, 'freesound_extractor')

# original audio files freesound features
path_feature_freesound_statistics = os.path.join(path_feature_freesound, 'statistics')
path_feature_freesound_statistics_left = os.path.join(path_feature_freesound, 'statistics_left')
path_feature_freesound_statistics_right = os.path.join(path_feature_freesound, 'statistics_right')
path_feature_freesound_statistics_average = os.path.join(path_feature_freesound, 'statistics_average')
path_feature_freesound_statistics_difference = os.path.join(path_feature_freesound, 'statistics_difference')

# original audio files freesound segment features
path_feature_freesound_statistics_seg_left = os.path.join(path_feature_freesound, 'statistics_seg_left')
path_feature_freesound_statistics_seg_right = os.path.join(path_feature_freesound, 'statistics_seg_right')
path_feature_freesound_statistics_seg_average = os.path.join(path_feature_freesound, 'statistics_seg_average')
path_feature_freesound_statistics_seg_difference = os.path.join(path_feature_freesound, 'statistics_seg_difference')

# augmented audio files freesound segment features
path_feature_freesound_statistics_seg_left_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_seg_left')
path_feature_freesound_statistics_seg_right_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_seg_right')
path_feature_freesound_statistics_seg_average_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_seg_average')
path_feature_freesound_statistics_seg_difference_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_seg_difference')

# original audio files freesound fold features
path_feature_freesound_statistics_folds = os.path.join(path_feature_freesound, 'statistics_folds')

path_feature_freesound_statistics_folds_left = os.path.join(path_feature_freesound, 'statistics_folds_left')
path_feature_freesound_statistics_folds_right = os.path.join(path_feature_freesound, 'statistics_folds_right')
path_feature_freesound_statistics_folds_average = os.path.join(path_feature_freesound, 'statistics_folds_average')
path_feature_freesound_statistics_folds_difference = os.path.join(path_feature_freesound, 'statistics_folds_difference')

# original audio files freesound fold segment features
path_feature_freesound_statistics_folds_seg_left = os.path.join(path_feature_freesound, 'statistics_folds_seg_left')
path_feature_freesound_statistics_folds_seg_right = os.path.join(path_feature_freesound, 'statistics_folds_seg_right')
path_feature_freesound_statistics_folds_seg_average = os.path.join(path_feature_freesound, 'statistics_folds_seg_average')
path_feature_freesound_statistics_folds_seg_difference = os.path.join(path_feature_freesound, 'statistics_folds_seg_difference')

# augmented audio files freesound fold segment features
path_feature_freesound_statistics_folds_seg_left_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_folds_seg_left')
path_feature_freesound_statistics_folds_seg_right_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_folds_seg_right')
path_feature_freesound_statistics_folds_seg_average_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_folds_seg_average')
path_feature_freesound_statistics_folds_seg_difference_augmentation = os.path.join(path_feature_freesound_augmentation, 'statistics_folds_seg_difference')

path_evaluation_setup = os.path.join(path_dcase2017_origin, 'evaluation_setup')

path_fold1_train_txt = os.path.join(path_evaluation_setup, 'fold1_train.txt')
path_fold1_eval_txt = os.path.join(path_evaluation_setup, 'fold1_evaluate.txt')

path_fold2_train_txt = os.path.join(path_evaluation_setup, 'fold2_train.txt')
path_fold2_eval_txt = os.path.join(path_evaluation_setup, 'fold2_evaluate.txt')

path_fold3_train_txt = os.path.join(path_evaluation_setup, 'fold3_train.txt')
path_fold3_eval_txt = os.path.join(path_evaluation_setup, 'fold3_evaluate.txt')

path_fold4_train_txt = os.path.join(path_evaluation_setup, 'fold4_train.txt')
path_fold4_eval_txt = os.path.join(path_evaluation_setup, 'fold4_evaluate.txt')

path_classifier = './classifiers_non_aug_lgb_overall'
path_results = './results_non_aug_lgb_overall'