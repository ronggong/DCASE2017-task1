import librosa
import numpy as np
import pickle
from Fdeltas import Fdeltas
from utilFunctions import evalSetupTxtReader
import os

def batchPitchshiftAugmentation(path_audio_origin, path_audio_target, channel_string):
    filenames_audio = [f for f in os.listdir(path_audio_origin) if os.path.isfile(os.path.join(path_audio_origin, f))]
    for ii, fn in enumerate(filenames_audio):
        print('calculating', ii, fn, 'augmentation for', channel_string, 'in total', len(filenames_audio))
        try:
            y, sr = librosa.load(os.path.join(path_audio_origin, fn), sr=44100)
            for jj in [-2, -1, 1, 2, -3, 3]:
                y_shift = librosa.effects.pitch_shift(y, sr, n_steps=np.random.normal(jj,1.0))
                librosa.output.write_wav(os.path.join(path_audio_target, fn.split('.')[0]+'_'+str(jj)+'.wav'), y_shift, sr)
        except:
            print(ii, fn, 'augmentation error.')

def featureMWFDAugmentation(path_feature_origin, path_feature_target, channel):
    """
    multiple_width frequency-delta data augmentation
    Input feature should be [nFrames, nDims]
    :param filenames_wav:
    :param labels:
    :param path_feature_origin:
    :return:
    """
    # filenames_audio = [f for f in os.listdir(path_audio_origin) if os.path.isfile(os.path.join(path_audio_origin, f))]
    filenames_audio_train, _ = evalSetupTxtReader(path_fold1_train_txt)
    filenames_audio_eval, _ = evalSetupTxtReader(path_fold1_eval_txt)
    filenames_audio = filenames_audio_train + filenames_audio_eval

    for ii, fn in enumerate(filenames_audio):
        # print('MWFD augmentation for channel', channel, ii, fn, len(filenames_audio))
        name_path, name_file = os.path.split(fn)
        name_file_noext = name_file.split('.')[0]
        try:
            feature = pickle.load(open(os.path.join(path_feature_origin, name_file_noext + '.pkl'), 'r'))
            feature3 = Fdeltas(feature, w=3)
            feature11 = Fdeltas(feature, w=11)
            feature19 = Fdeltas(feature, w=19)

            # print(feature3.shape, feature11.shape, feature19.shape)

            pickle.dump(feature3, open(os.path.join(path_feature_target, name_file_noext + '_deltas3.pkl'), 'wb'))
            pickle.dump(feature11, open(os.path.join(path_feature_target, name_file_noext + '_deltas11.pkl'), 'wb'))
            pickle.dump(feature19, open(os.path.join(path_feature_target, name_file_noext + '_deltas19.pkl'), 'wb'))
        except:
            print(ii, fn, 'MWFD augmentation error.')
    return

if __name__ == '__main__':
    from pathName import *
    # batchPitchshiftAugmentation(path_audio_left, os.path.join(path_audio_augmentation,'left'),'left')
    # batchPitchshiftAugmentation(path_audio_right, os.path.join(path_audio_augmentation,'right'),'right')
    # batchPitchshiftAugmentation(path_audio_average, os.path.join(path_audio_augmentation,'average'),'average')
    # batchPitchshiftAugmentation(path_audio_difference, os.path.join(path_audio_augmentation,'difference'),'difference')
    for channel in ['left', 'right', 'average', 'difference']:
        featureMWFDAugmentation(os.path.join(path_feature_CNNs_logMel_96bands, channel),
                                os.path.join(path_feature_CNNs_logMel_96bands,'augmentation',channel),
                                channel)