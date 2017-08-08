from os import path
import numpy
import pandas as pd
import pickle
from pathName import *

def evalSetupTxtReader(filename_txt):
    """
    Read the .txt evaluation setup file
    :param filename_txt:
    :return:
    """
    list_filename_wav = []
    list_label = []
    with open(filename_txt) as f:
        for ii, line in enumerate(f):
            filename_wav, label = line.split()
            list_filename_wav.append(filename_wav)
            list_label.append(label)
    return list_filename_wav, list_label

def feature_preprocessing(path_feature, filenames_wav, labels=None):
    """
    Grab features in pandas csv, output numpy array x, y and dumped filenames
    :param path_feature:
    :param filenames_wav:
    :param labels:
    :return:
    """

    X = []
    y = []
    filenames_wav_dumped = []
    for ii, fn in enumerate(filenames_wav):
        print('combining feature', ii, fn, len(filenames_wav), 'in path', path_feature)
        name_path, name_file = path.split(fn)
        name_file_noext = name_file.split('.')[0]
        try:
            # data = pd.DataFrame.from_csv(os.path.join(path_feature, name_file_noext+'.csv'))
            data = pd.read_csv(os.path.join(path_feature, name_file_noext+'.csv'), index_col=0, engine='python')

            # delete unnecessary features
            name_columns = data.columns.values.tolist()
            name_columns_to_delete = []
            for name in name_columns:
                if 'histogram' in name or \
                    'onset_times' in name or \
                    'bpm_intervals' in name or \
                    'metadata' in name or \
                        'beats_position' in name:
                    name_columns_to_delete.append(name)

            # feature matrix, axis 0: observation, axis 1: feature dimension
            data = data.drop(name_columns_to_delete, axis=1)
            # print(type(data.values[0][0]))
            if data.values.shape[1] != 714 and data.values.shape[1] != 820:
                print (name_file_noext, data.values.shape, 'not a valid feature dimension.')
            elif str(data.values.dtype) == 'object':
                print(ii, fn, 'feature contains string')
            else:
                X.append(data.values)
                if labels is not None:
                    for jj in range(data.values.shape[0]):
                        y.append(labels[ii])
                filenames_wav_dumped.append(fn)
        except:
            print(ii, fn, 'feature csv not found')

    return X, y, filenames_wav_dumped

def dumpFeature(path_feature, filenames_wav, labels, path_output, name_pkl_output, name_filenames_wav_dumped=None):
    """
    Take the features .csv from path_feature, stack them, and dump it into a .pkl
    :param path_feature:
    :param filenames_wav:
    :param path_output:
    :param name_pkl_output:
    :return:
    """
    X, y, filenames_wav_dumped = feature_preprocessing(path_feature, filenames_wav, labels)
    X = numpy.vstack(X)
    # print(X.shape, len(labels))
    pickle.dump((X, y),
                open(os.path.join(path_output, name_pkl_output), 'wb'))
    if name_filenames_wav_dumped is not None:
        pickle.dump({'filenames_wav_dumped': filenames_wav_dumped, 'feature_seg':10},
                    open(os.path.join(path_output, name_filenames_wav_dumped), 'wb'))

def filenamesLabelsAugmentation(filenames_wav, labels):
    filenames_wav_aug = []
    labels_aug = []
    for ii, fn in enumerate(filenames_wav):
        print('Generate augmentation filenames and labels', ii, fn, len(filenames_wav))
        name_path, name_file = path.split(fn)
        name_file_noext = name_file.split('.')[0]
        for jj in [-2, -1, 1, 2, -3, 3]:
            filenames_wav_aug.append(os.path.join(name_path, name_file_noext+'_'+str(jj)+'.wav'))
            labels_aug.append(labels[ii])
    return filenames_wav_aug, labels_aug

def dumpFeatureAugmentation(path_feature_origin,
                            path_feature_augmentation,
                            filenames_wav,
                            labels,
                            path_output,
                            name_pkl_output):
    """
    Generate the augmentation filenames, labels, take features and dump them with original features
    :param path_feature_origin:
    :param path_feature_augmentation:
    :param filenames_wav:
    :param labels:
    :param path_output:
    :param name_pkl_output:
    :return:
    """
    # original features
    X, y, _ = feature_preprocessing(path_feature_origin, filenames_wav, labels)
    X = numpy.vstack(X)

    # augmented features
    filenames_wav_aug, labels_aug = filenamesLabelsAugmentation(filenames_wav, labels)
    X_aug, y_aug, _ = feature_preprocessing(path_feature_augmentation, filenames_wav_aug, labels_aug)
    X_aug = numpy.vstack(X_aug)

    X_all = numpy.vstack((X, X_aug))
    y_all = y+y_aug

    print(X_all.shape, len(y_all))

    pickle.dump((X_all, y_all),
                open(os.path.join(path_output, name_pkl_output), 'wb'))


def dumpFeatureBatch(path_feature_freesound_statistics, path_feature_freesound_statistics_folds):
    """
    Batch process to dump feature of each fold into pickle
    :return:
    """
    paths_fold_train_eval = [[path_fold1_train_txt, path_fold1_eval_txt],
                             [path_fold2_train_txt, path_fold2_eval_txt],
                             [path_fold3_train_txt, path_fold3_eval_txt],
                             [path_fold4_train_txt, path_fold4_eval_txt]]

    for ii, (path_fold_train_txt, path_fold_eval_txt) in enumerate(paths_fold_train_eval):
        filenames_wav_fold_train, labels_fold_train = evalSetupTxtReader(path_fold_train_txt)
        filenames_wav_fold_eval, labels_fold_eval = evalSetupTxtReader(path_fold_eval_txt)
        dumpFeature(path_feature_freesound_statistics,
                    filenames_wav_fold_train,
                    labels_fold_train,
                    path_feature_freesound_statistics_folds,
                    'fold'+str(ii+1)+'_train.pkl')
        dumpFeature(path_feature_freesound_statistics,
                    filenames_wav_fold_eval,
                    labels_fold_eval,
                    path_feature_freesound_statistics_folds,
                    'fold'+str(ii+1)+'_eval.pkl',
                    'fold'+str(ii+1)+'_dict_filenames_wav.pkl')

def dumpFeatureBatchUnseenEval(path_feature_freesound_statistics, path_feature_freesound_statistics_folds):
    """
    Batch process to dump feature of each fold into pickle
    :return:
    """

    filenames_feature_unseen_eval = [f for f in os.listdir(path_feature_freesound_statistics)
                              if os.path.isfile(os.path.join(path_feature_freesound_statistics, f))]

    dumpFeature(path_feature_freesound_statistics,
                filenames_feature_unseen_eval,
                None,
                path_feature_freesound_statistics_folds,
                'unseen_eval.pkl',
                'unssen_eval_dict_filenames_wav.pkl')

def dumpFeatureBatchAugmentation(path_feature_freesound_statistics,
                                 path_feature_freesound_statistics_augmentation,
                                 path_feature_freesound_statistics_folds_augmentation):
    """
    Batch process to dump feature of each fold into pickle
    :return:
    """
    paths_fold_train_eval = [[path_fold1_train_txt, path_fold1_eval_txt],
                             [path_fold2_train_txt, path_fold2_eval_txt],
                             [path_fold3_train_txt, path_fold3_eval_txt],
                             [path_fold4_train_txt, path_fold4_eval_txt]]

    for ii, (path_fold_train_txt, path_fold_eval_txt) in enumerate(paths_fold_train_eval):
        filenames_wav_fold_train, labels_fold_train = evalSetupTxtReader(path_fold_train_txt)
        dumpFeatureAugmentation(path_feature_freesound_statistics,
                                path_feature_freesound_statistics_augmentation,
                                filenames_wav_fold_train,
                                labels_fold_train,
                                path_feature_freesound_statistics_folds_augmentation,
                                'fold' + str(ii + 1) + '_train_aug.pkl')

if __name__ == '__main__':
    # dumpFeatureBatchAugmentation(path_feature_freesound_statistics_seg_left,
    #                              path_feature_freesound_statistics_seg_left_augmentation,
    #                              path_feature_freesound_statistics_folds_seg_left_augmentation)
    #
    # dumpFeatureBatchAugmentation(path_feature_freesound_statistics_seg_right,
    #                              path_feature_freesound_statistics_seg_right_augmentation,
    #                              path_feature_freesound_statistics_folds_seg_right_augmentation)
    #
    # dumpFeatureBatchAugmentation(path_feature_freesound_statistics_seg_average,
    #                              path_feature_freesound_statistics_seg_average_augmentation,
    #                              path_feature_freesound_statistics_folds_seg_average_augmentation)
    #
    # dumpFeatureBatchAugmentation(path_feature_freesound_statistics_seg_difference,
    #                              path_feature_freesound_statistics_seg_difference_augmentation,
    #                              path_feature_freesound_statistics_folds_seg_difference_augmentation)

    # # feature non augmented dump
    # dumpFeatureBatch(path_feature_freesound_statistics_seg_left, path_feature_freesound_statistics_folds_seg_left)
    # dumpFeatureBatch(path_feature_freesound_statistics_seg_right, path_feature_freesound_statistics_folds_seg_right)
    # dumpFeatureBatch(path_feature_freesound_statistics_seg_average, path_feature_freesound_statistics_folds_seg_average)
    # dumpFeatureBatch(path_feature_freesound_statistics_seg_difference, path_feature_freesound_statistics_folds_seg_difference)

    # evaluation set feature dump
    for channel in ['right', 'average', 'difference']:
        dumpFeatureBatchUnseenEval(os.path.join(path_dcase2017_evaluation,
                                                'feature',
                                                'freesound_extractor',
                                                'statistics_seg_'+channel),
                                   os.path.join(path_dcase2017_evaluation,
                                                'feature',
                                                'freesound_extractor',
                                                'statistics_pkl_seg_' + channel))