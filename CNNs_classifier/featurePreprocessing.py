from sklearn import preprocessing
import pickle
import os

class FeaturePreprocessing(object):


    def __init__(self, fold_string, channel_string, path_preprocessing):
        self.fold_string = fold_string
        self.channel_string = channel_string
        self.path_preprocessing = path_preprocessing

    def featureScalingTrain(self, X):
        scaler = preprocessing.StandardScaler()
        scaler.fit(X)
        pickle.dump(scaler,
                    open(os.path.join(self.path_preprocessing,
                                      'feature_scaler_' +
                                      self.fold_string + '_' +
                                      self.channel_string + '.pkl'), 'wb'))
        return

    def featureScalerLoad(self):
        scaler = pickle.load(open(os.path.join(self.path_preprocessing,
                                               'feature_scaler_' +
                                               self.fold_string + '_' +
                                               self.channel_string + '.pkl'), 'r'))
        return scaler

    def imputerLabelEncoderTrain(self, X, y):
        """
        Imputer is a tool to fix the NaN in feature X
        :param X:
        :param y:
        :return:
        """
        imputer = preprocessing.Imputer()
        X = imputer.fit_transform(X)

        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        pickle.dump(imputer, open(os.path.join(self.path_preprocessing,
                                               'feature_imputer_' +
                                               self.fold_string + '_' +
                                               self.channel_string + '.pkl'), 'wb'))
        pickle.dump(le, open(os.path.join(self.path_preprocessing,
                                          'le_' +
                                          self.fold_string + '_' +
                                          self.channel_string + '.pkl'), 'wb'))

        return X,y,imputer,le

    def imputerLoad(self):
        imputer = pickle.load(open(os.path.join(self.path_preprocessing,
                                                "feature_imputer_" +
                                                self.fold_string + "_" +
                                                self.channel_string + ".pkl"), 'r'))
        return imputer

    def labelEncoderLoad(self):
        le = pickle.load(open(os.path.join(self.path_preprocessing,
                                           "le_" +
                                           self.fold_string + "_" +
                                           self.channel_string + ".pkl"), "r"))
        return le

if __name__ == "__main__":
    from pathName import *
    from utilFunctions import evalSetupTxtReader
    import numpy as np

    def concatenateFeature(filenames_wav, labels, path_feature, augmentation=False):
        """
        Concatenate features from [nFrames, nDims] into [N*nFrame, nDims]
        Input feature should be [nFrames, nDims]
        :param filenames_wav:
        :param labels:
        :param path_feature:
        :return:
        """
        X = []
        y = []
        for ii, fn in enumerate(filenames_wav):
            print('combining feature', ii, fn, len(filenames_wav))
            name_path, name_file = os.path.split(fn)
            name_file_noext = name_file.split('.')[0]
            if augmentation:
                for aug in ['-1', '-2', '-3', '1', '2', '3', 'deltas3', 'deltas11', 'deltas19']:
                    try:
                        feature = pickle.load(open(os.path.join(path_feature,name_file_noext + '_' + aug +'.pkl'), 'r'))
                        X.append(feature)
                        y.append(labels[ii])
                    except:
                        print(ii, fn, aug, 'load feature error.')
            else:
                try:
                    feature = pickle.load(open(os.path.join(path_feature,name_file_noext+'.pkl'), 'r'))
                    X.append(feature)
                    y.append(labels[ii])
                except:
                    print(ii, fn, 'load feature error.')
        X = np.array(np.vstack(X), dtype='float32')
        return X, y

    # save scaler, imputer, labelEncoder for each fold and each channel
    for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
        for channel in ['left', 'right', 'average', 'difference']:

            filenames_wav_train, labels_train = evalSetupTxtReader(os.path.join(path_evaluation_setup, fold+'_train.txt'))

            # features of the original audios
            path_feature_channel_origin = os.path.join(path_feature_CNNs_logMel_96bands, channel)
            X_origin, y_origin = concatenateFeature(filenames_wav_train, labels_train, path_feature_channel_origin)

            # features of the augmented audios
            path_feature_channel_aug = os.path.join(path_feature_CNNs_logMel_96bands, channel)
            X_aug, y_aug = concatenateFeature(filenames_wav_train, labels_train, path_feature_channel_aug, True)

            # concatenate the original features and augmented features
            X_all = np.vstack((X_origin, X_aug))
            y_all = y_origin + y_aug

            featurePREPROCESSING = FeaturePreprocessing(fold, channel, path_feature_CNNs_logMel_96bands_preprocessing)

            print(X_all.shape)
            featurePREPROCESSING.featureScalingTrain(X_all)
            featurePREPROCESSING.imputerLabelEncoderTrain(X_all, y_all)