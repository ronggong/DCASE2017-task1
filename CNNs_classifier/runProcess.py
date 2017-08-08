from pathName import *
from globalParams import *
from utilFunctions import evalSetupTxtReader, filenameExtRemover
from featurePreprocessing import FeaturePreprocessing
from modelTrainEval import ModelTrainEval
from CNNModels import modelVGG, modelMultiFilters, model2Layers1Filter, modelVGGJKU
from sklearn.model_selection import train_test_split
import pickle
from keras.models import load_model
import numpy as np

def calculateInputShape(full_filename, feature_seg):
    feature = pickle.load(open(full_filename,'r'))
    nFrames = feature.shape[0]
    nDims = feature.shape[1]
    return [int(nFrames/feature_seg), nDims, 1]


def filenameLabelSegmentation(X, y, seg=10):
    """
    duplicate and add number to the filename and label
    :param X: filename with extension
    :param y:
    :return:
    """
    X_seg = []
    y_seg = []
    for ii, fn in enumerate(X):
        filename_noext = filenameExtRemover(fn)
        for jj in range(seg):
            X_seg.append(filename_noext+'+'+str(jj))
            y_seg.append(y[ii])
    return X_seg, y_seg

def filenameLabelAugmentation(X, y):
    """
    filename and label augmentation
    :param X:
    :param y:
    :return:
    """
    X_aug = []
    y_aug = []
    for ii, fn in enumerate(X):
        filename_noext = filenameExtRemover(fn)
        for aug in ['-3','-2','-1','1','2','3','deltas3','deltas11','deltas19']:
            X_aug.append(filename_noext+'_'+aug)
            y_aug.append(y[ii])
    return X_aug, y_aug


mode = 'train'
for fold in ['fold1','fold2','fold3','fold4']:
    for channel in ['left','right','average','difference']:
        filename_base = fold + '_' + channel + '_vgg_jku_shuffle_bn_dropout'

        if feature_aug:
            filename_log = os.path.join(path_feature_CNNs_logMel_96bands_augmentation, 'log', filename_base + '.log')
            path_model = os.path.join(path_feature_CNNs_logMel_96bands_augmentation, 'models')
        else:
            filename_log = os.path.join(path_feature_CNNs_logMel_96bands, 'log', filename_base + '.log')
            path_model = path_feature_CNNs_logMel_96bands_models

        filename_model = os.path.join(path_model, filename_base + '.h5')

        filenames_wav_train_val, labels_train_val = evalSetupTxtReader(os.path.join(path_evaluation_setup, fold + '_train.txt'))

        # train validation set split
        filenames_wav_train, filenames_wav_val, labels_train, labels_val = train_test_split(filenames_wav_train_val,
                                                                                    labels_train_val,
                                                                                    test_size=0.1,
                                                                                    stratify=labels_train_val)

        # remove name extension
        filenames_train = [filenameExtRemover(fn) for fn in filenames_wav_train]
        filenames_val = [filenameExtRemover(fn) for fn in filenames_wav_val]

        # to have a full name
        filenames_train = [os.path.join(path_feature_CNNs_logMel_96bands, channel, fn + '.pkl') for fn in filenames_train]
        filenames_val = [os.path.join(path_feature_CNNs_logMel_96bands, channel, fn + '.pkl') for fn in filenames_val]

        # segment the train and validation filenames and labels
        # not to segment the feature if the feature is augmented
        if feature_aug:
            filenames_train_aug, labels_train_aug = filenameLabelAugmentation(filenames_wav_train, labels_train)

            # have full path name
            filenames_train_aug = [os.path.join(path_feature_CNNs_logMel_96bands_augmentation, channel, fn+'.pkl') for fn in filenames_train_aug]

            filenames_train = filenames_train + filenames_train_aug
            labels_train = labels_train + labels_train_aug

            featurePREPROCESSING = FeaturePreprocessing(fold, channel,
                                                        os.path.join(path_feature_CNNs_logMel_96bands_augmentation,
                                                                     'preprocessing'))
        else:

            featurePREPROCESSING = FeaturePreprocessing(fold, channel, path_feature_CNNs_logMel_96bands_preprocessing)

        # load feature scaler
        scaler = featurePREPROCESSING.featureScalerLoad()

        # load label encoder
        label_encoder = featurePREPROCESSING.labelEncoderLoad()

        # encode the labels
        labels_train = label_encoder.transform(labels_train)
        labels_val = label_encoder.transform(labels_val)


        # print(len(filenames_train))
        # print(labels_train)
        # print(len(filenames_val))
        # print(labels_val)

        input_shape = calculateInputShape(filenames_val[0], feature_seg)

        MODELTRAINEVAL = ModelTrainEval(path_model=path_model,
                                        scaler=scaler,
                                        file_size=64,
                                        preprocessing=True,
                                        input_shape=input_shape)

        if mode == 'train':

            # model = modelVGG(input_shape=input_shape,
            #                 filter_density=1)
            # input_shape = (43,96,1)
            model = modelVGGJKU(input_shape=input_shape,
                                filter_density=1)

            # model = modelMultiFilters(input_shape=input_shape)

            # model = model2Layers1Filter(input_shape=input_shape)

            model.summary()

            model = MODELTRAINEVAL.trainModel(model=model,
                                            filename_model=filename_model,
                                            filename_log=filename_log,
                                            filenames_train=filenames_train,
                                            labels_train=labels_train,
                                            filenames_val=filenames_val,
                                            labels_val=labels_val)

        else:
            print('start testing', fold, channel)

            filenames_wav_eval, labels_eval_old = evalSetupTxtReader(os.path.join(path_evaluation_setup, fold + '_evaluate.txt'))
            filenames_eval = [filenameExtRemover(fn) for fn in filenames_wav_eval]
            # print(filenames_eval)
            filenames_eval = [os.path.join(path_feature_CNNs_logMel_96bands, channel, fn + '.pkl') for fn in
                             filenames_eval]

            labels_eval_old = label_encoder.transform(labels_eval_old)

            model = load_model(filename_model)

            # filenames_eval = filenames_eval[:10]
            # labels_eval_old = labels_eval_old[:1]
            # print(filenames_eval)
            # for l in labels_eval_old:
            #     print(l)

            print('testing model', filename_base)
            pred_proba_eval, pred_eval = MODELTRAINEVAL.testModel(model,
                                                                filenames_eval,
                                                                file_size_test=1)
            accuracy = MODELTRAINEVAL.metricsEval(labels_eval_old, pred_eval)

            # filename_out = os.path.join(path_results, 'evaluation_set_accuracy.txt')
            # if os.path.isfile(filename_out):
            #     file = open(filename_out, 'a')
            # else:
            #     file = open(filename_out, 'w')
            #
            # file.write(fold +'_'+ channel+ '_' + str(accuracy) + '\n')
            # file.close()

            # dump the eval prediction proba
            pred_proba_eval = np.vstack(pred_proba_eval)
            print(pred_proba_eval.shape)
            results_cnns_dict = {}
            for ii, fn in enumerate(filenames_wav_val):
                pathname, fn_nopath = os.path.split(fn)
                results_cnns_dict[fn_nopath] = np.mean(pred_proba_eval[ii * feature_seg:(ii + 1) * feature_seg, :], axis=0)
            pickle.dump(results_cnns_dict,
                        open(os.path.join(path_results, 'y_pred_'+fold+'_'+channel+'.pkl'), 'wb'))
