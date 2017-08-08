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

def calculateInputShape(full_filename):
    feature = pickle.load(open(full_filename,'r'))
    nFrames = feature.shape[0]
    nDims = feature.shape[1]
    return [nFrames, nDims, 1]


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

def predictionProbaDumper(pred_proba, filenames_wav, path_results, fold, channel):
    # dump the eval prediction proba
    pred_proba = np.vstack(pred_proba)
    # print(pred_proba_eval.shape)
    results_cnns_dict = {}
    for ii, fn in enumerate(filenames_wav):
        pathname, fn_nopath = os.path.split(fn)
        results_cnns_dict[fn_nopath] = np.mean(pred_proba[ii * feature_seg:(ii + 1) * feature_seg, :], axis=0)
    pickle.dump(results_cnns_dict,
                open(os.path.join(path_results, 'y_pred_' + fold + '_' + channel + '.pkl'), 'wb'))


dict_epochs = {'fold1left': 36, 'fold1right': 17, 'fold1average': 19, 'fold1difference': 33,
               'fold2left': 28, 'fold2right': 18, 'fold2average': 30, 'fold2difference': 31,
               'fold3left': 19, 'fold3right': 19, 'fold3average': 37, 'fold3difference': 32,
               'fold4left': 40, 'fold4right': 24, 'fold4average': 13, 'fold4difference': 46}

mode = 'eval'
for fold in ['fold1','fold2','fold3','fold4']:
    for channel in ['left','right','average','difference']:
        filename_base = fold + '_' + channel + '_vgg_jku_shuffle_bn_dropout_split_first'


        filename_log = os.path.join(path_feature_CNNs_logMel_96bands, 'log', filename_base + '.log')
        path_model = path_feature_CNNs_logMel_96bands_models

        filename_model = os.path.join(path_model, filename_base + '.h5')

        filenames_wav_train_val, labels_train_val = evalSetupTxtReader(os.path.join(path_evaluation_setup, fold + '_train.txt'))

        # train validation set split
        filenames_wav_train, filenames_wav_val, labels_train, labels_val = train_test_split(filenames_wav_train_val,
                                                                                    labels_train_val,
                                                                                    test_size=0.1,
                                                                                    stratify=labels_train_val)

        # # remove name extension
        # filenames_train = [filenameExtRemover(fn) for fn in filenames_wav_train]
        # filenames_val = [filenameExtRemover(fn) for fn in filenames_wav_val]
        filenames_train, labels_train = filenameLabelSegmentation(filenames_wav_train, labels_train)
        filenames_val, labels_val = filenameLabelSegmentation(filenames_wav_val, labels_val)

        # to have a full name
        filenames_train = [os.path.join(path_feature_CNNs_logMel_96bands, channel, fn + '.pkl') for fn in filenames_train]
        filenames_val = [os.path.join(path_feature_CNNs_logMel_96bands, channel, fn + '.pkl') for fn in filenames_val]

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

        input_shape = calculateInputShape(filenames_val[0])

        MODELTRAINEVAL = ModelTrainEval(path_model=path_model,
                                        scaler=scaler,
                                        file_size=30,
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

            # model.summary()

            model = MODELTRAINEVAL.trainModel(model=model,
                                            filename_model=filename_model,
                                            filename_log=filename_log,
                                            filenames_train=filenames_train,
                                            labels_train=labels_train,
                                            filenames_val=filenames_val,
                                            labels_val=labels_val,
                                              epochs=dict_epochs[fold+channel])
        elif mode == 'eval':
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

            filename_out = os.path.join(path_results, 'evaluation_set_accuracy_vgg_jku_split_first.txt')
            if os.path.isfile(filename_out):
                file = open(filename_out, 'a')
            else:
                file = open(filename_out, 'w')

            file.write(fold +'_'+ channel+ '_' + str(accuracy) + '\n')
            file.close()

            # dump the eval prediction proba
            predictionProbaDumper(pred_proba_eval,filenames_eval,path_results,fold,channel)

        else:

            filenames_wav_test, _ = evalSetupTxtReader(
                os.path.join(path_evaluation_setup, 'test.txt'), tosplit=False)
            filenames_test = [filenameExtRemover(fn) for fn in filenames_wav_test]
            # print(filenames_eval)
            filenames_test = [os.path.join(path_feature_CNNs_logMel_96bands_unseenEval, channel, fn + '.pkl') for fn in
                              filenames_test]

            model = load_model(filename_model)

            print('testing model', filename_base)
            pred_proba_test, pred_test = MODELTRAINEVAL.testModel(model,
                                                                  filenames_test,
                                                                  file_size_test=1)
            predictionProbaDumper(pred_proba_test,filenames_test,
                                  os.path.join(path_feature_CNNs_logMel_96bands_unseenEval,'results'),
                                  fold, channel)


