import pickle
import os
import numpy as np
from scipy.io import savemat, loadmat
from pathName import *
from xgb_classification import eval_metrics, labelEncoder_test
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from subprocess import call
import matplotlib.pyplot as plt
import itertools



path_results_GBM = './results_non_aug_lgb_overall' # lgb results
path_results_CNNs = '/Volumes/Rong Seagat/classifiers/Edu/results' # Edu CNNs results
# path_results_CNNs = '/Volumes/Rong Seagat/dcase/task1/TUT-acoustic-scenes-2017-development/feature/CNNs/log-mel/96bands/results'
# path_feature_GBM = '/Users/gong/Documents/MTG document/dataset/dcase/developmentSet' # development set
path_feature_GBM = '/Users/gong/Documents/MTG document/dataset/dcase/TUT-acoustic-scenes-2017-evaluation/feature/freesound_extractor'

path_submission = '../submission/'


channel = "average"
fold = "fold1"

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def rank(list, order='descend'):
    """
    the ranking of a numpy list,
    if in descend order, the output[i] indicates the descending rank of list[i] in the input list
    :param list:
    :param order:
    :return:
    """
    temp = list.argsort()
    ranks = np.empty(len(list), int)
    ranks[temp] = np.arange(len(list))
    if order == 'ascend':
        return ranks
    else: return len(list)-1-ranks

def loadData(fold, channel, path_results_GBM, path_feature_GBM, path_results_CNNs, load_y=True, unseenEval=False):

    if unseenEval:
        # for unseen evaluation set
        results_xgb = pickle.load(open(os.path.join(path_results_GBM, "predict_proba_unseen", "y_pred_" + fold + "_seg_" + channel + ".pkl"), "rb"))
        dict_filenames_wav = pickle.load(open(os.path.join(path_feature_GBM, "statistics_pkl_seg_" + channel, "unssen_eval_dict_filenames_wav.pkl"), "rb"))
        # _,y = pickle.load(open(os.path.join(path_feature_GBM, "statistics_pkl_seg_" + channel, "unseen_eval.pkl"), "rb"))
    else:
        results_xgb = pickle.load(open(os.path.join(path_results_GBM, "predict_proba", "y_pred_" + fold + "_seg_" + channel + ".pkl"), "rb"))
        dict_filenames_wav = pickle.load(open(os.path.join("./dict_filenames_wav", channel, fold+"_dict_filenames_wav.pkl"), "rb"))


    y_test = []
    if load_y:
        _,y = pickle.load(open(os.path.join(path_feature_GBM, "statistics_folds_seg_" + channel, fold + "_eval.pkl"), "rb"))
        y = labelEncoder_test(y, fold+'_seg', channel, path_classifier)
        # re-organize y
        for ii in range(len(y) / 10):
            y_test.append(y[ii * 10])

    # edu pickle
    if unseenEval:
        results_cnns = pickle.load(open(os.path.join(path_results_CNNs, 'rec_probs_fold0.pickle'), "rb"))  # edu
    else:
        results_cnns = pickle.load(open(os.path.join(path_results_CNNs, 'rec_probs_'+fold+'.pickle'), "rb")) # edu

    # results_cnns = pickle.load(open(os.path.join(path_results_CNNs, 'y_pred_'+fold+'_'+channel+'.pkl'), "rb"))


    # print(len(results_cnns))
    # re-organize xgboost results
    results_xgb_dict = {}
    for ii, fn in enumerate(dict_filenames_wav['filenames_wav_dumped']):
        _, fn_nopath = os.path.split(fn)
        results_xgb_dict[fn_nopath] = np.mean(results_xgb[ii*10:(ii+1)*10, :], axis=0)
    # print(len(results_xgb_dict))

    return results_xgb_dict, dict_filenames_wav, y_test, results_cnns

def score_fusion(fold, channel, path_results_GBM,
                 path_feature_GBM, path_results_CNNs,
                 load_y=False, unseenEval=True, mode='mean'):
    """
    Fuse the xgboost and CNNs scores with the mode - mean, product or rank
    :param fold:
    :param channel:
    :param mode:
    :return:
    """

    results_xgb, \
    dict_filenames_wav, \
    y_test, \
    results_cnns = loadData(fold, channel,
                             path_results_GBM,
                             path_feature_GBM,
                             path_results_CNNs,
                            load_y=load_y,
                            unseenEval=unseenEval)

    # fuse the scores
    results_fused = {}
    for fn in results_cnns:
        if unseenEval:
            fn = fn.split('.')[0]
            fn_xgb = fn+'.csv'
            fn_cnns = fn+'.wav'
        else:
            fn_xgb, fn_cnns = fn, fn
        try:
            if mode == 'mean':
                results_fused[fn_cnns] = np.argmax((results_xgb[fn_xgb] + results_cnns[fn_cnns])/2.0)
            elif mode == 'product':
                results_fused[fn_cnns] = np.argmax(results_xgb[fn_xgb]*results_cnns[fn_cnns])
            elif mode == 'rank':
                results_fused[fn_cnns] = np.argmin(rank(results_xgb[fn_xgb]) + rank(results_cnns[fn_cnns]))
        except KeyError:
            print(fn, 'not found')

    # resort the prediction according to the filenames
    predict_resort = []
    predict_xgb = []
    predict_cnns = []
    for fn in dict_filenames_wav['filenames_wav_dumped']:
        _, fn_nopath = os.path.split(fn)
        if unseenEval:
            fn_wav = fn_nopath.split('.')[0]+'.wav'
            fn_csv = fn_nopath.split('.')[0] +'.csv'
        else:
            fn_wav, fn_csv = fn_nopath, fn_nopath

        predict_resort.append(results_fused[fn_wav])
        predict_xgb.append(np.argmax(results_xgb[fn_csv]))
        predict_cnns.append(np.argmax(results_cnns[fn_wav]))

    if load_y:
        # print predict_resort
        accuracy= eval_metrics(predict_resort, y_test)
        # print results_xgb.shape
    else:
        accuracy = None

    return accuracy, y_test, predict_resort, predict_xgb, predict_cnns, dict_filenames_wav['filenames_wav_dumped']

def prepareData4Multifocal(fold, channel, filenames_train, filenames_test, y_train):

    results_xgb, dict_filenames_wav, y, results_cnns = loadData(fold, channel)


    # filenames = dict_filenames_wav_min['filenames_wav_dumped']


    results_xgb_train = []
    results_cnns_train = []
    results_xgb_test = []
    results_cnns_test = []
    for fn in filenames_train:
        _, fn_nopath = os.path.split(fn)
        results_xgb_train.append(results_xgb[fn_nopath])
        results_cnns_train.append(results_cnns[fn_nopath.split('.')[0]+'.pkl'])
    for fn in filenames_test:
        _, fn_nopath = os.path.split(fn)
        results_xgb_test.append(results_xgb[fn_nopath])
        results_cnns_test.append(results_cnns[fn_nopath.split('.')[0]+'.pkl'])

    mat_2_save = {'xgb_train': results_xgb_train,
                  'xgb_test': results_xgb_test,
                  'cnns_train': results_cnns_train,
                  'cnns_test': results_cnns_test,
                  'y_train': y_train}

    savemat('./prediction2fuse_multifocal/results_rong_test_'+fold+'_'+channel, mat_2_save)

def prepareData4MultifocalTrain(fold, channel, filenames_train, y_train,
                                path_results_GBM, path_feature_GBM, path_results_CNNs):

    results_xgb, dict_filenames_wav, y, results_cnns = loadData(fold, channel,
                                                                path_results_GBM,
                                                                path_feature_GBM,
                                                                path_results_CNNs)


    # filenames = dict_filenames_wav_min['filenames_wav_dumped']

    results_xgb_train = []
    results_cnns_train = []
    for fn in filenames_train:
        _, fn_nopath = os.path.split(fn)
        results_xgb_train.append(results_xgb[fn_nopath])
        results_cnns_train.append(results_cnns[fn_nopath.split('.')[0]+'.pkl'])

    mat_2_save = {'xgb_train': results_xgb_train,
                  'cnns_train': results_cnns_train,
                  'y_train': y_train}

    savemat('./prediction2fuse_multifocal/results_rong_development_train_'+fold+'_'+channel, mat_2_save)

def prepareData4MultifocalUnseenEval(fold, channel, filenames, path_results_GBM, path_feature_GBM, path_results_CNNs):

    results_xgb, \
    dict_filenames_wav, \
    y, results_cnns = loadData(fold, channel, path_results_GBM,
                                path_feature_GBM, path_results_CNNs,
                               load_y=False, unseenEval=True)
    results_xgb_test = []
    results_cnns_test = []
    for fn in filenames:
        _, fn_nopath = os.path.split(fn)
        results_xgb_test.append(results_xgb[fn_nopath])
        results_cnns_test.append(results_cnns[fn_nopath.split('.')[0]+'.pkl'])

    mat_2_save = {'xgb_test': results_xgb_test,
                  'cnns_test': results_cnns_test}

    savemat('./prediction2fuse_multifocal/results_rong_evaluation_unseen_' + fold + '_' + channel, mat_2_save)


def score_fusion_multifocal(fold, string=''):
    """
    load y_pred fusion, then evaluate it.
    :param fold:
    :param channel:
    :return:
    """
    y_fusion = loadmat('./prediction2fuse_multifocal/y_fusion_rong_test_'+string+fold+'.mat')
    y_pred = y_fusion['y_fusion'][0]

    filenamesLabel = pickle.load(open('./prediction2fuse_multifocal/filenamesLabel_rong_test_'+fold+'.pkl', 'rb'))
    y_test = filenamesLabel['y_test']


    return y_pred, y_test

if __name__ == '__main__':

    from functools import reduce
    ###### edu's system evaluation

    # class_name = ['beach', 'bus', 'cafe/restaurant', 'car', 'city center',
    #    'forest path', 'grocery store', 'home', 'library', 'metro station',
    #    'office', 'park', 'residential area', 'train', 'tram']
    # list_acc = []
    # y_test_all, y_pred_all = [], []
    # for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
    #     accuracy, y_test, y_pred, predict_xgb, predict_cnns, filenames = score_fusion(fold, channel, path_results_GBM,
    #                                             path_feature_GBM, path_results_CNNs,
    #                                             load_y=True, unseenEval=False)
    #     list_acc.append(accuracy)
    #     y_test_all = y_test_all + y_test
    #     y_pred_all = y_pred_all + y_pred
    # print np.mean(list_acc)
    # cnf_matrix = confusion_matrix(y_test_all, y_pred_all)
    # acc_class = cnf_matrix.diagonal()/np.sum(cnf_matrix, axis=1, dtype=np.float32)
    # print(acc_class)


    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_name)
    # plt.show()


    ###### edu's system unseen
    # le = pickle.load(open(os.path.join(path_classifier, "le_fold1_seg_average.pkl"), "r"))
    # print(le.inverse_transform(range(15)))
    # accuracy, y_test, \
    # predict_resort, predict_gbm, predict_cnns, filenames = score_fusion('fold1', 'average', path_results_GBM,
    #                                         path_feature_GBM, path_results_CNNs,
    #                                          load_y=False, unseenEval=True)
    #
    # print(len(predict_resort))
    # sum_overlapped = 0
    # for ii in range(len(predict_resort)):
    #     print(predict_resort[ii], predict_gbm[ii], predict_cnns[ii])
    #     if predict_resort[ii] == predict_cnns[ii]:
    #         sum_overlapped += 1
    #
    # with open(os.path.join(path_submission, 'edu', 'Fusion.txt'), 'wb') as f:
    #     for ii, fn in enumerate(filenames):
    #         fn = 'audio/'+fn.split('.')[0]+'.wav'
    #         f.write(fn+"\t"+le.inverse_transform(predict_resort[ii])+"\n")

    ##### on the development test set, multiFocal
    """
    # fusion step 1
    for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
        print(fold)
        results_xgb_left, dict_filenames_wav_left, y_left, results_cnns_left = loadData(fold, 'left')
        results_xgb_right, dict_filenames_wav_right, y_right, results_cnns_right = loadData(fold, 'right')
        results_xgb_average, dict_filenames_wav_average, y_average, results_cnns_average = loadData(fold, 'average')
        results_xgb_difference, dict_filenames_wav_difference, y_difference, results_cnns_difference = loadData(fold,
                                                                                                                'difference')

        filenames_left = dict_filenames_wav_left['filenames_wav_dumped']
        filenames_right = dict_filenames_wav_right['filenames_wav_dumped']
        filenames_average = dict_filenames_wav_average['filenames_wav_dumped']
        filenames_difference = dict_filenames_wav_difference['filenames_wav_dumped']

        print(len(filenames_left), len(filenames_right), len(filenames_average), len(filenames_difference))
        print(len(results_cnns_left), len(results_cnns_right), len(results_cnns_average), len(results_cnns_difference))
        print(len(y_left), len(y_right), len(y_average), len(y_difference))


        # because filenames are different, we need to search for their intersection
        filenames_intersect = reduce(np.intersect1d,
                                     (filenames_left, filenames_right, filenames_average, filenames_difference))
        y_intersect = [y_left[filenames_left.index(fn)] for fn in filenames_intersect]

        filenames_train, filenames_test, y_train, y_test = train_test_split(filenames_intersect, y_intersect, test_size=0.5,
                                                                            stratify=y_intersect)

        filenamesLabel_2_save = {'filenames_train': filenames_train,
                                 'filenames_test': filenames_test,
                                 'y_train': y_train,
                                 'y_test': y_test}

        pickle.dump(filenamesLabel_2_save,
                    open('./prediction2fuse_multifocal/filenamesLabel_rong_test_' + fold + '.pkl',
                         'wb'))

        for channel in ['left', 'right', 'average', 'difference']:
            prepareData4Multifocal(fold, channel, filenames_train, filenames_test, y_train)

    # step 2: calculate the fusion prediction
    call(['/Applications/MATLAB_R2013a.app/bin/matlab', '-nodisplay', '-r', 'multifocal_fusion_multichannel;quit;'])

    """

    # step 3: evaluate the prediction

    print('all')
    y_test_all = []
    y_pred_all = []
    for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
        y_pred, y_test = score_fusion_multifocal(fold)
        y_pred_all = y_pred_all+ y_pred.tolist()
        y_test_all = y_test_all+ y_test

    acc = eval_metrics(y_test_all, y_pred_all)
    cmat = confusion_matrix(y_test_all, y_pred_all)
    acc_class = cmat.diagonal() / np.sum(cmat, axis=1, dtype=np.float32)
    print(acc)
    print(acc_class)

    print('gbm')
    y_test_all = []
    y_pred_all = []
    for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
        y_pred, y_test = score_fusion_multifocal(fold, 'gbm')
        y_pred_all = y_pred_all+ y_pred.tolist()
        y_test_all = y_test_all + y_test

    acc = eval_metrics(y_test_all, y_pred_all)
    cmat = confusion_matrix(y_test_all, y_pred_all)
    acc_class = cmat.diagonal() / np.sum(cmat, axis=1, dtype=np.float32)
    print(acc)
    print(acc_class)

    print('cnns')
    y_test_all = []
    y_pred_all = []
    for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
        y_pred, y_test = score_fusion_multifocal(fold, 'cnns')
        y_pred_all = y_pred_all+ y_pred.tolist()
        y_test_all = y_test_all + y_test

    acc = eval_metrics(y_test_all, y_pred_all)
    cmat = confusion_matrix(y_test_all, y_pred_all)
    acc_class = cmat.diagonal() / np.sum(cmat, axis=1, dtype=np.float32)
    print(acc)
    print(acc_class)




    #### on the development test set for fit the multiFocal model, unseen eval set for outputting the prediction score
    # path_results_GBM_development_and_unseenEval = './results_non_aug_lgb'
    # path_feature_GBM_development = '/Users/gong/Documents/MTG document/dataset/dcase/developmentSet'  # development set
    # path_feature_GBM_unseenEval = '/Users/gong/Documents/MTG document/dataset/dcase/TUT-acoustic-scenes-2017-evaluation/feature/freesound_extractor'
    # path_results_CNNs_developemnt = '/Volumes/Rong Seagat/dcase/task1/TUT-acoustic-scenes-2017-development/feature/CNNs/log-mel/96bands/results'
    # path_results_CNNs_unseenEval = '/Volumes/Rong Seagat/dcase/task1/TUT-acoustic-scenes-2017-development/feature/CNNs/log-mel/96bands/unseenEval/results'
    # dict_filenames_wav_unseenEval = pickle.load(open(os.path.join(path_feature_GBM_unseenEval, "statistics_pkl_seg_left", "unssen_eval_dict_filenames_wav.pkl"), "rb"))
    # filenames_unseenEval = dict_filenames_wav_unseenEval['filenames_wav_dumped']
    # le = pickle.load(open(os.path.join(path_classifier, "le_" + fold + "_seg_" + channel + ".pkl"), "r"))

    # step 1: prepare data for the multiFocal train
    # for fold in ['fold1', 'fold2', 'fold3', 'fold4']:
    #     print(fold)
        # results_xgb_left, \
        # dict_filenames_wav_left, \
        # y_left, \
        # results_cnns_left = loadData(fold, 'left',
        #                             path_results_GBM_development_and_unseenEval,
        #                             path_feature_GBM_development,
        #                             path_results_CNNs_developemnt)
        # results_xgb_right, \
        # dict_filenames_wav_right,\
        # y_right, \
        # results_cnns_right = loadData(fold, 'right',
        #                                 path_results_GBM_development_and_unseenEval,
        #                                 path_feature_GBM_development,
        #                                 path_results_CNNs_developemnt)
        # results_xgb_average, \
        # dict_filenames_wav_average, \
        # y_average, \
        # results_cnns_average = loadData(fold, 'average',
        #                                 path_results_GBM_development_and_unseenEval,
        #                                 path_feature_GBM_development,
        #                                 path_results_CNNs_developemnt)
        # results_xgb_difference, \
        # dict_filenames_wav_difference, \
        # y_difference, \
        # results_cnns_difference = loadData(fold,
        #                                 'difference',
        #                                 path_results_GBM_development_and_unseenEval,
        #                                 path_feature_GBM_development,
        #                                 path_results_CNNs_developemnt)
        #
        # filenames_left = dict_filenames_wav_left['filenames_wav_dumped']
        # filenames_right = dict_filenames_wav_right['filenames_wav_dumped']
        # filenames_average = dict_filenames_wav_average['filenames_wav_dumped']
        # filenames_difference = dict_filenames_wav_difference['filenames_wav_dumped']
        #
        # # print(len(filenames_left), len(filenames_right), len(filenames_average), len(filenames_difference))
        # # print(len(results_cnns_left), len(results_cnns_right), len(results_cnns_average), len(results_cnns_difference))
        # # print(len(y_left), len(y_right), len(y_average), len(y_difference))
        #
        # # because filenames are different, we need to search for their intersection
        # filenames_intersect = reduce(np.intersect1d,
        #                              (filenames_left, filenames_right, filenames_average, filenames_difference))
        # y_intersect = [y_left[filenames_left.index(fn)] for fn in filenames_intersect]
        #
        # for channel in ['left', 'right', 'average', 'difference']:
        #     prepareData4MultifocalTrain(fold, channel, filenames_intersect, y_intersect,
        #                                 path_results_GBM_development_and_unseenEval,
        #                                 path_feature_GBM_development,
        #                                 path_results_CNNs_developemnt)

        #### step 2: prepare data for multiFocal unseen Eval
        #
        # for channel in ['left', 'right', 'average', 'difference']:
        #     prepareData4MultifocalUnseenEval(fold, channel,
        #                                      filenames_unseenEval,
        #                                      path_results_GBM_development_and_unseenEval,
        #                                      path_feature_GBM_unseenEval,
        #                                      path_results_CNNs_unseenEval)

    #### step 3: train the multiFocal model and predict on unseen Eval
    # call(['/Applications/MATLAB_R2013a.app/bin/matlab', '-nodisplay', '-r', 'multifocal_fusion_multichannel_unseenEval;quit;'])

    # y_fusion = loadmat('./prediction2fuse_multifocal/y_fusion_rong_evaluation_unseen_gbmfold.mat')
    # y_pred = y_fusion['y_fusion'][0]
    #
    # sum_iden = 0
    # for ii in range(1620):
    #     if predict_gbm[ii] == y_pred[ii]:
    #         sum_iden += 1
    # print(sum_iden)

    # with open(os.path.join(path_submission, 'multiFocal', 'Fusion.txt'), 'wb') as f:
    #     for ii, fn in enumerate(filenames_unseenEval):
    #         fn = 'audio/'+fn.split('.')[0]+'.wav'
    #         f.write(fn+"\t"+le.inverse_transform(y_pred[ii])+"\n")
