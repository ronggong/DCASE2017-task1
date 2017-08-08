
import pandas as pd
import time
import numpy
import pickle
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint
from sklearn.cross_validation import StratifiedKFold, train_test_split
import xgboost as xgb

from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from pathName import *
# from utils import visualization


# IRMAS - 30
THRESHOLD = 20

labels = None



def loadFeature(path_feature, name_pkl):
    """
    load feature pickle
    :param path_feature:
    :param name_pkl:
    :return:
    """
    X, y = pickle.load(open(os.path.join(path_feature, name_pkl)))
    return X, y


def feature_scaling_train(X, dim_string, channel_string, scaler_path):
    scaler = preprocessing.StandardScaler()
    scaler.fit(X)
    pickle.dump(scaler, open(path.join(scaler_path, 'feature_scaler_'+dim_string+'_'+channel_string+'.pkl'), 'wb'))
    # X = scaler.transform(X)
    return

def feature_scaling_test(X, dim_string, channel_string, scaler_path):
    scaler = pickle.load(open(path.join(scaler_path,'feature_scaler_'+dim_string+'_'+channel_string+'.pkl'),'r'))
    X = scaler.transform(X)

    return X

def buildEstimators(mode, dim_string, channel_string, best_params, path_classifier):
    if mode == 'train' or mode == 'cv':
        # best parameters got by gridsearchCV, best score: 1
        estimators = [('anova_filter', SelectKBest(f_classif, k=best_params['anova_filter__k'])),
                      ('xgb', xgb.XGBClassifier(learning_rate=best_params['xgb__learning_rate'],
                                                n_estimators=best_params['xgb__n_estimators'],
                                                max_depth=best_params['xgb__max_depth'],
                                                objective=best_params['xgb__objective'],
                                                n_jobs=-1))]
        clf = Pipeline(estimators)
    elif mode == 'test':
        clf = pickle.load(open(path.join(path_classifier,"xgb_classifier_"+dim_string+"_"+channel_string+".pkl"), "r"))
    return clf

def imputerLabelEncoder_train(X,y,dim_string, channel_string, scaler_path):
    """
    Imputer is a tool to fix the NaN in feature X
    :param X:
    :param y:
    :return:
    """
    imputer = preprocessing.Imputer()
    imputer.fit(X)

    le = preprocessing.LabelEncoder()
    le.fit(y)

    pickle.dump(imputer, open(path.join(scaler_path, 'feature_imputer_'+dim_string+'_'+channel_string+'.pkl'), 'wb'))
    pickle.dump(le, open(path.join(scaler_path, 'le_'+dim_string+'_'+channel_string+'.pkl'), 'wb'))

    return imputer,le

def imputer_test(X, dim_string, channel_string):
    imputer = pickle.load(open(path.join(path_classifier, "feature_imputer_" + dim_string + "_" + channel_string + ".pkl"), 'r'))
    X = imputer.transform(X)
    return X

def labelEncoder_test(y,dim_string,channel_string, scaler_path):
    le = pickle.load(open(path.join(scaler_path,"le_"+dim_string+"_"+channel_string+".pkl"), "r"))
    y = le.transform(y)
    return y

def save_results(y_test, y_pred, labels, fold_number=0):
    pickle.dump(y_test, open("y_test_fold{number}.plk".format(number=fold_number), "w"))
    pickle.dump(y_pred, open("y_pred_fold{number}.plk".format(number=fold_number), "w"))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("Micro stats:")
    print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
    print("Macro stats:")
    print(precision_recall_fscore_support(y_test, y_pred, average='macro'))
    # try:
    #     visualization.plot_confusion_matrix(confusion_matrix(y_test, y_pred),
    #                                         title="Test CM fold{number}".format(number=fold_number),
    #                                         labels=labels)
    # except:
    #     pass


def train_test(clf, X, y, labels):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    save_results(y_test, y_pred, labels)

def train_evaluate_stratified(clf, X, y, labels):
    skf = StratifiedKFold(y, n_folds=10)
    for fold_number, (train_index, test_index) in enumerate(skf):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        save_results(y_test, y_pred, labels, fold_number)


def grid_search(clf, X, y):
    # params = dict(anova_filter__k=['all'],
    #     # anova_filter__k=[50, 100, 'all'],
    #               xgb__max_depth=[3, 5, 10], xgb__n_estimators=[50, 100, 300, 500],
    #               xgb__learning_rate=[0.05, 0.1])
    n_iter_search = 20
    params = dict(anova_filter__k=[250, 500, 'all'],
                  # anova_filter__k=[50, 100, 'all'],
                  xgb__max_depth=sp_randint(3, 10), xgb__n_estimators=sp_randint(300, 500),
                  xgb__learning_rate=[0.05, 0.01],
                  xgb__objective=['multi:softprob'])
    # gs = GridSearchCV(clf, param_grid=params, n_jobs=4, cv=5, verbose=2)
    gs = RandomizedSearchCV(clf, param_distributions=params,n_jobs=4,cv=5,n_iter=n_iter_search,verbose=2)
    gs.fit(X, y)

    print "Best estimator:"
    print gs.best_estimator_
    print "Best parameters:"
    print gs.best_params_
    print "Best score:"
    print gs.best_score_

    y_pred = gs.predict(X)
    y_test = y

    return gs.best_params_

def train_save(clf, X, y,dim_string,channel_string):
    clf.fit(X, y)
    pickle.dump(clf, open(path.join(path_classifier,"xgb_classifier_"+dim_string+"_"+channel_string+".pkl"), "w"))

def prediction(clf, X, y, mode_eval='normal', seg = 10):
    if mode_eval =='normal':
        y_pred = clf.predict(X)
        y_test = y
    elif mode_eval == 'major_vote':
        y_pred = []
        y_test = []
        y_pred_seg = clf.predict(X)
        ii = 0
        while ii < len(y_pred_seg)/seg:
            y_pred.append(numpy.bincount(y_pred_seg[ii*seg:(ii+1)*seg]).argmax())
            y_test.append(y[ii*seg])
            ii += 1
    elif mode_eval == 'mean':
        y_pred_proba = clf.predict_proba(X)
        # print y_pred_proba.shape
        # y_pred_proba = numpy.reshape(y_pred_proba, (len(y_pred_proba)/15, 15))
        y_pred = []
        y_test = []
        ii = 0
        while ii < y_pred_proba.shape[0] / seg:
            y_pred.append(numpy.argmax(numpy.sum(y_pred_proba[ii*seg:(ii+1)*seg, :], axis=0)))
            y_test.append(y[ii*seg])
            ii += 1
    return y_test, y_pred

def eval_metrics(y_test, y_pred):
    # print(classification_report(y_test, y_pred))
    # print confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:")
    print(accuracy)
    # print("Micro stats:")
    # print(precision_recall_fscore_support(y_test, y_pred, average='micro'))
    # print("Macro stats:")
    # print(precision_recall_fscore_support(y_test, y_pred, average='macro'))

    # print(y_pred)
    return accuracy

def trainTestProcess(X, y, mode, dim_string, channel_string, mode_eval='normal'):

    X[numpy.isinf(X)] = numpy.iinfo('i').max

    if mode == 'train' or mode == 'cv':
        feature_scaling_train(X, dim_string, channel_string, path_classifier)

    X = feature_scaling_test(X, dim_string, channel_string, path_classifier)

    if mode == 'train' or mode == 'cv':
        imputerLabelEncoder_train(X, y, dim_string, channel_string, path_classifier)

    # X = imputer_test(X, dim_string)
    y = labelEncoder_test(y, dim_string, channel_string, path_classifier)

    # print X,y

    best_params_fold1 = {'xgb__objective': 'multi:softprob', 'xgb__learning_rate': 0.05, 'xgb__n_estimators': 388, 'xgb__max_depth': 6, 'anova_filter__k': 'all'} # fold1
    best_params_fold2 = {'xgb__objective': 'multi:softprob', 'xgb__learning_rate': 0.05, 'xgb__n_estimators': 452, 'xgb__max_depth': 4, 'anova_filter__k': 'all'} # fold2
    best_params_fold3 = {'xgb__objective': 'multi:softprob', 'xgb__learning_rate': 0.05, 'xgb__n_estimators': 379, 'xgb__max_depth': 4, 'anova_filter__k': 'all'} # fold3
    best_params_fold4 = {'xgb__objective': 'multi:softprob', 'xgb__learning_rate': 0.05, 'xgb__n_estimators': 486, 'xgb__max_depth': 9, 'anova_filter__k': 'all'} # fold4

    if dim_string == 'fold1':
        best_params = best_params_fold1
    elif dim_string == 'fold2':
        best_params = best_params_fold2
    elif dim_string == 'fold3':
        best_params = best_params_fold3
    elif dim_string == 'fold4':
        best_params = best_params_fold4

    clf = buildEstimators(mode, dim_string, channel_string, best_params, path_classifier)

    if mode == 'cv':
        best_params = grid_search(clf, X, y)
        clf = buildEstimators(mode, dim_string, channel_string, best_params, path_classifier)
        train_save(clf, X, y, dim_string, channel_string)
        return True
    elif mode == 'train':
        train_save(clf, X, y, dim_string, channel_string)
        return True
    elif mode == 'test':
        y_test, y_pred = prediction(clf, X, y, mode_eval)
        eval_metrics(y_test, y_pred)
        y_pred = clf.predict_proba(X)

        return y_pred

def batchProcess(path_feature_freesound_statistics_folds, mode, channel_string, mode_eval='normal'):

    if mode == 'train' or mode == 'cv':
        dataset_string = 'train'
    elif mode == 'test':
        dataset_string = 'eval'

    for ii in range(1,5):
        dim_string = 'fold'+str(ii)
        print(path_feature_freesound_statistics_folds, dim_string, dataset_string)
        X, y = loadFeature(path_feature_freesound_statistics_folds, dim_string+'_'+dataset_string+'.pkl')
        start_time = time.time()
        y_pred = trainTestProcess(X, y, mode, dim_string, channel_string, mode_eval)
        end_time = time.time()
        print('Elapse time', end_time-start_time)

        if mode == 'test':
            pickle.dump(y_pred, open(os.path.join(path_results, "predict_proba/y_pred_"+dim_string+"_"+channel_string+".pkl"),"wb"))


if __name__ == "__main__":
    # datafile = sys.argv[1]
    # mode=sys.argv[2]
    mode = 'test'
    # dim_string=sys.argv[3]

    #
    # batchProcess(path_feature_freesound_statistics_folds_seg_left, mode, channel_string='seg_left', mode_eval='mean')
    # batchProcess(path_feature_freesound_statistics_folds_seg_right, mode, channel_string='seg_right', mode_eval='mean')
    batchProcess(path_feature_freesound_statistics_folds_seg_average, mode, channel_string='seg_average', mode_eval='mean')
    # batchProcess(path_feature_freesound_statistics_folds_seg_difference, mode, channel_string='seg_difference', mode_eval='mean')

    # batchProcess(path_feature_freesound_statistics_folds_seg_left_augmentation, mode, channel_string='seg_left_aug', mode_eval='mean')
    # batchProcess(path_feature_freesound_statistics_folds_seg_right_augmentation, mode, channel_string='seg_right_aug', mode_eval='mean')
    # batchProcess(path_feature_freesound_statistics_folds_seg_average_augmentation, mode, channel_string='seg_average_aug',mode_eval='mean')
    # batchProcess(path_feature_freesound_statistics_folds_seg_difference_augmentation, mode, channel_string='seg_difference_aug',mode_eval='mean')


    # # check label encoders if they are the same
    # le_fold1 = pickle.load(open(path.join(path_classifier, "le_fold1.pkl"), "r"))
    # le_fold2 = pickle.load(open(path.join(path_classifier, "le_fold2.pkl"), "r"))
    # le_fold3 = pickle.load(open(path.join(path_classifier, "le_fold3.pkl"), "r"))
    # le_fold4 = pickle.load(open(path.join(path_classifier, "le_fold4.pkl"), "r"))
    # print(le_fold1.inverse_transform(range(15)))
    # print(le_fold2.inverse_transform(range(15)))
    # print(le_fold3.inverse_transform(range(15)))
    # print(le_fold4.inverse_transform(range(15)))


