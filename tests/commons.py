import sys
import pandas
import sklearn.feature_selection as sk
from sklearn.model_selection import StratifiedKFold
import time
from timeit import default_timer as timer
from contextlib import contextmanager
import numpy as np
from itertools import islice, count
from collections import namedtuple
import os
import argparse
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle


# from Commons.ConfusionMatrix import ConfusionMatrix


Dataset = namedtuple('Dataset', 'X_tr, Y_tr, X_val, Y_val, X_test, Y_test')



class Timer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.timer = timer

    def __enter__(self):
        self.start = self.timer()
        return self

    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # milliseconds
        if self.verbose:
            print('elapsed time: %f ms' % self.elapsed)


def decorator_time_measurement(func):
    def timeM(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        return result, end - start
    return timeM


def get_rows_to_skip(dataset_file):
    with open(dataset_file) as f:
        lines = f.readlines()
        # get list of all possible lins starting by beerId
        num = [i for i, l in enumerate(lines) if l.startswith("@")]
        # if not found value return 0 else get first value of list subtracted by 1
        num = 0 if len(num) == 0 else num[-1] + 1

    return num


def loadData(pathToDataset):
    '''
    Load csv format data from file

    :param pathToDataset: filesystem path to dataset file
    :return: pair of pandas data frame (first with data, second labels)
    '''
    num = get_rows_to_skip(pathToDataset)
    data = pandas.read_csv(pathToDataset, sep=',', skiprows=num, header=None)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    return X, Y


def loadFold(path_To_fold):
    X_test, Y_test = loadData(os.path.join(path_To_fold, "test.csv"))
    X_val, Y_val = loadData(os.path.join(path_To_fold, "validation.csv"))
    X_tr, Y_tr = loadData(os.path.join(path_To_fold, "train.csv"))

    return Dataset(X_tr, Y_tr, X_val, Y_val, X_test, Y_test)


def load_all_5_folds(path_to_dataset):
    fold_1 = loadFold(os.path.join(path_to_dataset,"1"))

    if os.path.exists(os.path.join(path_to_dataset, "2")):
        fold_2 = loadFold(os.path.join(path_to_dataset, "2"))
        fold_3 = loadFold(os.path.join(path_to_dataset, "3"))
        fold_4 = loadFold(os.path.join(path_to_dataset, "4"))
        fold_5 = loadFold(os.path.join(path_to_dataset, "5"))
        return [fold_1,fold_2,fold_3,fold_4,fold_5]

    return [fold_1]


def loadFold_T_equal_V(path_To_fold):
    X_test, Y_test = loadData(os.path.join(path_To_fold, "test.csv"))
    X_val, Y_val = loadData(os.path.join(path_To_fold, "validation.csv"))
    X_tr, Y_tr = loadData(os.path.join(path_To_fold, "train.csv"))

    if len(X_tr.index) != len(X_val.index):
        newTr_X = pandas.concat([X_tr, X_val], ignore_index=True)
        newTr_Y = pandas.concat([Y_tr, Y_val], ignore_index=True)

        import copy
        return Dataset(newTr_X, newTr_Y, copy.deepcopy(newTr_X), copy.deepcopy(newTr_Y), X_test, Y_test)


    return Dataset(X_tr, Y_tr, X_val, Y_val, X_test, Y_test)


def load_all_5_folds_T_equal_V(path_to_dataset):
    fold_1 = loadFold_T_equal_V(os.path.join(path_to_dataset,"1"))

    if os.path.exists(os.path.join(path_to_dataset, "2")):
        fold_2 = loadFold_T_equal_V(os.path.join(path_to_dataset, "2"))
        fold_3 = loadFold_T_equal_V(os.path.join(path_to_dataset, "3"))
        fold_4 = loadFold_T_equal_V(os.path.join(path_to_dataset, "4"))
        fold_5 = loadFold_T_equal_V(os.path.join(path_to_dataset, "5"))
        return [fold_1,fold_2,fold_3,fold_4,fold_5]

    return [fold_1]


def scaleData(dataset:Dataset):
    '''
    Scale data to range 0,1
    :param dataset:
    :return: transformed dataset
    '''
    scaler = MinMaxScaler(copy=False, feature_range=(0, 1))

    scaler.fit(dataset.X_tr)

    X_tr = scaler.transform(dataset.X_tr)
    X_val = scaler.transform(dataset.X_val)
    X_test = scaler.transform(dataset.X_test)

    return Dataset(X_tr,dataset.Y_tr, X_val=X_val, Y_val=dataset.Y_val, X_test=X_test, Y_test=dataset.Y_test)


def scaleData_11(dataset:Dataset):
    '''
    Scale data to range -1,1
    :param dataset:
    :return: transformed dataset
    '''
    scaler = MinMaxScaler(copy=False, feature_range=(-1, 1))

    scaler.fit(dataset.X_tr)

    X_tr = scaler.transform(dataset.X_tr)
    X_val = scaler.transform(dataset.X_val)
    X_test = scaler.transform(dataset.X_test)

    return Dataset(X_tr,dataset.Y_tr, X_val=X_val, Y_val=dataset.Y_val, X_test=X_test, Y_test=dataset.Y_test)


from sklearn.preprocessing import  StandardScaler

def standarizeData(dataset:Dataset):
    '''
    :param dataset:
    :return: transformed dataset
    '''
    scaler = StandardScaler(copy=False)

    scaler.fit(dataset.X_tr)

    X_tr = scaler.transform(dataset.X_tr)
    X_val = scaler.transform(dataset.X_val)
    X_test = scaler.transform(dataset.X_test)

    return Dataset(X_tr,dataset.Y_tr, X_val=X_val, Y_val=dataset.Y_val, X_test=X_test, Y_test=dataset.Y_test)


def filterDataset(dataset:Dataset):
    '''
    Filter dataset with algorithms from MutualInfoCalcualtion after thresholding (the same set as given to SVM)
    :param dataset:
    :return:
    '''
    from MutualInfoCalculation.mutualInfo import testTimes
    maskOfFeatures = testTimes(dataset.X_tr, dataset.Y_tr)
    mask =  np.array([maskOfFeatures > 0.00],dtype=bool)
    mask3 = np.ravel(mask)

    train = dataset.X_tr.values[:, mask3]
    val = dataset.X_val.values[:, mask3]
    test = dataset.X_test.values[:, mask3]

    return Dataset(X_tr=train,X_val=val,X_test=test,Y_tr=dataset.Y_tr,Y_val=dataset.Y_val,Y_test=dataset.Y_test)


# def get_confusion_matrix(dataset, classifier):
#     with Timer() as t:
#         tn, fp, fn, tp = confusion_matrix(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)).ravel()
#         cm_val = ConfusionMatrix.from_explicit_numbers(tp, fp, tn, fn)
#     time_validation = t.elapsed

#     # assert f1_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) ==  cm_val.f1()
#     # assert accuracy_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.accuracy()
#     # assert precision_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.precision()
#     # assert recall_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.recall()

#     with Timer() as t:
#         tn, fp, fn, tp = confusion_matrix(y_true=dataset.Y_test, y_pred=classifier.predict(dataset.X_test)).ravel()
#         conf_test = ConfusionMatrix.from_explicit_numbers(tp, fp, tn, fn)
#     time_test = t.elapsed
#     return cm_val, conf_test, time_validation, time_test


# def get_confusion_matrix_automl(dataset, classifier):
#     with Timer() as t:
#         tn, fp, fn, tp = confusion_matrix(y_true=dataset.Y_val, y_pred=classifier.predict(pandas.DataFrame(dataset.X_val))).ravel()
#         cm_val = ConfusionMatrix.from_explicit_numbers(tp, fp, tn, fn)
#     time_validation = t.elapsed

#     # assert f1_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) ==  cm_val.f1()
#     # assert accuracy_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.accuracy()
#     # assert precision_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.precision()
#     # assert recall_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.recall()

#     with Timer() as t:
#         tn, fp, fn, tp = confusion_matrix(y_true=dataset.Y_test, y_pred=classifier.predict(pandas.DataFrame(dataset.X_test))).ravel()
#         conf_test = ConfusionMatrix.from_explicit_numbers(tp, fp, tn, fn)
#     time_test = t.elapsed
#     return cm_val, conf_test, time_validation, time_test

# def get_confusion_matrix_multiclass(dataset, classifier):
#     with Timer() as t:
#         cm_val = ConfusionMatrix.from_pred_true(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val))
#     time_validation = t.elapsed

#     # assert f1_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) ==  cm_val.f1()
#     # assert accuracy_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.accuracy()
#     # assert precision_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.precision()
#     # assert recall_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.recall()

#     with Timer() as t:
#         conf_test = ConfusionMatrix.from_pred_true(y_true=dataset.Y_test, y_pred=classifier.predict(dataset.X_test))
#     time_test = t.elapsed
#     return cm_val, conf_test, time_validation, time_test