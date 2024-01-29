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
from sklearn.metrics import roc_auc_score, precision_recall_curve
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LassoCV, Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pickle


# from ResultAnalize.main import mergeAllToSummaryReport

import math


class ConfusionMatrix:
    def __init__(self, confusionMatrix):
        # the order come from c++ solution
        # out << matrix.truePositive() << "\t" << matrix.falsePositive() << "\t" << matrix.trueNegative() << "\t" << matrix.falseNegative();

        self.tp = float(confusionMatrix[0])
        self.fp = float(confusionMatrix[1])
        self.tn = float(confusionMatrix[2])
        self.fn = float(confusionMatrix[3])

    @classmethod
    def from_explicit_numbers(cls, tp, fp, tn, fn):
        return cls([tp,fp,tn,fn])

    def __add__(self, o):
        return ConfusionMatrix([self.tp + o.tp, self.fp + o.fp, self.tn + o.tn, self.fn + o.fn])

    def accuracy(self):
        if (self.tp + self.fn + self.fp + self.tn) == 0:
            return 0
        return (self.tp + self.tn) / (self.tp + self.fn + self.fp + self.tn)

    def f1(self):
        precision = self.precision()
        recall = self.recall()
        if (precision + recall) == 0:
            return 0
        return (2 * precision * recall) / (precision + recall)

    def precision(self):
        if (self.tp + self.fp) == 0:
            return 0
        return self.tp / (self.tp + self.fp)

    def recall(self):
        if (self.tp + self.fn) == 0:
            return 0
        return self.tp / (self.tp + self.fn)

    def MCC(self):
        self.tp = float(self.tp)
        self.fp = float(self.fp)
        self.tn = float(self.tn)
        self.fn = float(self.fn)

        if math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)) == 0:
            return 0
        return ((self.tp * self.tn) - (self.fp * self.fn)) / math.sqrt((self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn))





from hypopt import GridSearch

# from sklearn.model_selection import cross_validate

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

    if len(X_tr.index) != len(X_val.index):
        newTr_X = pandas.concat([X_tr, X_val], ignore_index=True)
        newTr_Y = pandas.concat([Y_tr, Y_val], ignore_index=True)

        return Dataset(newTr_X, newTr_Y, newTr_X, newTr_Y, X_test, Y_test)

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


def filterDataset(dataset:Dataset):
    '''
    Filter dataset with algorithms from MutualInfoCalcualtion after thresholding (the same set as given to SVM)
    :param dataset:
    :return:
    '''
    maskOfFeatures = testTimes(dataset.X_tr, dataset.Y_tr)
    mask =  np.array([maskOfFeatures > 0.00],dtype=bool)
    mask3 = np.ravel(mask)

    train = dataset.X_tr.values[:, mask3]
    val = dataset.X_val.values[:, mask3]
    test = dataset.X_test.values[:, mask3]

    return Dataset(X_tr=train,X_val=val,X_test=test,Y_tr=dataset.Y_tr,Y_val=dataset.Y_val,Y_test=dataset.Y_test)


def parse() -> str:
    """
    Parse arguments
    :return: str
    """
    parser = argparse.ArgumentParser(description='Experiments ')

    parser.add_argument('-t',
                        action="store",
                        dest="training_set",
                        help='Training set path')
    parser.add_argument('-K',
                        action="store",
                        dest="K",
                        type=int,
                        help='number of samples per class')

    parser.add_argument('-a',
                        action="store",
                        dest="algorithmName",
                        help='name of algorithm to run')

    parsed = parser.parse_args()

    return parsed.training_set, parsed.K, parsed.algorithmName


def get_confusion_matrix(dataset, classifier):
    with Timer() as t:
        tn, fp, fn, tp = confusion_matrix(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)).ravel()
        cm_val = ConfusionMatrix.from_explicit_numbers(tp, fp, tn, fn)
    time_validation = t.elapsed

    # assert f1_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) ==  cm_val.f1()
    # assert accuracy_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.accuracy()
    # assert precision_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.precision()
    # assert recall_score(y_true=dataset.Y_val, y_pred=classifier.predict(dataset.X_val)) == cm_val.recall()

    with Timer() as t:
        tn, fp, fn, tp = confusion_matrix(y_true=dataset.Y_test, y_pred=classifier.predict(dataset.X_test)).ravel()
        conf_test = ConfusionMatrix.from_explicit_numbers(tp, fp, tn, fn)
    time_test = t.elapsed
    return cm_val, conf_test, time_validation, time_test


def nonLinearSvm(dataset:Dataset):
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'degree': [2,3,4,5,6,7], 'kernel': ['poly']}
    ]
    # Grid-search all parameter combinations using a validation set.
    svclassifier = GridSearch(model=SVC(), param_grid=param_grid)
    with Timer() as t:
        svclassifier.fit(dataset.X_tr, dataset.Y_tr, dataset.X_val, dataset.Y_val)
    training_time = t.elapsed

    svclassifier = svclassifier.model

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


def rbfSvm(dataset:Dataset):
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [1000, 100, 10, 1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
    ]
    # Grid-search all parameter combinations using a validation set.
    svclassifier = GridSearch(model=SVC(), param_grid=param_grid)
    with Timer() as t:
        svclassifier.fit(dataset.X_tr, dataset.Y_tr, dataset.X_val, dataset.Y_val)
    training_time = t.elapsed

    svclassifier = svclassifier.model

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


def linearSvm(dataset:Dataset):
    param_grid = [
        {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']}
    ]
    # Grid-search all parameter combinations using a validation set.
    svclassifier = GridSearch(model=SVC(), param_grid=param_grid)
    with Timer() as t:
        svclassifier.fit(dataset.X_tr, dataset.Y_tr, dataset.X_val, dataset.Y_val)
    training_time = t.elapsed
    svclassifier = svclassifier.model

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier



import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.svm import SVC,SVR
from sklearn import datasets
import scipy.stats as stats

# HPO Algorithm 3: Hyperband
from hyperband import HyperbandSearchCV
from scipy.stats import randint as sp_randint
from random import randrange as sp_randrange

def rbfSvm_Hyperband(dataset:Dataset):
    rf_params = {
        'C': stats.uniform(0, 50),
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
    }

    clf = SVC(gamma='scale', verbose=True, max_iter=10000000)

    with Timer() as t:
        hyper = HyperbandSearchCV(clf, param_distributions=rf_params, cv=3, min_iter=1, max_iter=50, n_jobs=-1, scoring='accuracy',
                                  resource_param='C', verbose=1)
        hyper.fit(dataset.X_tr, dataset.Y_tr)
    training_time = t.elapsed

    print(hyper.best_params_)
    print("AUC:" + str(hyper.best_score_))

    svclassifier = hyper.best_estimator_

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier

from skopt import Optimizer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# HPO Algorithm 4: BO-GP
def rbfSvm_bo_gp(dataset:Dataset):
    rf_params = {
        'C': Real(0.01, 50),
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid']
    }
    clf = SVC(gamma='scale',  verbose=True, max_iter=100000)

    with Timer() as t:
        hyper = BayesSearchCV(clf, rf_params, cv=3, n_iter=20, n_jobs=-1, scoring='accuracy', verbose=1)
        hyper.fit(dataset.X_tr, dataset.Y_tr)
    training_time = t.elapsed

    print(hyper.best_params_)
    print("bo_gp AUC:" + str(hyper.best_score_))

    svclassifier = hyper.best_estimator_

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


# HPO Algorithm 5: BO-TPE
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score, StratifiedKFold




def rbfSvm_bo_tpe(dataset:Dataset):
    space = {
        'C': hp.normal('C', 0.01, 50),
        "kernel": hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    }

    def objective(params):
        params = {
            'C': abs(float(params['C'])),
            "kernel": str(params['kernel']),
        }
        clf = SVC(gamma='scale',  verbose=True, max_iter=10000000, **params)
        score = cross_val_score(clf, dataset.X_tr, dataset.Y_tr, scoring='accuracy',
                                cv=StratifiedKFold(n_splits=3)).mean()
        return {'loss': -score, 'status': STATUS_OK}


    kenrel_list = ['linear', 'poly', 'rbf', 'sigmoid']
    with Timer() as t:
        best = fmin(fn=objective,space=space,algo=tpe.suggest, max_evals=20)

        best['C'] = abs(best['C'])
        best['kernel'] = kenrel_list[best['kernel']]
        print(best)
        clf = SVC(max_iter=10000000, **best)
        clf.fit(dataset.X_tr, dataset.Y_tr)
    training_time = t.elapsed

    print(best)
    # print("bo_gp AUC:" + str(hyper.best_score_))

    svclassifier = clf

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


# HPO Algorithm 6: PSO
import optunity
import optunity.metrics

def rbfSvm_pso(dataset:Dataset):
    search = {
        'C': (0, 50),
        'kernel': [0, 4]
    }

    @optunity.cross_validated(x=dataset.X_tr, y=dataset.Y_tr.tolist(), num_folds=3)
    def performance(x_train, y_train, x_test, y_test,C=None,kernel=None):
        # fit the model
        if kernel < 1:
            ke = 'linear'
        elif kernel < 2:
            ke = 'poly'
        elif kernel < 3:
            ke = 'rbf'
        else:
            ke = 'sigmoid'
        model = SVC(C=float(C),
                    kernel=ke, max_iter=10000000, verbose=True)
        scores = np.mean(cross_val_score(model, x_train, y_train, cv=3, n_jobs=-1,
                                         scoring="accuracy"))
        print(f'PSO accuracy {scores},  kernel={ke},  C={float(C)}')
        return scores

    with Timer() as t:
        optimal_configuration, info, _ = optunity.maximize(performance,
                                                           solver_name='particle swarm',
                                                           num_evals=20,
                                                           **search)
        print(optimal_configuration)
        print("PSO AUC:" + str(info.optimum))
        if optimal_configuration['kernel'] < 1:
            optimal_configuration['kernel'] = 'linear'
        elif  optimal_configuration['kernel'] < 2:
            optimal_configuration['kernel'] = 'poly'
        elif  optimal_configuration['kernel'] < 3:
            optimal_configuration['kernel'] = 'rbf'
        else:
            optimal_configuration['kernel'] = 'sigmoid'
        clf = SVC(max_iter=10000000, **optimal_configuration)
        clf.fit(dataset.X_tr, dataset.Y_tr)
    training_time = t.elapsed

    svclassifier = clf

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


# HPO Algorithm 7: Genetic Algorithm
from tpot import TPOTClassifier

def rbfSvm_ga(dataset:Dataset):
    parameters = {
        'C': np.random.uniform(0.001, 1000, 100),
        'gamma': np.random.uniform(0.001, 1000, 100),
        "kernel": ['rbf'],
        'max_iter': [1000000]
    }


    X_train = np.concatenate((dataset.X_tr, dataset.X_val))
    Y_train = np.concatenate((dataset.Y_tr, dataset.Y_val))


    # The indices which have the value -1 will be kept in train.
    train_indices = np.full((len(dataset.X_tr),), -1, dtype=int)

    # The indices which have zero or positive values, will be kept in test
    test_indices = np.full((len(dataset.X_val),), 0, dtype=int)
    test_fold = np.append(train_indices, test_indices)

    # print(test_fold)
    # OUTPUT:
    # array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    #        -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    #        -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    from sklearn.model_selection import PredefinedSplit
    ps = PredefinedSplit(test_fold)

    # Check how many splits will be done, based on test_fold
    # ps.get_n_splits()
    # OUTPUT: 1

    # for train_index, test_index in ps.split():
    #     print("TRAIN:", train_index, "TEST:", test_index)

    with Timer() as t:
        ga2 = TPOTClassifier(generations=5, population_size=10, offspring_size=10, n_jobs=1,
                             verbosity=1, early_stop=5, template = 'Classifier', random_state = 42,
                             config_dict={'sklearn.svm.SVC': parameters},cv=2, scoring='roc_auc')
        ga2.fit(X_train, Y_train)
    training_time = t.elapsed / 2

    print(f"GA time: {training_time}")
    # print(ga2.best_params_)
    # print("AUC:" + str(ga2.best_score_))

    svclassifier = ga2.fitted_pipeline_._final_estimator

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier



from mealpy.evolutionary_based.GA import BaseGA
from mealpy.evolutionary_based.MA import BaseMA

def rbfSvm_mealpy_custom(dataset:Dataset):
    lb = [0.001, 0.001]
    ub = [1000, 1000]
    problem_size = 2
    batch_size = 10 #not used at all
    verbose = True
    epoch = 10
    pop_size = 5


    def my_objective(solution=None):
        clf = SVC(gamma=solution[0], C=solution[1], verbose=True, max_iter=10000000,)
        clf.fit(dataset.X_tr, dataset.Y_tr)

        y_val_prob = clf.decision_function(dataset.X_val)
        score_val = roc_auc_score(dataset.Y_val, y_val_prob)

        # score = cross_val_score(clf, dataset.X_tr, dataset.Y_tr, scoring='auc',
        #                         cv=StratifiedKFold(n_splits=1)).mean()
        return score_val

    try:

        with Timer() as t:
            md1 = BaseMA(my_objective, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
            best_pos1, best_fit1, list_loss1 = md1.train()
            clf = SVC(gamma=best_pos1[0], C=best_pos1[1], verbose=True, max_iter=10000000, )
            clf.fit(dataset.X_tr, dataset.Y_tr)

        training_time = t.elapsed

    except Exception as e:
        print(f"ERROR: {e}")
        return rbfSvm_mealpy_custom(dataset)

    # print(ga2.best_params_)
    # print("AUC:" + str(ga2.best_score_))

    svclassifier = clf

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier



from mealpy.evolutionary_based.DE import BaseDE

def rbfSvm_differential_evolution(dataset:Dataset):
    lb = [0.001, 0.001]
    ub = [1000, 1000]
    problem_size = 2
    batch_size = 10 #not used at all
    verbose = True
    epoch = 10
    pop_size = 5


    def my_objective(solution=None):
        clf = SVC(gamma=solution[0], C=solution[1], verbose=True, max_iter=10000000,)
        clf.fit(dataset.X_tr, dataset.Y_tr)

        y_val_prob = clf.decision_function(dataset.X_val)
        score_val = roc_auc_score(dataset.Y_val, y_val_prob)

        # score = cross_val_score(clf, dataset.X_tr, dataset.Y_tr, scoring='auc',
        #                         cv=StratifiedKFold(n_splits=1)).mean()
        return score_val

    try:

        with Timer() as t:
            md1 = BaseDE(my_objective, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
            best_pos1, best_fit1, list_loss1 = md1.train()
            clf = SVC(gamma=best_pos1[0], C=best_pos1[1], verbose=True, max_iter=10000000, )
            clf.fit(dataset.X_tr, dataset.Y_tr)

        training_time = t.elapsed
    except Exception as e:
        print(f"ERROR: {e}")
        return rbfSvm_differential_evolution(dataset)
    # print(ga2.best_params_)
    # print("AUC:" + str(ga2.best_score_))

    svclassifier = clf

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


from mealpy.swarm_based.PSO import BasePSO

def rbfSvm_PSO_mealpy(dataset:Dataset):
    lb = [0.001, 0.001]
    ub = [1000, 1000]
    problem_size = 2
    batch_size = 10 #not used at all
    verbose = True
    epoch = 10
    pop_size = 5


    def my_objective(solution=None):
        clf = SVC(gamma=solution[0], C=solution[1], verbose=True, max_iter=10000000,)
        clf.fit(dataset.X_tr, dataset.Y_tr)

        y_val_prob = clf.decision_function(dataset.X_val)
        score_val = roc_auc_score(dataset.Y_val, y_val_prob)

        # score = cross_val_score(clf, dataset.X_tr, dataset.Y_tr, scoring='auc',
        #                         cv=StratifiedKFold(n_splits=1)).mean()
        return score_val
    try:
        with Timer() as t:
            md1 = BasePSO(my_objective, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
            best_pos1, best_fit1, list_loss1 = md1.train()
            clf = SVC(gamma=best_pos1[0], C=best_pos1[1], verbose=True, max_iter=10000000, )
            clf.fit(dataset.X_tr, dataset.Y_tr)

        training_time = t.elapsed
    except Exception as e:
        print(f"ERROR: {e}")
        return rbfSvm_PSO_mealpy(dataset)
    # print(ga2.best_params_)
    # print("AUC:" + str(ga2.best_score_))

    svclassifier = clf

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier


from mealpy.physics_based.MVO import OriginalMVO

def rbfSvm_MVO_mealpy(dataset:Dataset):
    lb = [0.001, 0.001]
    ub = [1000, 1000]
    problem_size = 2
    batch_size = 10 #not used at all
    verbose = True
    epoch = 10
    pop_size = 5


    def my_objective(solution=None):
        clf = SVC(gamma=solution[0], C=solution[1], verbose=True, max_iter=10000000,)
        clf.fit(dataset.X_tr, dataset.Y_tr)

        y_val_prob = clf.decision_function(dataset.X_val)
        score_val = roc_auc_score(dataset.Y_val, y_val_prob)

        # score = cross_val_score(clf, dataset.X_tr, dataset.Y_tr, scoring='auc',
        #                         cv=StratifiedKFold(n_splits=1)).mean()
        return score_val

    try:
        with Timer() as t:
            md1 = OriginalMVO(my_objective, lb, ub, problem_size, batch_size, verbose, epoch, pop_size)
            best_pos1, best_fit1, list_loss1 = md1.train()
            clf = SVC(gamma=best_pos1[0], C=best_pos1[1], verbose=True, max_iter=10000000, )
            clf.fit(dataset.X_tr, dataset.Y_tr)

        training_time = t.elapsed
    except Exception as e:
        print(f"ERROR: {e}")
        return rbfSvm_MVO_mealpy(dataset)
    # print(ga2.best_params_)
    # print("AUC:" + str(ga2.best_score_))

    svclassifier = clf

    y_val_prob = svclassifier.decision_function(dataset.X_val)
    y_test_prob = svclassifier.decision_function(dataset.X_test)

    score_val = roc_auc_score(dataset.Y_val, y_val_prob)
    score_test = roc_auc_score(dataset.Y_test, y_test_prob)

    cm_val, conf_test, time_validation, time_test = get_confusion_matrix(dataset, svclassifier)

    return score_val, score_test, cm_val, conf_test, training_time, time_validation, time_test, svclassifier

def experiment(traningSetPath, K):
    foldsDatasets = load_all_5_folds(traningSetPath)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    PathForResults = os.path.join(traningSetPath, "scikit_" + timestr)
    os.makedirs(PathForResults)

    allAlgorithmsDict = {}

    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'linearSvm',  PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'SvmPoly',  PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'rbfSvm',  PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'hyperband', PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'bogp', PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'botpe', PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    # algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'pso', PathForResults, K=K)
    # allAlgorithmsDict[algorithmName] = algorithmResults
    algorithmResults, algorithmName = get_single_method_results(foldsDatasets, 'ga', PathForResults, K=K)
    allAlgorithmsDict[algorithmName] = algorithmResults





    return allAlgorithmsDict


def get_single_method_results(foldsDatasets, algorithmName, save_path, K):
    score_val = []
    score_test = []
    cm_val = []
    conf_test = []
    folds = 0
    training_time_sum = []
    time_validation_sum = []
    time_test_sum = []
    sv_number = []

    try:
        for fold in foldsDatasets:
            dataset = scaleData(fold)

            #select subset of given size
            y_values = list(set(dataset.Y_tr))
            ones = dataset.Y_tr == y_values[0]
            ones = ones.loc[ones == True].index
            zeros = dataset.Y_tr == y_values[1]
            zeros = zeros.loc[zeros == True].index
            import random
            ones_selected = random.choices(ones, k=K)
            zeros_selected = random.choices(zeros, k=K)
            subset_index = ones_selected + zeros_selected
            dataset= Dataset(dataset.X_tr[subset_index], dataset.Y_tr[subset_index], X_val=dataset.X_val, Y_val=dataset.Y_val, X_test=dataset.X_test, Y_test=dataset.Y_test)


            if algorithmName == 'linearSvm':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = linearSvm(dataset)
            elif algorithmName == 'SvmPoly':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = nonLinearSvm(dataset)
            elif algorithmName == 'rbfSvm':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm(dataset)
            elif algorithmName == 'hyperband':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_Hyperband(dataset)
            elif algorithmName == 'bogp':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_bo_gp(dataset)
            elif algorithmName == 'botpe':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_bo_tpe(dataset)
            elif algorithmName == 'pso':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_pso(dataset)
            elif algorithmName == 'ga':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_ga(dataset)
            elif algorithmName == 'baseGA':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_ga(dataset)
            elif algorithmName == 'baseDE':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_differential_evolution(dataset)
            elif algorithmName == 'basePSO':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_PSO_mealpy(dataset)
            elif algorithmName == 'MVO':
                scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_MVO_mealpy(dataset)


            # save
            timestr = time.strftime("%Y%m%d-%H%M%S")
            experiment_model = algorithmName + "_" + str(folds) + "_" + timestr + '.pkl'
            with open(os.path.join(save_path, experiment_model), 'wb') as f:
                pickle.dump(model, f)

            # load
            # with open('model.pkl', 'rb') as f:
            #     clf2 = pickle.load(f)



            score_val.append(scv)
            score_test.append(sct)
            training_time_sum.append(training_time)
            time_validation_sum.append(time_validation)
            time_test_sum.append(time_test)
            sv_number.append(0)     #not use only with SVM
            # sv_number.append(len(model.support_vectors_))     #not use only with SVM
            cm_val.append(cmv)
            conf_test.append(cmt)
            folds += 1
        score_val = np.array(score_val)
        score_test = np.array(score_test)
        training_time_sum = np.array(training_time_sum)
        time_validation_sum = np.array(time_validation_sum)
        time_test_sum = np.array(time_test_sum)
        sv_number = np.array(sv_number)

    except Exception as e:
        score_val = []
        score_test = []
        cm_val = [ConfusionMatrix.from_explicit_numbers(0, 0, 0, 0)]
        conf_test = [ConfusionMatrix.from_explicit_numbers(0, 0, 0, 0)]
        training_time_sum = []
        time_validation_sum = []
        time_test_sum = []
        sv_number = []
        score_val = np.array(score_val)
        score_test = np.array(score_test)
        training_time_sum = np.array(training_time_sum)
        time_validation_sum = np.array(time_validation_sum)
        time_test_sum = np.array(time_test_sum)
        sv_number = np.array(sv_number)
        print(e)

    # ColumnsNames = {"time": 3, "FitVal": 4, "SV": 6, "FitTEST": 8, "clTimeVal": 13, "clTimeTest": 15,
    #                 "featuresNumber": 17, "TrSize": 18}

    algorithmResults = {"FitVal": (score_val.mean(), score_val.std()),
                        "FitTEST": (score_test.mean(), score_test.std()),
                        "ConfusionMatrixVal": cm_val,
                        "ConfusionMatrixTest": conf_test,
                        "time": (training_time_sum.mean(), training_time_sum.std()),
                        "clTimeVal": (time_validation_sum.mean(), time_validation_sum.std()),
                        "clTimeTest": (time_test_sum.mean(), time_test_sum.std()),
                        "SV" : (sv_number.mean(), sv_number.std())}    #not use only with SVM
    return algorithmResults, algorithmName


def run_single_dataset(path_to_data, algorithmName, K):
    fold = loadFold(path_to_data)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    PathForResults = os.path.join(path_to_data, "scikit_" + timestr)
    os.makedirs(PathForResults)

    score_val = []
    score_test = []
    cm_val = ConfusionMatrix.from_explicit_numbers(0, 0, 0, 0)
    conf_test = ConfusionMatrix.from_explicit_numbers(0, 0, 0, 0)
    folds = 0
    training_time_sum = []
    time_validation_sum = []
    time_test_sum = []
    sv_number = []

    try:
        dataset = scaleData(fold)

        # select subset of given size
        y_values = list(set(dataset.Y_tr))
        ones = dataset.Y_tr == y_values[0]
        ones = ones.loc[ones == True].index
        zeros = dataset.Y_tr == y_values[1]
        zeros = zeros.loc[zeros == True].index
        import random
        ones_selected = random.choices(ones, k=K)
        zeros_selected = random.choices(zeros, k=K)
        subset_index = ones_selected + zeros_selected

        dataset= Dataset(dataset.X_tr[subset_index], dataset.Y_tr[subset_index], X_val=dataset.X_val, Y_val=dataset.Y_val, X_test=dataset.X_test, Y_test=dataset.Y_test)

        model = None

        model = None

        if algorithmName == 'linearSvm':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = linearSvm(dataset)
        elif algorithmName == 'SvmPoly':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = nonLinearSvm(dataset)
        elif algorithmName == 'rbfSvm':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm(dataset)
        elif algorithmName == 'hyperband':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_Hyperband(dataset)
        elif algorithmName == 'bogp':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_bo_gp(dataset)
        elif algorithmName == 'botpe':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_bo_tpe(dataset)
        elif algorithmName == 'pso':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_pso(dataset)
        elif algorithmName == 'ga':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_ga(dataset)
        elif algorithmName == 'baseGA':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_ga(dataset)
        elif algorithmName == 'baseDE':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_differential_evolution(dataset)
        elif algorithmName == 'basePSO':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_PSO_mealpy(dataset)
        elif algorithmName == 'MVO':
            scv, sct, cmv, cmt, training_time, time_validation, time_test, model = rbfSvm_MVO_mealpy(dataset)
        # save
        timestr = time.strftime("%Y%m%d-%H%M%S")
        experiment_model = algorithmName + "_" + str(folds) + "_" + timestr + '.pkl'
        with open(os.path.join(path_to_data, experiment_model), 'wb') as f:
            pickle.dump(model, f)

        # load
        # with open('model.pkl', 'rb') as f:
        #     clf2 = pickle.load(f)

        score_val.append(scv)
        score_test.append(sct)
        training_time_sum.append(training_time)
        time_validation_sum.append(time_validation)
        time_test_sum.append(time_test)
        sv_number.append(len(model.support_vectors_))  # not use only with SVM
        cm_val = cm_val + cmv
        conf_test = conf_test + cmt
    except Exception as e:
        score_val = []
        score_test = []
        cm_val = ConfusionMatrix.from_explicit_numbers(0, 0, 0, 0)
        conf_test = ConfusionMatrix.from_explicit_numbers(0, 0, 0, 0)
        training_time_sum = []
        time_validation_sum = []
        time_test_sum = []
        sv_number = []
        score_val = np.array(score_val)
        score_test = np.array(score_test)
        training_time_sum = np.array(training_time_sum)
        time_validation_sum = np.array(time_validation_sum)
        time_test_sum = np.array(time_test_sum)
        sv_number = np.array(sv_number)
        print(e)

    # ColumnsNames = {"time": 3, "FitVal": 4, "SV": 6, "FitTEST": 8, "clTimeVal": 13, "clTimeTest": 15,
    #                 "featuresNumber": 17, "TrSize": 18}

    # algorithmResults = {"FitVal": (score_val.mean(), score_val.std()),
    #                     "FitTEST": (score_test.mean(), score_test.std()),
    #                     "ConfusionMatrixVal": cm_val,
    #                     "ConfusionMatrixTest": conf_test,
    #                     "time": (training_time_sum.mean(), training_time_sum.std()),
    #                     "clTimeVal": (time_validation_sum.mean(), time_validation_sum.std()),
    #                     "clTimeTest": (time_test_sum.mean(), time_test_sum.std()),
    #                     "SV": (sv_number.mean(), sv_number.std())}  # not use only with SVM
    # return algorithmResults, algorithmName
    if model == None:
        print("There was some critical error, using some defaults")
        return 10,10, 1000

    return model.gamma, model.C, training_time + time_validation + time_test




def test_on_all_datasets():
    traningSetPath = r'D:\journal_datasets_subsets'
    allResultsDict = {}

    for trPath in os.listdir(traningSetPath):
        if os.path.isdir(os.path.join(traningSetPath,trPath)):
            datasetName = trPath
            print(datasetName)
            results = experiment(os.path.join(traningSetPath, trPath), K=8)
            allResultsDict[datasetName] = results
            with open(os.path.join(traningSetPath, datasetName + '_results.pickle'), 'wb') as fp:
                pickle.dump(allResultsDict, fp, protocol=pickle.HIGHEST_PROTOCOL)


    with open(os.path.join(traningSetPath, 'data_python_comparison.pickle'), 'wb') as fp:
        pickle.dump(allResultsDict, fp, protocol=pickle.HIGHEST_PROTOCOL)




    # mergeAllToSummaryReport(traningSetPath,[], allResultsDict)


def main(argv):
    test_on_all_datasets()


    traningSetPath, K, algorithmName = parse()
    # traningSetPath = r'C:\PHD\experiments_2D3'

    gamma, c, time = run_single_dataset(traningSetPath, algorithmName, K)

    dirToSaveResults = os.path.dirname(traningSetPath)


    results = np.asarray([gamma,c, time])

    np.savetxt(os.path.join(dirToSaveResults, f"_{algorithmName}_parameters_selected.txt"), X=results.reshape(1, -1), delimiter=',',
               fmt='%f')
    import datetime
    date = datetime.datetime.now().strftime('%Y-%m-%d__%H_%M_%S')
    np.savetxt(os.path.join(dirToSaveResults, f"__{date}__parameters_selected.txt"), X=results.reshape(1, -1),
               delimiter=',', fmt='%d')
    pass


if __name__ == "__main__":
    main(sys.argv)
