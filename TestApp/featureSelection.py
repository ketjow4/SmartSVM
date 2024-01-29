
from typing import Tuple
import argparse

import pandas as pd
import numpy as np
import os
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



import time

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
    data = pd.read_csv(pathToDataset, sep=',', skiprows=num, header=None)

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    return X, Y



def loadEsaData(path):
    dataset = pd.read_csv(path, sep=';')

    X = dataset.iloc[:, :-1]
    Y = dataset.iloc[:, -1]

    X.pop('event_id')
    X.pop('mission_id')
    X.pop('delta_p')
    X.pop('last_cdm_risk')

    # X = X.reset_index()
    # X = X.fillna(0)

    # print(pd.isnull(X).sum() > 0)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)

    return X,Y


def tranfromForSvm():


    features_filter = pd.read_csv(r"D:\PHD\ESA_Konkurs\wektor_cech_5_na_9_rfe.txt", sep=',', header=None, dtype=bool)
    features_filter = features_filter.values

    dataset = pd.read_csv(r"D:\PHD\ESA_Konkurs\train.csv", sep=';')
    dataset.pop('event_id')
    dataset.pop('mission_id')
    dataset.pop('delta_p')
    dataset.pop('last_cdm_risk')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataset.iloc[:, :-1])
    dataset.iloc[:, :-1] = scaler.transform(dataset.iloc[:, :-1])

    dataset = dataset.values[:, np.append(features_filter, True)]

    np.savetxt(os.path.join(r'D:\PHD\ESA_Konkurs\Przetworzone' + "\\train.csv"), dataset, delimiter=",")

    test = pd.read_csv(r"D:\PHD\ESA_Konkurs\test.csv", sep=',')
    test.pop('event_id')
    test.pop('mission_id')
    test.pop('delta_p')
    test.pop('last_cdm_risk')

    test = scaler.transform(test)

    test = test[:, features_filter.reshape(105)]

    np.savetxt(os.path.join(r'D:\PHD\ESA_Konkurs\Przetworzone' + "\\test.csv"), test, delimiter=",")

    pass


def parse() -> str:
    """
    Parse arguments
    :return: str
    """
    parser = argparse.ArgumentParser(description='Feature Selection for SVM')

    parser.add_argument('-t',
                        action="store",
                        dest="training_set",
                        help='Training set path')

    return parser.parse_args().training_set



def featureImportanceSelection(X,Y, n_estimators=20, scoring='roc_auc'):

    model = ExtraTreesClassifier(n_estimators=n_estimators, verbose=0)
    # model = SVC(kernel='linear')

    if any (Y.value_counts() < 5):
        selector = RFECV(model, step=0.1, cv=2, n_jobs=-1, scoring=scoring)
    else:
        selector = RFECV(model, step=0.1, cv=5, n_jobs=-1, scoring=scoring)
    selector = selector.fit(X, Y)
    print(selector.support_)
    print(selector.n_features_)

    # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    # plt.show()

    # zmienna = np.array(selector.support_)
    #
    # a = np.argsort(zmienna)
    # th2 = 0.5 #take best 50% of features
    # b = a[:int(len(zmienna)*th2)]
    # bb = a[int(len(zmienna)*th2):]
    # zmienna[b] = 0
    # zmienna[bb] = 1
    print(scoring)
    print(selector.grid_scores_)

    return selector.support_



def main():
    import sys
    # sys.stdout = open(r"D:\PHD\ESA_Konkurs\console_log.txt", "w")

    # tranfromForSvm()

    # traningSetPath = r"D:\PHD\ESA_Konkurs\train.csv"
    traningSetPath = parse()

    start = time.time()

    X,Y = loadData(traningSetPath)

    end = time.time()
    print("Loading took: {}".format(str(end - start)))

    # measure time elapsed by feature selection
    start = time.time()

    n_estimators = 10
    # wektorCech_f1 = featureImportanceSelection(X, Y, n_estimators=n_estimators,  scoring='f1')
    # wektorCech_ba = featureImportanceSelection(X, Y, n_estimators=n_estimators, scoring='balanced_accuracy')
    wektorCech_roc = featureImportanceSelection(X, Y, n_estimators=n_estimators)

    end = time.time()
    print(str(end - start))

    # print(wektorCech)


    dirToSaveResults = os.path.dirname(traningSetPath)


    f = open(os.path.join(dirToSaveResults, "timeOfRFECV1.txt"), "a+")
    f.write(str(end - start))
    f.close()

    np.savetxt(os.path.join(dirToSaveResults,"featureSelection.txt"), X=wektorCech_roc.reshape(1,-1), delimiter=',', fmt='%d')
    import datetime
    date = datetime.datetime.now().strftime('%Y-%m-%d__%H_%M_%S')
    np.savetxt(os.path.join(dirToSaveResults, f"__{date}__featureSelection.txt"), X=wektorCech_roc.reshape(1, -1), delimiter=',',fmt='%d')
    # np.savetxt(os.path.join(dirToSaveResults, "featureSelection_balanced_accuracy.txt"), X=wektorCech_ba.reshape(1, -1),
    #            delimiter=',', fmt='%d')
    # np.savetxt(os.path.join(dirToSaveResults, "featureSelection_f1.txt"), X=wektorCech_f1.reshape(1, -1),
    #            delimiter=',', fmt='%d')



if __name__ == "__main__":
    main()