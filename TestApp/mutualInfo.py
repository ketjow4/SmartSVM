import sys
import os
import time
import argparse
from typing import Tuple

import numpy as np



def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from warnings import simplefilter
# ignore all future warnings

#need to fix joblib import in library as joblib is not a part of sklearn starting 0.23 version
from stability_selection import StabilitySelection


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import RandomizedLogisticRegression
import sklearn.feature_selection as sk
import pandas

RANDOM=42

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

    if pathToDataset.endswith(".groups"):
        data = data.iloc[:, :-1] #remove last column with groups info

    X = data.iloc[:, :-1]
    Y = data.iloc[:, -1]

    return X, Y


def maskToProbabilites(mask_features):
    scores = np.asarray(mask_features, dtype=float)

    sum_of_scores = np.sum(scores)
    probabilities = np.divide(scores, sum_of_scores)
    return probabilities

def featureImportanceSelection(X,Y):
    start = time.time()
    model = ExtraTreesClassifier(n_estimators=10, random_state=RANDOM)

    if any(Y.value_counts() < 5):
        selector = RFECV(model, step=0.1, cv=2, n_jobs=-1)
    else:
        selector = RFECV(model, step=0.1, cv=5, n_jobs=-1)

    selector = selector.fit(X, Y)
    # print(selector.support_)
    # print(selector.n_features_)

    # # Plot number of features VS. cross-validation scores
    # plt.figure()
    # plt.xlabel("Number of features selected")
    # plt.ylabel("Cross validation score (nb of correct classifications)")
    # plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
    # plt.show()
    end = time.time()
    print("Recursive feature elimination time: " + str(end - start))

    return maskToProbabilites(selector.get_support())


def varianceThreshold(features,Y):
    start = time.time()
    # variances, _pval = sk.f_classif(X=features, y=Y)
    selection = sk.SelectPercentile(sk.f_classif, percentile=10)
    selection.fit(X=features, y=Y)
    mask = selection.get_support()
    scores = np.asarray(mask, dtype=int)

    # scores = np.nan_to_num(scores)
    end = time.time()
    print("Variance threshold " + str(end - start))

    sum_of_scores = np.sum(scores)
    probabilities = np.divide(scores, sum_of_scores)
    return probabilities


# super slow do not use !!!!
# def ReliefFSelection(features,Y):
#     start = time.time()
#
#     relief = ReliefF(n_features_to_select=20, n_jobs=-1, verbose=False, n_neighbors=5)
#     Y = Y.values
#     features = np.array(features.values, dtype=np.float64)
#     relief.fit(X=features, y=Y)
#     end = time.time()
#
#     print("ReliefF time: " + str(end - start))
#
#     result = relief.feature_importances_
#     scores = np.asarray(result, dtype=float)
#
#     sum_of_scores = np.sum(scores)
#     probabilities = np.divide(scores, sum_of_scores)
#
#     return probabilities



from stability_selection.bootstrap import stratified_bootstrap

def StabilitySelectionMy(features,Y):
    start = time.time()

    n_bootstrap_iterations = 50

    if len(features) > 20000:
        sample_fraction = 0.05
    else:
        sample_fraction = 0.1

    if sum(Y == 1) / len(Y) < 0.05 or len(Y) < 100:  # in the case of extreamly imbalanced sets we need to do it diffrently
        sample_fraction = 1.0
        n_bootstrap_iterations = 1

    # print("Started")
    clf = StabilitySelection(base_estimator=LogisticRegression(penalty='l1', solver='liblinear'), n_jobs=-1,
                             n_bootstrap_iterations=n_bootstrap_iterations, lambda_grid=np.asarray([1]),
                             sample_fraction=sample_fraction,
                             random_state=RANDOM,
                             bootstrap_func=stratified_bootstrap)

    try:
        clf.fit(features, Y)
    except Exception as e:
        if len(features) < 1000:
            sample_fraction = 0.5
        clf = StabilitySelection(base_estimator=LogisticRegression(penalty='l1', solver='liblinear'), n_jobs=-1,
                                 n_bootstrap_iterations=n_bootstrap_iterations, lambda_grid=np.asarray([1]),
                                 sample_fraction=sample_fraction,
                                 random_state=RANDOM,
                                 bootstrap_func=stratified_bootstrap)
        clf.fit(features, Y)


    # print("Finished")

    print(clf.get_support(indices=True))

    # deprecated version from scikit 0.18
    # clf = RandomizedLogisticRegression()   #n_jobs=-1, n_resampling=50
    # clf.fit(X=features, y=Y)

    end = time.time()
    print("Randomizaed Logistic Regression time: " + str(end - start))

    return maskToProbabilites(clf.get_support())
    pass


def calculate_mutual_info(features, Y):
    start = time.time()

    mutual_info = sk.mutual_info_classif(X=features, y=Y, random_state=RANDOM)
    scores = np.asarray(mutual_info, dtype=float)

    end = time.time()
    print("Mutual info score time: " + str(end - start))

    scores = np.nan_to_num(scores)
    sum_of_scores = np.sum(scores)
    probabilities = np.divide(scores, sum_of_scores)

    return probabilities

def threshold(a, threshmin=None, threshmax=None, newval=0):
    mask = (a < threshmin) | (a > threshmax)
    a1 = a.copy()
    a1[mask] = newval
    return a1



def testTimes(X, Y):
    # for i in range(0,10):
    ss = StabilitySelectionMy(X, Y)
    mi = calculate_mutual_info(X, Y)
    fi = featureImportanceSelection(X,Y)
    vt = varianceThreshold(X,Y)

    if np.isnan(ss).any():
        ss = np.nan_to_num(ss)

    if np.isnan(mi).any():
        mi = np.nan_to_num(mi)

    if np.isnan(fi).any():
        fi = np.nan_to_num(fi)

    if np.isnan(vt).any():
        vt = np.nan_to_num(vt)

    sum_ = (mi + ss + fi + vt)
    sum_ = np.divide(sum_, 4)
    b = {i: sum_[i] for i in range(0, len(sum_))}

    a1_sorted_keys = sorted(b, key=b.get, reverse=True)
    for r in a1_sorted_keys:
        print(r, b[r])

    # thr = np.median(sum_) + sum_.mean()

    filtered = threshold(sum_, threshmin=max(fi) / 2, threshmax=1.0, newval=0)

    if sum(filtered) == 0 or np.count_nonzero(filtered) < 4:
        if sum(sum_) == 0:
            sum_.fill(1 / sum_.shape[0])
            return sum_
        else:
            return sum_
    else:
        return filtered




def fetureImportance(X, Y):
    # for i in range(0,10):
    mi = calculate_mutual_info(X, Y)
    ss = StabilitySelectionMy(X,Y)
    fi = featureImportanceSelection(X,Y)
    vt = varianceThreshold(X,Y)

    sum_ = (mi + ss + fi + vt)
    sum_ = np.divide(sum_, 4)
    b = {i: sum_[i] for i in range(0, len(sum_))}

    # a1_sorted_keys = sorted(b, key=b.get, reverse=True)
    # for r in a1_sorted_keys:
    #     print(r, b[r])

    # thr = np.median(sum_) + sum_.mean()

    # sum_ = threshold(sum_, threshmin=max(fi) / 2, threshmax=1.0, newval=0)

    return sum_


def parse() -> Tuple[str, str]:
    """
    Parse arguments
    :return: str
    """
    parser = argparse.ArgumentParser(description='Mutual information calculation')

    parser.add_argument('-t',
                        action="store",
                        dest="training_set",
                        help='Training set path')

    parser.add_argument('-o',
                        action="store",
                        dest="output_path",
                        help='Output path')

    return parser.parse_args().training_set, parser.parse_args().output_path


def main(argv):
    simplefilter(action='ignore', category=Warning)

    traningSetPath, output_path = parse()
    # traningSetPath, output_path = r"D:\debug\Shuttle__2vs6\4\train.csv", r'D:\debug\Shuttle__2vs6'

    X, Y = loadData(traningSetPath)

    start = time.time()

    probabilites_of_features = testTimes(X,Y) #featureImportanceSelection(X, Y)

    print(probabilites_of_features)

    end = time.time()

    dirToSaveResults = os.path.dirname(traningSetPath)

    f = open(os.path.join(output_path, "timeOfEnsembleFeatures.txt"), "a+")
    f.write(str(end - start))
    f.write("\n")
    f.close()

    np.savetxt(os.path.join(output_path, "probabilites_of_features.txt"),
               X=probabilites_of_features.reshape(1, -1), delimiter=',', fmt='%1.8f')

    pass

if __name__ == "__main__":
    main(sys.argv)