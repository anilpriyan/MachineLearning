import numpy as np
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    model = LinearSVC()
    model.random_state = 0
    model.C = 0.1
    model.fit(train_x, train_y)
    pred_test_y = model.predict(test_x)
    return pred_test_y
    #raise NotImplementedError


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    svc = LinearSVC()
    svc.random_state = 0
    svc.C = 0.1
    svc.fit(train_x, train_y)
    pred_test_y = svc.predict(test_x)
    return pred_test_y
    #raise NotImplementedError


def compute_test_error_svm(test_y, pred_test_y):
    #raise NotImplementedError
    return 1 - np.mean(pred_test_y == test_y)