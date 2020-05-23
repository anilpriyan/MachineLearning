import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse


def augment_feature_vector(X):
    """
    Adds the x[i][0] = 1 feature for each data point x[i].

    Args:
        X - a NumPy matrix of n data points, each with d - 1 features

    Returns: X_augment, an (n, d) NumPy array with the added feature for each datapoint
    """
    column_of_ones = np.zeros([len(X), 1]) + 1
    return np.hstack((column_of_ones, X))

def compute_probabilities(X, theta, temp_parameter):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, d) NumPy array, where row j represents the parameters of our model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)
    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    #YOUR CODE HERE
    H = np.matmul(theta, np.transpose(X))/temp_parameter
    H = np.exp(np.apply_along_axis(lambda row: np.subtract(row, np.max(row)), 0, H))
    H = np.apply_along_axis(lambda row: np.divide(row, np.sum(row)), 0, H) 
    return H
    # H = np.zeros([X.shape[0], theta.shape[0]])
    # for i in range(X.shape[0]):
    #     c = np.array([])
    #     for j in range(theta.shape[0]):
    #         c = np.append(c, np.dot(theta[j], X[i]) / temp_parameter)
    #     m = np.max(c)
    #     prob = np.array([])
    #     sum = 0
    #     for j in range(theta.shape[0]):
    #         exponent = np.dot(theta[j], X[i]) / temp_parameter - m
    #         p = np.exp(exponent)        
    #         sum = sum + p 
    #         prob = np.append(prob, p)   
    #     l = prob / sum
    #     H[i] = l
    # return np.transpose(H)
    #raise NotImplementedError

def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Computes the total cost over every datapoint.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns
        c - the cost value (scalar)
    """
    #YOUR CODE HERE
    c = 0
    r = np.array([])
    probs = compute_probabilities(X, theta, temp_parameter)
    for i in range(X.shape[0]):
        for j in range(theta.shape[0]):
            reg = lambda_factor * np.dot(theta[j], theta[j]) / 2 
            c = c + reg
        p = probs[Y[i]][i] 
        if np.abs(p) > 0.01:    
            r = np.append(r, np.log(p))      
    c = c - np.mean(r)        
    return c
    # probs = compute_probabilities(X, theta, temp_parameter)
    # for i in range(X.shape[0]):
    #     for j in range(theta.shape[0]):
    #         reg = lambda_factor * np.dot(theta[j], theta[j]) / 2 
    #         c = c + reg
    #     p = probs[i][Y[i]] 
    #     if np.abs(p) < 0.0001:    
    #         c = c - np.log(p)/X.shape[0]  
    # return c
    #raise NotImplementedError

def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """
    Runs one step of batch gradient descent

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
    """
    #YOUR CODE HERE
    #raise NotImplementedError

    probs = compute_probabilities(X, theta, temp_parameter)
    num_examples = X.shape[0]
    num_labels = theta.shape[0]
    M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    M = np.subtract(M, probs)/(temp_parameter * num_examples)
    F = np.matmul(M,X)
    G = lambda_factor * theta - F
    return theta - alpha * G  

    # probs = compute_probabilities(X, theta, temp_parameter)
    # num_examples = X.shape[0]
    # num_labels = theta.shape[0]
    # M = sparse.coo_matrix(([1]*num_examples, (Y,range(num_examples))), shape=(num_labels,num_examples)).toarray()
    # # M = np.subtract(M, probs)/(temp_parameter * num_labels)
    # # F = np.matmul(M,X)
    # # G = F + lambda_factor * theta
    # # return theta - alpha * G    
    # G = np.zeros(theta.shape)
    # for  j in range(num_labels):
    #     for i in range(num_examples):
    #         G[j] = G[j] - (X[i] * (M[j,i] - probs[j,i]))/(temp_parameter * num_examples)
    #     G[j] = G[j] + lambda_factor * theta[j]
    # return theta - alpha * G     

def update_y(train_y, test_y):
    """
    Changes the old digit labels for the training and test set for the new (mod 3)
    labels.

    Args:
        train_y - (n, ) NumPy array containing the labels (a number between 0-9)
                 for each datapoint in the training set
        test_y - (n, ) NumPy array containing the labels (a number between 0-9)
                for each datapoint in the test set

    Returns:
        train_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                     for each datapoint in the training set
        test_y_mod3 - (n, ) NumPy array containing the new labels (a number between 0-2)
                    for each datapoint in the test set
    """
    #YOUR CODE HERE
    train_y_mod3 = np.mod(train_y, 3)
    test_y_mod3 = np.mod(test_y, 3)
    return train_y_mod3, test_y_mod3   
    #raise NotImplementedError

def compute_test_error_mod3(X, Y, theta, temp_parameter):
    """
    Returns the error of these new labels when the classifier predicts the digit. (mod 3)

    Args:
        X - (n, d - 1) NumPy array (n datapoints each with d - 1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-2) for each
            data point
        theta - (k, d) NumPy array, where row j represents the parameters of our
                model for label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        test_error - the error rate of the classifier (scalar)
    """
    #YOUR CODE HERE
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(np.mod(assigned_labels,3) == np.mod(Y,3))
    #raise NotImplementedError

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array. Here, theta is a k by d NumPy array
    where row j represents the parameters of our model for label j for
    j = 0, 1, ..., k-1

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)

    Returns:
        theta - (k, d) NumPy array that is the final value of parameters theta
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    X = augment_feature_vector(X)
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for i in range(num_iterations):
    #     cost_function_progression.append(compute_cost_function(X, Y, theta, lambda_factor, temp_parameter))
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression

def get_classification(X, theta, temp_parameter):
    """
    Makes predictions by classifying a given dataset

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d - 1 features)
        theta - (k, d) NumPy array where row j represents the parameters of our model for
                label j
        temp_parameter - the temperature parameter of softmax function (scalar)

    Returns:
        Y - (n, ) NumPy array, containing the predicted label (a number between 0-9) for
            each data point
    """
    X = augment_feature_vector(X)
    probabilities = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(probabilities, axis = 0)

def plot_cost_function_over_time(cost_function_history):
    plt.plot(range(len(cost_function_history)), cost_function_history)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def compute_test_error(X, Y, theta, temp_parameter):
    error_count = 0.
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
