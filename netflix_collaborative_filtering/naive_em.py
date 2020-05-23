"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
import common
import math



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    def squared_distance(x, mu):
        sd = np.square(np.linalg.norm(x-mu, 2, 1))
        return sd
   
    def gaussian_probability(x, mu, var):
        return ((1/(2 * np.pi * var))**(0.5 * x.shape[1]) * np.exp(-1/(2 * var ) * squared_distance(x, mu))) 

    K = mixture.mu.shape[0]
    s = X.shape[0]
    n = np.zeros((s, K))

    for k in range(K):
        n[:, k] = gaussian_probability(X, mixture.mu[k], mixture.var[k]) 

    p_theta = np.matmul(n, mixture.p)

    p_j_i = np.zeros((K,s))
    l = np.ndarray(K)
    for k in range(K):
        p_j_i[k] = (mixture.p[k] * n[:,k])/p_theta
        l[k] = np.sum(p_j_i[k] * np.log(mixture.p[k] * n[:,k]/p_j_i[k]))

    return p_j_i.T, np.sum(l)


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    #raise NotImplementedError
    def squared_distance(x, mu):
        sd = np.square(np.linalg.norm(x-mu, 2, 1))
        return sd

    # mu = np.matmul(post.T, X)/np.sum(post)

    K = post.shape[1]
    d = X.shape[1]
    n = X.shape[0]
    var = np.ndarray(K)
    p = np.ndarray(K)
    mu = np.ndarray((K,d))
    for k in range(K):
        mu[k] = np.dot(X.T, post.T[k])/np.sum(post.T[k])
        var[k] = np.dot(post.T[k], squared_distance(X, mu[k]))/(np.sum(post.T[k]) * d)
        p[k] = np.sum(post.T[k])

    return GaussianMixture(mu, var, p/np.sum(p))


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    #raise NotImplementedError
    prev_l = None
    l = None
    while(prev_l is None or l - prev_l > np.abs(l) * 10**-6):
        prev_l = l
        post, l = estep(X, mixture)
        mixture = mstep(X, post)
        print(l)

    return(mixture, post, l)    
