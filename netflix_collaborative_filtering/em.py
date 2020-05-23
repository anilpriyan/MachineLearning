"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    #raise NotImplementedError
def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """
    n, _ = X.shape
    K, _ = mixture.mu.shape
    post = np.zeros((n, K))

    ll = 0
    for i in range(n):
        mask = (X[i, :] != 0)
        for j in range(K):
            log_likelihood = log_gaussian(X[i, mask], mixture.mu[j, mask],
                                          mixture.var[j])
            post[i, j] = np.log(mixture.p[j] + 1e-16) + log_likelihood
        total = logsumexp(post[i, :])
        post[i, :] = post[i, :] - total
        ll += total

    return np.exp(post), ll


def log_gaussian(x: np.ndarray, mean: np.ndarray, var: float) -> float:
    """Computes the log probablity of vector x under a normal distribution

    Args:
        x: (d, ) array holding the vector's coordinates
        mean: (d, ) mean of the gaussian
        var: variance of the gaussian

    Returns:
        float: the log probability
    """
    d = len(x)
    log_prob = -d / 2.0 * np.log(2 * np.pi * var)
    log_prob -= 0.5 * ((x - mean)**2).sum() / var
    return log_prob  


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
     #raise NotImplementedError
    def marginal_mu(x, post, mu):
        n_users, n_movies = x.shape
        n_clusters = post.shape[1]
        c_m = np.zeros((n_clusters, n_movies))
        p_m = np.zeros((n_clusters, n_movies))
        m_mu = np.zeros((n_clusters, n_movies))
        for k in range(n_clusters):
            for i in range(n_movies): 
                for u in range(n_users):   
                    if(x[u,i] == 0):
                        continue
                    c_m[k, i] = c_m[k, i] + post[u, k] * x[u, i]
                    p_m[k, i] = p_m[k, i] + post[u, k]
                if(p_m[k, i] < 1):
                    m_mu[k, i] = mu[k, i]
                else:
                    m_mu[k, i] = c_m[k, i]/p_m[k, i]                          
        return m_mu        

    def squared_distance(x, mu):
        sd = np.square(np.linalg.norm(x-mu))
        return sd

    def marginal_var(x, post, mu):
        n_users, n_movies = x.shape
        n_clusters = post.shape[1]
        c_sd = np.zeros(n_clusters)
        c_n = np.zeros(n_clusters)
        m_var = np.zeros(n_clusters)
        for k in range(n_clusters):
            for u in range(n_users): 
                c_u = np.array([])
                c_mu = np.array([])
                for i in range(n_movies):                  
                    if(x[u,i] == 0):
                        continue
                    c_u = np.append(c_u, x[u, i])
                    c_mu = np.append(c_mu, mu[k, i])
                c_sd[k] = c_sd[k] + post[u, k] * squared_distance(c_u, c_mu) 
                c_n[k] = c_n[k] + c_u.shape[0] * post[u, k] 
            m_var[k] = np.maximum(c_sd[k]/c_n[k], min_variance)                       
        return m_var  

    
    mu = marginal_mu(X, post, mixture.mu)
    var = marginal_var(X, post, mu)

    K = post.shape[1]
    n = X.shape[0]
    p = np.ndarray(K)

    for k in range(K):
        p[k] = np.sum(post.T[k])

    return GaussianMixture(mu, var, p/n)


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
        mixture = mstep(X, post, mixture)
        print(l)

    return(mixture, post, l)


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    def squared_distance(x, mu):
        # sd = np.square(np.linalg.norm(x-mu, 2, 1))
        sd = np.square(np.linalg.norm(x-mu))
        return sd

    def gaussian_probability(x, mu, var): 
        return ((1/(2 * np.pi * var))**(0.5 * mu.shape[0]) * np.exp(-1/(2 * var ) * squared_distance(x, mu)))   
   
    def marginal_gaussian_probability(x, mu, var):
        n_users, n_movies = x.shape
        n_soft_counts = np.zeros(n_users)
        for u in range(n_users):
            c_u = np.array([])
            c_mu = np.array([])
            for i in range(n_movies):
                if(x[u,i] == 0):
                    continue
                c_u = np.append(c_u, x[u, i])
                c_mu = np.append(c_mu, mu[i])
            n_soft_counts[u] = gaussian_probability(c_u, c_mu, var)
        return n_soft_counts        

    K = mixture.mu.shape[0]
    s = X.shape[0]
    n = np.zeros((s, K))

    for k in range(K):
        n[:, k] = marginal_gaussian_probability(X, mixture.mu[k], mixture.var[k]) 

    p_theta = np.matmul(n, mixture.p)

    p_j_i = np.zeros((K,s))
    f_u_j = np.zeros((K,s))
    log_l = np.ndarray((K,s))
    log_l_k = np.ndarray(K)


    for k in range(K):
        p_j_i[k] = (mixture.p[k] * n[:,k])/p_theta  
    
    n, d = X.shape
    Y = np.zeros((n, d))
    K = p_j_i.T.shape[1]
    for u in range(n):
        u_c = np.argmax(p_j_i.T[u, :])
        for i in range(d):
            if(X[u, i] == 0):
                Y[u, i] = np.dot(mixture.mu.T[i, :], p_j_i.T[u, :])
            else:
                Y[u, i] = X[u, i]
    return Y

 

