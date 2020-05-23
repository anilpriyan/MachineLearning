import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

X = X_gold

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here
mixture, post = common.init(X, K, seed)
mixture, post, l = em.run(X, mixture, post)
bic = common.bic(X, mixture, l)
print("bic = ", bic)
# title = "Incomplete - > K = {}, seed = {},  log likelyhood = {}, bic = {} plot.png".format(K, seed, int(l), int(bic))
title = "test log plot"
common.plot(X, mixture, post, title) 