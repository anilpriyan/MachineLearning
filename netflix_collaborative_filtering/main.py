import numpy as np
import kmeans
import common
import naive_em
import em
import embak3


X = np.loadtxt("toy_data.txt")
# X = np.loadtxt("netflix_incomplete.txt")
#X = np.loadtxt("incomplete_mine.txt")

# TODO: Your code here
# for K in 1,12:
#     for seed in range(0,5):
#         print("K = {}, seed = {}".format(K, seed))
# K = 1
# seed = 0
        # mixture, post = common.init(X, K, seed)
        # mixture, post, cost = kmeans.run(X, mixture, post)
        # title = "K = {}, seed = {},  cost = {} plot.png".format(K, seed, int(cost))
        # common.plot(X, mixture, post, title)

K = 3
seed = 0
mixture, post = common.init(X, K, seed)
# mixture, post, l = naive_em.run(X, mixture, post)
mixture, post, l = em.run(X, mixture, post)
# mixture, post, l = embak3.run(X, mixture, post)
# bic = common.bic(X, mixture, l)
# print("bic = ", bic)
# title = "K = {}, seed = {},  log likelyhood = {} plot.png".format(K, seed, int(l))
# common.plot(X, mixture, post, title)   
print(em.fill_matrix(X, mixture))
             