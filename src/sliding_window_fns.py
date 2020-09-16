''' Feb 28, 2017
BP: Functions related to computing the sliding window'''

from __future__ import division
import numpy as np
import sys
import os
import numpy.linalg as la
from scipy.spatial.distance import squareform
import persistence_fns as pf


def sliding_window(time_series, d, tau):
    ''' Given a time series, return [f(t), f(t + tau), f(t + 2 * tau), ..., f(t + d * tau)];
    i.e. return a sliding window embedding'''
    sw_embedding = []
    for i in range(len(time_series) - ((d - 1) * tau)):  # -1 b/c python loop ends at last index
        sw_embedding.append(time_series[i:i + d * tau:tau])
    return sw_embedding


# def function to get distance matrix from sliding window vectors
def sw_distance_matrix(sw_vecs, bn_dist_matrix=None, rescale = None):
    ''' Get the L2 distance between sliding window vectors of barcodes; if bottleneck distance
    matrix is given, sliding window index is sufficent; otherwise, bottleneck dist needs to be
    calculated for each corresponding component of the sw vectors; is costlier'''
    nVecs = len(sw_vecs)
    if bn_dist_matrix is None:
        dist_matrix = []
        for i in range(nVecs):
            sw_vec1 = sw_vecs[i]
            for j in range(i + 1, nVecs):
                sw_vec2 = sw_vecs[j]
                dist = la.norm(list(map(lambda x, y: pf.get_bottleneck_dist(x, y), sw_vec1, sw_vec2)))
                dist_matrix.append(dist)
    elif bn_dist_matrix is not None:
        dist_matrix = []
        for i in range(nVecs):
            sw_vec1 = sw_vecs[i]
            for j in range(i + 1, nVecs):
                sw_vec2 = sw_vecs[j]
                dist = la.norm(list(map(lambda x, y: bn_dist_matrix[x][y], sw_vec1, sw_vec2)))
                dist_matrix.append(dist)

    squared_dist_matrix = squareform(dist_matrix)

    if rescale:
        squared_dist_matrix = squared_dist_matrix*rescale/np.max(squared_dist_matrix)
    return squared_dist_matrix





