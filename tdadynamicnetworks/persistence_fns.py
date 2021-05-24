''' Feb 28, 2017
BP: Functions relating to computing persistent homology'''

from __future__ import division
import os
import sys
import numpy as np
import gudhi as g
from scipy.spatial.distance import squareform
from scipy import sparse
from ripser import ripser
from joblib import Parallel, delayed
from wasserstein import wasserstein_distance

# get the maximum persistence object given a PD, in array form: an array of list of length 2.
def get_maximum_persistence(PD):
    num_dim = len(PD)

    #has to be a 2D array
    def max_pers(array): 
        if len(array) == 0:
            return 0
        diff = array[:,1] - array[:,0]
        return max(diff)

    return list(map(max_pers,PD))


# turn graph into a valid filtration; max of lamda * node weights and edge weights
def get_filtration(node_wts, edge_wts):
    """
    Turn graph into a valid filtration; each off diagonal entry of the adjacency matrix is the
    maximum of edge weights and node weights
    Parameters
    ----------
    node_wts: ndarray(N)
        A list of node weights
    edge_wts: scipy.sparse(N, N)
        A sparse matrix of edge weights
    """
    assert node_wts.size == edge_wts.shape[0], 'Unequal number of edge wts and node wts'
    N = len(node_wts)
    ew = edge_wts.tocoo()
    row, col, data = ew.row, ew.col, np.array(ew.data)
    data = np.maximum(data, node_wts[row])
    data = np.maximum(data, node_wts[col])
    filtration_matrix = sparse.coo_matrix((data, (row, col)), shape=(N, N))
    #filtration_matrix += sparse.spdiags(node_wts, 0, N, N) # why dont we need this line?
    return filtration_matrix


# get a valid rips filtration
def get_rips_complex(filtration_matrix):
    return ripser(filtration_matrix, distance_matrix=True, maxdim=0)['dgms'][0]


def get_bottleneck_dist(b1, b2, e=0):
    ''' wrapping gudhi in a function for ease;'''
    return g.bottleneck_distance(b1, b2, e)


def get_bottleneck_dist_matrix(barcodes):
    ''' Given a set of barcodes computes the pairwise bottleneck distance and returns the distance
    matrix'''
    # how is the infinity points matched up?
    nBarcodes = len(barcodes)
    dist_matrix = []
    for i in range(nBarcodes):
        print('Computing Row %s' %i)
        bi = np.array(barcodes[i])
        bi = bi[np.isfinite(bi[:, 1]), :]
        for j in range(i + 1, nBarcodes):
            bj = np.array(barcodes[j])
            bj = bj[np.isfinite(bj[:, 1]), :]
            bottle_dist = get_bottleneck_dist(bi, bj)
            dist_matrix.append(bottle_dist)
    return squareform(dist_matrix)


def get_wasserstein_dist_matrix(barcodes, p):
    ''' Given a set of barcodes computes the pairwise bottleneck distance and returns the distance
    matrix'''
    # how is the infinity points matched up?
    nBarcodes = len(barcodes)
    dist_matrix_jobs = []

    barcodes = [np.array(b) for b in barcodes]
    barcodes = [b[np.isfinite(b[:, 1]), :] for b in barcodes]

    for i in range(nBarcodes):
        for j in range(i+1, nBarcodes):
            dist_matrix_jobs.append((barcodes[i],barcodes[j]))

    dist_matrix = Parallel(n_jobs = 4)(delayed(get_wasserstein_dist)(b[0],b[1],p) for b in dist_matrix_jobs)

    return squareform(dist_matrix)

def get_wasserstein_dist(bi,bj,p):
    return wasserstein_distance(bi, bj, order = p) # interal power will be 2 instead of infinity  
