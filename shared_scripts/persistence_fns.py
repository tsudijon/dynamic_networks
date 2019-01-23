''' Feb 28, 2017
BP: Functions relating to computing persistent homology'''

from __future__ import division
import os
import sys
import numpy as np
gudhi_path = os.path.expanduser('~/projects/tda_libraries/GUDHI/build/cython/')
sys.path.append(gudhi_path)
import dionysus as d
import gudhi as g
from scipy.spatial.distance import squareform

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
    ''' Turn graph into a valid filtration; each off diagonal entry of the adjacency matrix is the
    maximum of edge weights and node weights'''
    adjacency_matrix = squareform(edge_wts)
    np.fill_diagonal(adjacency_matrix, node_wts)
    filtration_matrix = np.zeros_like(adjacency_matrix)

    for i in range(len(adjacency_matrix)):
        for j in range(len(adjacency_matrix)):
            filtration_matrix[i, j] = np.max((adjacency_matrix[i, i], adjacency_matrix[j, j],
                                             adjacency_matrix[i, j]))
    return filtration_matrix


# get a valid rips filtration
def get_rips_complex(filtration_matrix, max_dim=1):
    ''' get a rips filtration after turning the distance matrix into a valid filtration'''
    filtered_edge_wts = filtration_matrix[np.triu_indices_from(filtration_matrix,
                                                               k=1)].astype(float)
    filtered_node_wts = filtration_matrix[np.diag_indices_from(filtration_matrix)].astype(float)

    rips = d.fill_rips(filtered_edge_wts, max_dim, 2000)

    # change the node birth time; dionsysus sets it to 0
    for i, simplex in enumerate(rips):
        if simplex.dimension() == 0:
            rips[i] = d.Simplex([simplex[0]], filtered_node_wts[i])
    return rips


def get_bottleneck_dist(b1, b2, e=0):
    ''' wrapping gudhi in a function for ease;'''
    return g.bottleneck_distance(b1, b2, e)


def get_bottleneck_dist_matrix(barcodes):
    ''' Given a set of barcodes computes the pairwise bottleneck distance and returns the distance
    matrix'''
    nBarcodes = len(barcodes)
    dist_matrix = []
    for i in range(nBarcodes):
        for j in range(i + 1, nBarcodes):
            bottle_dist = get_bottleneck_dist(barcodes[i], barcodes[j])
            dist_matrix.append(bottle_dist)
    return squareform(dist_matrix)
