''' February 19, 2018
BP: Rewriting the functions to detect periodicity in dynamic graphs; functions to turn graphs
to valid filtration'''

from __future__ import division
import numpy as np
import dionysus as d
from scipy.spatial.distance import squareform


def shifted_arctan(x):
    ''' a possible phi function'''
    return 1 - np.arctan(x)


def shifted_inv_fn(x):
    ''' a possible phi function'''
    return 1 / (x + 1)


def exp_decay_fn(a, x):
    ''' a possible phi function'''
    return a ** (-x)


def softplus(x):
    ''' a possible phi function'''
    return np.log(1 + np.exp(x))


def relu(x):
    ''' a possible phi function'''
    return np.max(0, x)


# pass node weights and edge weights thru the phi function
def weight_fn(node_wts, edge_wts, lamda, phi='softplus'):
    if phi == 'softplus':
        phi_fn = lambda x: softplus(x)
    elif phi == 'relu':
        phi_fn = lambda x: relu(x)
    else:
        print('Phi fn not recognized')

    phi_node_wts = np.array(map(phi_fn, lamda * node_wts))
    phi_edge_wts = np.array(map(phi_fn, edge_wts))
    return phi_node_wts, phi_edge_wts


# turn graph into a valid filtration; max of node weights and edge weights
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
    ''' get rips filtration'''
    filtered_edge_wts = filtration_matrix[np.triu_indices_from(filtration_matrix,
                                                               k=1)].astype(float)
    filtered_node_wts = filtration_matrix[np.diag_indices_from(filtration_matrix)].astype(float)

    rips = d.fill_rips(filtered_edge_wts, max_dim, 2000)

    # change the node birth time; dionsysus sets it to 0
    for i, simplex in enumerate(rips):
        if simplex.dimension() == 0:
            rips[i] = d.Simplex([simplex[0]], filtered_node_wts[i])
    return rips


# sliding window function





