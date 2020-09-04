import scipy.spatial as sp
from scipy.spatial.distance import squareform
from scipy import sparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import itertools as itr
import os
import sys
sys.path.append('../src/')
import graph_fns as gf
import persistence_fns as pf
import sliding_window_fns as sw
from ripser import ripser
from persim import plot_diagrams
from sklearn import manifold
import scipy.io as sio
import itertools as itr

from scipy.spatial.distance import euclidean

"""###########################################
Functions for sampling the grids
###########################################"""

def sample_uniform(N):
    cds = np.random.uniform(0,1,(N,2))
    return cds


"""###########################################
Functions for accessing and maintaining aspects
of the graph
###########################################"""

# simplex_list is an array of arrays
def simplex_list_to_edge_list(simplex_list):
    edges = set()
    for simplex in simplex_list:
        simp_edges = set(itr.combinations(simplex,2)) 
        edges = edges.union(simp_edges)
    return edges

def get_node_wts(t,points, obsfn):
    return obsfn(t,np.array(points))

def get_edge_wts_rgg(points, threshold, alpha = 1.0):
    """
    map edges to their distances, for the random geometric graph model
    use this to create a dynamic network, edges are these distances.
    Params:
    -------
    points: list or array
        coordinates of points on the graph
    threshold:
        radius of connectivity
    alpha: float
        Amount by which to weight distances
    Returns
    -------
    edges: scipy.sparse(N, N)
        A sparse matrix with the edge weights

    """
    v = np.array(points)
    edges = itr.product(range(len(v)),range(len(v)))
    ds = [( euclidean(v[e[0],:], v[e[1],:]),e) for e in edges]

    ## pick out edges less than threshold r
    edges = [e for d,e in ds if (d < threshold)]
    ds = [d for d,e in ds if (d < threshold)]
    ds = alpha*np.array(ds + ds)
    
    e0 = np.array([e[0] for e in edges] + [e[1] for e in edges])
    e1 = np.array([e[1] for e in edges] + [e[0] for e in edges])
    
    return sparse.coo_matrix((ds, (e0, e1)), shape=(len(v), len(v)))

def critical_rgg_scaling(n):
    """
    Critical scaling given in the theory, pay special attention to coefs
    """
    k = 2 # dimension
    r = 1.0
    volume = 4.0*np.pi*r**2 # volume of manifold (sphere surface area)
    tau = 0.1 # condition number assumption
    C = 1.0 # can be whatever
    C_1 = 4^k*volume/((3.0*np.pi)/4.0)
    C_2 = C_1*C**(-k)*np.exp(k*C/(8*tau))

    alpha = 1.0/C_2
    return C*(np.log(alpha*n)/alpha*n)**0.5


"""###########################################
Functions to generate observation functions on the grid
###########################################"""

def convert_cds(cds):
    return cds[:,0], cds[:,1]

### Stationary Periodic
def periodic_plane_stationary(t,cds, T):
    """
    Parameters
    ----------
    t: float
        Time index
    cds: ndarray(N, 3)
        Cartesion coordinates of sphere points
    T: float
        Period of a cycle
    """
    x,y = convert_cds(cds)
    return 2*np.cos(2*x*t/T) + 3*np.sin(3*y*t/T)


### Dynamic Periodic (periodic with some drift)
def periodic_plane_nonstationary(t, cds, T):
    """
    Parameters
    ----------
    t: float
        Time index
    cds: ndarray(N, 3)
        Cartesion coordinates of sphere points
    T: float
        Period of a cycle
    """
    import math
    x,y = convert_cds(cds)
    x += math.modf(np.pi*t)[1]
    y += math.modf(np.pi**2*t)[1]
    return 2*np.cos(2*x*t/T) + 3*np.sin(3*y*t/T)

### Non periodic example. time varying random field?