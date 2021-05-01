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


from ripser import ripser
from persim import plot_diagrams
from sklearn import manifold
import scipy.io as sio
import itertools as itr

from scipy.spatial.distance import euclidean

import multiprocessing as mp
from joblib import Parallel, delayed
sys.path.append('/Users/timothysudijono/projects/Research/dynamic_networks/data/')
from helper import memoize

"""###########################################
Functions for sampling the grids
###########################################"""

def sample_uniform(N, L_x = 1, L_y = 1):
    cds = np.random.uniform((0,0),(L_x,L_y),(N,2))
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

def get_edge_wts_rgg(points, threshold):
    """
    map edges to their birthtimes, for the random geometric graph model
    use this to create a dynamic network, edges are these distances.
    Params:
    -------
    points: list or array
        coordinates of points on the graph
    threshold:
        radius of connectivity
    Returns
    -------
    edges: scipy.sparse(N, N)
        A sparse matrix with the edge weights

    """
    from sklearn.metrics.pairwise import euclidean_distances
    v = np.array(points)
    ds = euclidean_distances(v,v)

    # convert to birthtimes
    ds = np.where(ds < threshold, -np.inf, np.inf)

    return sparse.coo_matrix(ds)

def critical_rgg_scaling(n):
    """
    Critical scaling given in the theory, pay special attention to coefs
    """
    return np.sqrt(np.log(n)/(np.pi*n))

def supercritical_rgg_scaling(n):
    return 1.25*np.sqrt(np.log(n)/(np.pi*n))  

def supercritical_rgg_scaling_circle(n):
    return 10*np.log(n)/(np.pi*n)  

def critical_rgg_scaling_circle(n):
    return np.log(n)/(np.pi*n)  

"""###########################################
Functions to generate observation functions on the grid
###########################################"""

def convert_cds(cds):
    return cds[:,0], cds[:,1]

def lookup_periodic_1D_array(array, t_idx, cds):
    """
    Array is a t x L_x x L_y array, where the first axis is over time.

    cds should points which lie in the domain [0,L_x] times [0,L_y]
    """
    
    n_x = array.shape[1]
    cds = cds*n_x
    cds = cds.astype(int)

    return array[t_idx, cds]

def lookup_periodic_array(array, t_idx, cds):
    """
    Array is a t x L_x x L_y array, where the first axis is over time.

    cds should points which lie in the domain [0,L_x] times [0,L_y]
    """
    
    n_x = array.shape[1]
    n_y = array.shape[2]
    cds[:,0] = cds[:,0]*n_x
    cds[:,1] = cds[:,1]*n_y
    cds = cds.astype(int)

    return array[t_idx, cds[:,0],cds[:,1]]


### Stationary Periodic
def periodic_plane_stationary(t,cds, T):
    """
    Parameters
    ----------
    t: float
        Time index
    cds: ndarray(N, 2)
    T: float
        Period of a cycle
    """
    x,y = convert_cds(cds)
    #return 0.5*np.cos(2*x*t) + 0.3*np.sin(4*y*t) + np.cos(16*x*t) + 0.8*np.sin(8*y*t)
    return 0.25*np.cos(2*np.pi*(t+0.3))*np.cos(2*x) + 0.3*np.cos(2*np.pi*(t+1))*np.sin(4*y) + \
        np.cos(2*np.pi*(t+1.6))*np.cos(16*x) + 0.8*np.cos(2*np.pi*t)*np.sin(8*y)

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


def periodic_plane_random_cos_series(t,cds,T, seed = 17):
    np.random.seed(seed)
    x,y = convert_cds(cds)
    vals = np.zeros_like(x)
    for i in np.arange(5,10):
        for j in np.arange(5,10):
            r = np.random.rand(3)
            vals += np.cos(2*np.pi*(t+r[0])/T)*np.cos((i*x + j*y) + r[1])*r[2]
            #print('$\cos(2\pi(t + {:.2f})/T)\cdot \cos({}x + {}y + {:.2f}) \cdot {:.2f}$'.format(r[0],i,j,r[1],r[2]))
    return vals

def nonstationary_periodic_plane_random_cos_series(t,cds,T, seed = 17):
    np.random.seed(seed)

    cds = np.remainder(t*np.array([[0.05,0.08]]) + cds, 1.0) # apply isometry to coorindates

    x,y = convert_cds(cds)
    vals = np.zeros_like(x)
    for i in np.arange(1,4):
        for j in np.arange(1,4):
            r = np.random.rand(3)
            vals += np.cos(2*np.pi*(t+r[0])/T)*np.cos((i*x + j*y) + r[1])*r[2]
    return vals

#rf1 = sample_Gaussian_field(seed = 17)
#rf2 = sample_Gaussian_field(seed = 289)
#rf2 = -rf1

# can make this even better via interpolation
def periodic_plane_random_field(t, cds, T):
    # initialize the random field once, then just look up.
    r = t%T
    if r <= T/2.:
        field = rf1*(1.-2.*r/T) + rf2*(2.*r/T)
    else:
        field = rf1*(2.*r/T) + rf1*(1-2.*r/T)
    
    # look up cds in field, rounded; should implement this in a more robust way
    idx = 100*np.round(cds,2).astype(int)

    return field[idx]


import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel_mat(mat, var = 1):
    return np.exp(-mat/(2*var))

def gram_matrix(grid):
    xs,ys = np.meshgrid(grid,grid)
    coords = np.array((xs.ravel(),ys.ravel())).T

    diffs = coords[:,np.newaxis] - coords
    return rbf_kernel_mat((diffs**2).sum(axis = 2), 0.001)

@memoize
def sample_Gaussian_field(seed = 17):

    grid = np.arange(0, 1, 0.01)
    mean = np.zeros(len(grid)**2)
    gram = gram_matrix(grid)
    field = np.random.multivariate_normal(mean, gram, seed).reshape(len(grid),len(grid))
    return field


### Non periodic example. time varying random field?
from mpl_toolkits.mplot3d import Axes3D  
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def plot_field(grid, field):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    xs,ys = np.meshgrid(grid,grid)
    surf = ax.plot_surface(xs, ys,
                            field,
                        cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-2.0, 2.0)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def get_edge_wts_rgg_torus(points, threshold):
    """
    map edges to their birthtimes, for the random geometric graph model
    use this to create a dynamic network, edges are these distances.
    Params:
    -------
    points: list or array
        coordinates of points on the graph
    threshold:
        radius of connectivity
    Returns
    -------
    edges: scipy.sparse(N, N)
        A sparse matrix with the edge weights

    """
    ### It might be nice to have a vectorized version of this function.
    from scipy.spatial.distance import pdist, squareform
    def smaller_xy(p,q):
        xdiff = abs(p[0] - q[0])
        if xdiff > 0.5:
            xdiff = 1.0 - xdiff

        ydiff = abs(p[1] - q[1])
        if ydiff > 0.5:
            ydiff = 1.0 - ydiff

        return xdiff**2 + ydiff**2
    ds = squareform(pdist(points, smaller_xy))

    # convert to birthtimes
    ds = np.where(ds < threshold**2, -np.inf, np.inf)

    return sparse.coo_matrix(ds)

def get_edge_wts_rgg_circle(points, threshold): 
    """
    map edges to their birthtimes, for the random geometric graph model
    use this to create a dynamic network, edges are these distances.
    Params:
    -------
    points: list or array
        coordinates of points on the graph
    threshold:
        radius of connectivity
    Returns
    -------
    edges: scipy.sparse(N, N)
        A sparse matrix with the edge weights

    """
    ### It might be nice to have a vectorized version of this function.
    N = len(points)
    ds = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            ds[i,j] = np.abs(points[i] - points[j])

    ds[ds > 0.5] = 1. - ds[ds > 0.5]
    # convert to birthtimes
    ds = np.where(ds < threshold, -np.inf, np.inf)

    return sparse.coo_matrix(ds)

def get_edge_wts_rgg_interval(points, threshold): 
    """
    map edges to their birthtimes, for the random geometric graph model
    use this to create a dynamic network, edges are these distances.
    Params:
    -------
    points: list or array
        coordinates of points on the graph
    threshold:
        radius of connectivity
    Returns
    -------
    edges: scipy.sparse(N, N)
        A sparse matrix with the edge weights

    """
    ### It might be nice to have a vectorized version of this function.
    N = len(points)
    ds = np.zeros((N,N))
    
    for i in range(N):
        for j in range(N):
            ds[i,j] = np.abs(points[i] - points[j])

    # convert to birthtimes
    ds = np.where(ds < threshold, -np.inf, np.inf)

    return sparse.coo_matrix(ds)