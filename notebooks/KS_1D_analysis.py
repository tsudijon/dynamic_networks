from __future__ import division
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import squareform
import importlib
import scipy.spatial as sp

## Load in d.n. analysis code
sys.path.append('../src/')
import graph_fns as gf
import persistence_fns as pf
import sliding_window_fns as sw
from ripser import ripser
from persim import plot_diagrams
from sklearn import manifold

## Load in sphere code
sys.path.append('../data/simple_egs/')
import PlaneExample as plane
import ContinuousTimeSensorModel as ctsm
importlib.reload(ctsm)
importlib.reload(plane)
importlib.reload(pf)
import multiprocessing as mp
from joblib import Parallel, delayed
from matplotlib.lines import Line2D
import time 


soln = scipy.io.loadmat('real_data_analyses/Kuramoto/KS_20xT.mat')

## Setup Sensor Lifetimes 
max_lifetime = 5000

## get the time varying isometry here to p.
obsfn = lambda t, p: plane.lookup_periodic_1D_array(soln['data'], t, p)  # Create function which looks up values.

lambda1 = 5
lambda2 = lambda1

start = time.time()
sensor_lifetimes = ctsm.get_sensor_lifetimes(250, max_lifetime, lambda1, lambda2, manifold = 'circle')

end = time.time()
print("Sampling Sensor Lifetimes", end - start) 

## Create the Dynamic Network
step_size = 2 ## maybe should be some multiple of this
ts = np.arange(0,max_lifetime,step_size).astype(int)

start = time.time()
(node_wts,edge_wts, allpoints) = ctsm.sample_dynamic_geometric_graph(sensor_lifetimes, ts,
                                                                     obsfn = obsfn, manifold = 'circle')
end = time.time()
print("Sampling Dynamic Network", end - start) 

start = time.time()
filtration_matrix = list(map(lambda n, e: pf.get_filtration(n, e), node_wts, edge_wts))
end = time.time()
print("Converting to filtration matrices", end - start) 

start = time.time()
num_cores = mp.cpu_count() - 4
barcodes = Parallel(n_jobs = num_cores)(delayed(pf.get_rips_complex)(filt) for filt in filtration_matrix)
end = time.time()
print("Computing barcodes", end - start)


import pickle
#save the bn_dist_matrix
with open('real_data_analyses/barcodes_KS1D.pkl', 'wb') as f:
    pickle.dump(barcodes, f) 

start = time.time()
w2_dist_matrix = pf.get_wasserstein_dist_matrix(barcodes,2)
end = time.time()
print("Computing bottleneck", end - start) 

#save the bn_dist_matrix
with open('real_data_analyses/w2_dist_matrix_KS1D.pkl', 'wb') as f:
    pickle.dump(w2_dist_matrix, f) 

