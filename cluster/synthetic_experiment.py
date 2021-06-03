from __future__ import division
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import importlib
import scipy.spatial as sp

## Load in sphere code
import tdadynamicnetworks.examples.PlaneExample as plane
import tdadynamicnetworks.examples.ContinuousTimeSensorModel as ctsm

## Load in d.n. analysis code
import tdadynamicnetworks.graph_fns as gf
import tdadynamicnetworks.persistence_fns as pf
import tdadynamicnetworks.sliding_window_fns as sw
from ripser import ripser
from persim import plot_diagrams
import time

manifold = 'torus'
seed = 25

## Setup Sensor Lifetimes 
max_lifetime = 4

T = 1
#obsfn = lambda t, p: plane.periodic_plane_stationary(t,p,1) # T doesn't matter here
obsfn = lambda t, p: plane.nonstationary_periodic_plane_random_cos_series(t,p,T, seed=seed) # add some sort of isometry. 
# seed 16 = mobius

# can add some random time varying rotation, 
# or on torus can add some noise.

lambda1 = 5
lambda2 = lambda1

start = time.time()
sensor_lifetimes = ctsm.get_sensor_lifetimes(1500, max_lifetime, lambda1, lambda2,
											domain_lengths = (1,1), manifold = 'plane', seed = seed)

end = time.time()
print("Sampling Sensor Lifetimes", end - start) 

## Create the Dynamic Network
fac = 5
step_size = 0.05/fac
ts = np.arange(0,max_lifetime,step_size) 

start = time.time()
(node_wts,edge_wts, allpoints) = ctsm.sample_dynamic_geometric_graph(sensor_lifetimes, ts,
                                                                     obsfn = obsfn, manifold = manifold)
end = time.time()
print("Sampling Dynamic Network", end - start) 

import multiprocessing as mp
from joblib import Parallel, delayed

start = time.time()
filtration_matrix = list(map(lambda n, e: pf.get_filtration(n, e), node_wts, edge_wts))
end = time.time()
print("Converting to filtration matrices", end - start) 

start = time.time()
num_cores = mp.cpu_count() - 4
barcodes = Parallel(n_jobs = num_cores)(delayed(pf.get_rips_complex)(filt) for filt in filtration_matrix)
end = time.time()
print("Computing barcodes", end - start) 

start = time.time()
bn_dist_matrix = pf.get_bottleneck_dist_matrix(barcodes)
end = time.time()
print("Computing bottleneck", end - start) 

wl = 2.0*T
d = int(wl/(step_size))
swe = sw.sliding_window(range(len(barcodes)), d=d, tau=1, # Dummy time series?
                                    max_index = int(6.0*T/step_size) )

print("Number of points in SW Embedding:", len(swe))
sw_dist_matrix = sw.sw_distance_matrix(swe, bn_dist_matrix)
sw_dist_matrix /= np.max(sw_dist_matrix)

PDs2 = ripser(sw_dist_matrix, distance_matrix=True, maxdim=1, coeff=2)['dgms']
PDs3 = ripser(sw_dist_matrix, distance_matrix=True, maxdim=1, coeff=3)['dgms']
dgm1 = PDs3[1]
score = 0
if dgm1.size > 0:
    score = np.max(dgm1[:, 1] - dgm1[:, 0])
print(score)

if PDs3[1].shape[0] > 0:
	print('smallest birth time:', min(PDs3[1][:,0]))

plt.figure(figsize=(10, 10))
plt.subplot(221)
plot_diagrams(PDs3)
plt.subplot(222)
plt.imshow(sw_dist_matrix, cmap='magma_r')
plt.subplot(223)
plt.hist(sw_dist_matrix.flatten(), bins = 20)
plt.colorbar()
plt.show()