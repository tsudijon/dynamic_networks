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
obsfn = lambda t, p: plane.periodic_plane_random_cos_series(t,p,1, seed) # T doesn't matter here

# can add some random time varying rotation, 
# or on torus can add some noise.

lambda1 = 100
lambda2 = lambda1

start = time.time()
#sensor_lifetimes = ctsm.get_fixed_sensors(250, max_lifetime, manifold = 'plane')
sensor_lifetimes = ctsm.get_sensor_lifetimes(1000, max_lifetime, lambda1, lambda2,
											domain_lengths = (1,1), manifold = 'plane', seed = seed)

end = time.time()
print("Sampling Sensor Lifetimes", end - start) 

## Create the Dynamic Network
fac = 1
step_size = 0.05/fac
ts = np.arange(0,max_lifetime,step_size) 

start = time.time()
(node_wts,edges, allpoints) = ctsm.sample_dynamic_geometric_graph(sensor_lifetimes, ts,
                                                                     obsfn = obsfn, manifold = manifold)
end = time.time()
print("Sampling Dynamic Network", end - start) 

import multiprocessing as mp
from joblib import Parallel, delayed

start = time.time()
filtration_matrix = list(map(lambda n, e: pf.get_filtration(n, e), node_wts, edges))
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
d = int(wl/step_size)
swe = sw.sliding_window(range(len(barcodes)), d=d, tau=1, # Dummy time series?
                                    max_index = int(6.0*T/step_size) )

print("Number of points in SW Embedding:", len(swe))
sw_dist_matrix = sw.sw_distance_matrix(swe, bn_dist_matrix)
sw_dist_matrix /= np.max(sw_dist_matrix)

PDs2 = ripser(sw_dist_matrix, distance_matrix=True, maxdim=1, coeff=2)['dgms']
PDs3 = ripser(sw_dist_matrix, distance_matrix=True, maxdim=1, coeff=3)['dgms']
dgm1 = PDs3[1]
max_pers_score = 0
pers_ratio = 0
if dgm1.size > 0:
    max_pers_score = np.max(dgm1[:, 1] - dgm1[:, 0])
if dgm1.size > 1:
    second_max_pers = 
    pers_ratio = second_max_pers / max_pers_score
print(max_pers_score, pers_ratio)

plt.figure(figsize=(10, 5))
plt.subplot(121)
plot_diagrams(PDs3)
plt.subplot(122)
plt.imshow(sw_dist_matrix, cmap='magma_r')
plt.colorbar()
plt.savefig("NetworkDist.png")


plt.figure(figsize=(12, 6))
for i, t in enumerate(ts):
    plt.clf()
    plt.subplot(121)
    X = allpoints[i]
    f = node_wts[i]
    plt.scatter(X[:, 0], X[:, 1], c=f, cmap='magma_r', zorder=10)
    for e in edges[i]:
        x1, y1 = X[e[0], :]
        x2, y2 = X[e[1], :]
        plt.plot([x1, x2], [y1, y2], c='C0', linewidth=1, linestyle='--')
    plt.subplot(122)
    plot_diagrams(barcodes[i])
    plt.savefig("Network{}.png".format(i))