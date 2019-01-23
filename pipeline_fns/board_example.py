from __future__ import division
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # fix this issue with the python backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold

with open(os.path.expanduser("~/path_to_dyn_networks.txt")) as pathfile:
    project_dir = pathfile.readline()
    project_dir = project_dir.rstrip()

# import required tda packages
pyRipser_path = project_dir + 'dynamic_networks/pyRipser/'
dionysus_path = project_dir + 'tda_libraries/dionysus/build/bindings/python'
gudhi_path = project_dir + 'tda_libraries/gudhi/build/cython/'
# import the main functions
pipeline_path = project_dir + 'dynamic_networks/pipeline_fns/'

sys.path.append(pipeline_path)
sys.path.append(pyRipser_path)
sys.path.append(dionysus_path)
sys.path.append(gudhi_path)
import dionysus as dnsys
import gudhi as g
import pipeline_fns as pf
import TDA as tda
import TDAPlotting as tdap


# load in the network
edge_weights = [list(0.1 * np.ones(6)) for _ in range(0, 5)]
node_weights = [[30, 20, 10, 0], [10, 0, 30, 20], [0, 10, 20, 30], [20, 30, 0, 10], [30, 20, 10, 0]]

# straight line homotopy to interpolate
interp_edge_weights, interp_node_weights = pf.interpolate_graphs(edge_weights, node_weights, 6)


# get barcodes for the graphs
lamda = 1
max_filtration = 0
barcodes, birth_matrices, birth_graphs, orig_graphs, dgms = \
    pf.vectorize_get_graph_barcodes(interp_edge_weights, interp_node_weights, lamda=lamda,
                                    filtr_max_dim=max_filtration)

# run sliding window on the barcodes
d = 10
tau = 1
shift = 1
SW, SW_indices = pf.get_sliding_window(barcodes, d=d, tau=tau, shift=shift)
bottleneck_dist_matrix = pf.get_bottleneck_distance_matrix(barcodes)

# get the distance matrix for the barcode vectors
pers_distance_matrix = pf.get_persistence_vector_dist_matrix(SW_indices, bottleneck_dist_matrix)


# run ripser on the SW embedding
maxdim = 2
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)

# plot
mp1 = np.max(pds[1][:, 1] - pds[1][:, 0])
mp0 = np.max(pds[0][:, 1] - pds[0][:, 0])
fig = plt.figure(figsize=(16, 10))
plt.suptitle('Board example dataset %d d, %d tau, Z/%d Z coeff' % (d, tau, coeff))
ax1 = fig.add_subplot(1, 2, 1)
plt.title('H0, mp0=%0.2f' % mp0)
tdap.plotDGM(pds[0])

ax2 = fig.add_subplot(1, 2, 2)
plt.title('H1, mp1=%0.2f' % mp1)
tdap.plotDGM(pds[1])


# mds
mds = manifold.MDS(3,dissimilarity = 'precomputed')
points = mds.fit_transform(pers_distance_matrix).T
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111, projection='3d')
#ax = Axes3D(fig)
ax.scatter(points[0],points[1],points[2], c = points[2],cmap=plt.cm.rainbow)
plt.show()


