''' August 6, 2017
Test pipeline on a synthetic MIT Dataset '''
from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import copy

dionysus_path = os.path.abspath('./Downloads/dionysus/build/bindings/python/')
gudhi_path = os.path.abspath('./Downloads/Gudhi.0/build/cython/')
pipeline_path = os.path.abspath('./Downloads/pyRipser/ICERM_project/')
ripser_path = os.path.abspath('./Downloads/pyRipser/')

sys.path.append(gudhi_path)
sys.path.append(dionysus_path)
sys.path.append(pipeline_path)
sys.path.append(ripser_path)
import DGMTools as dgmt
import TDA as tda
import pipeline_fns as pf
import TDAPlotting as tdap
import dionysus as dnsys
import gudhi as gudhi
import spike_to_graphs_fns as stg

import numpy as np
from scipy.spatial.distance import squareform
import xlrd


# read in the MIT data;
seq1 = [[0], [1], [1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 0, 0], [1], [0]]
seq2 = [[0], [1], [0, 0, 1, 1, 0, 1], [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 0, 0, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 0, 1, 1], [1, 0, 0, 1, 1, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1], [1], [0]]

distance_list = seq1 + seq2

# dist_list = [[epsilon if x == 0 else x for x in d] for d in distance_list]

node_weights_list = []
for i, c in enumerate(distance_list):
    node_weights_list.append(np.zeros(squareform(c).shape[0]).tolist())

# get barcodes for the graphs
lamda = 1
max_filtration = 0
barcodes, birth_matrices, birth_graphs, orig_graphs, dgms = \
    pf.vectorize_get_graph_barcodes(distance_list, node_weights_list, lamda=lamda,
                                    filtr_max_dim=max_filtration)

# bottleneck distance matrix
tmp_barcodes = copy.deepcopy(barcodes)
for code in tmp_barcodes:
    code[0][code[0] == np.inf] = 1

bottleneck_dist_matrix = pf.get_bottleneck_distance_matrix(tmp_barcodes)

# run sliding window on the barcodes
d = 5
tau = 2
shift = 1
SW, SW_indices = pf.get_sliding_window(barcodes, d=d, tau=tau, shift=shift)


# get the distance matrix for the barcode vectors
pers_distance_matrix = pf.get_persistence_vector_dist_matrix(SW_indices, bottleneck_dist_matrix)

# # run mds on the persistence matrix
# a = copy.deepcopy(pers_distance_matrix)
# a[a == 0] = 0.001
# a[a == np.inf] = 0
# mds = manifold.MDS(n_components=2, n_init=4, dissimilarity='precomputed', metric=False)
# pers_coords = mds.fit_transform(a)
# x, y = np.split(pers_coords, 2, axis=1)

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.title('MDS of persistence barcode vectors')
# ax.scatter(x, y, s=30)


# run ripser on the SW embedding
maxdim = 2
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)

fig = plt.figure(figsize=(16, 10))
plt.suptitle('MIT dataset %d d, %d tau, Z/%d Z coeff' % (d, tau, coeff))
ax1 = fig.add_subplot(1, 3, 1)
plt.title('H0')
plt.xlim(-0.5, 2)
plt.ylim(-0.5, 2)
tdap.plotDGM(pds[0])

ax2 = fig.add_subplot(1, 3, 2)
plt.title('H1')
plt.xlim(-0.5, 2)
plt.ylim(-0.5, 2)
tdap.plotDGM(pds[1])

ax3 = fig.add_subplot(1, 3, 3)
plt.title('H2')
plt.xlim(-0.5, 2)
plt.ylim(-0.5, 2)
tdap.plotDGM(pds[2])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/Users/biraj/Downloads/mit_data/plots/synthetic/MIT_%dd_%dtau.png' % (d, tau))
