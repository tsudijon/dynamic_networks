''' August 4, 2017
Test pipeline on MIT Dataset '''
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
binsize = '30mins'  # hours
data_path = '/Users/biraj/Downloads/mit_data/'
book = xlrd.open_workbook(data_path + 'reality_commons_data_weighted_%s.xlsx' % binsize)
sheet = book.sheet_by_name('reality_commons_data_30_min')
data = np.array([[int(sheet.cell_value(r, c)) for c in range(sheet.ncols)
                if sheet.cell_value(r, c) != '']
                for r in range(sheet.nrows)])

distance_list = []
for row in data:
    shape = int(np.sqrt(len(row)))  # size of the whole adjacency matrix
    sq_matrix = np.reshape(row, (shape, shape))  # but diagonal is 1
    np.fill_diagonal(sq_matrix, 0)  # fill diag with 0
    distance_list.append(squareform(sq_matrix))

node_weights_list = []
for i, c in enumerate(distance_list):
    node_weights_list.append(np.zeros(squareform(c).shape[0]).tolist())

# get barcodes for the graphs
lamda = 1
max_filtration = 0
barcodes, birth_matrices, birth_graphs, orig_graphs, _ = \
    pf.vectorize_get_graph_barcodes(distance_list, node_weights_list, lamda=lamda,
                                    filtr_max_dim=max_filtration)

# permute
t = np.random.randint(0, len(barcodes), len(barcodes))
permuted_barcodes = np.array(barcodes)[t].tolist()

# run sliding window on the barcodes
d = 5
tau = 1
shift = 1
SW, SW_indices = pf.get_sliding_window(permuted_barcodes, d=d, tau=tau, shift=shift)
bottleneck_dist_matrix = pf.get_bottleneck_distance_matrix(permuted_barcodes)

# get the distance matrix for the barcode vectors
pers_distance_matrix = pf.get_persistence_vector_dist_matrix(SW_indices, bottleneck_dist_matrix)

# run mds on the persistence matrix
a = copy.deepcopy(pers_distance_matrix)
# a[a == 0] = 0.001
# a[a == np.inf] = 0
mds = manifold.MDS(n_components=3, n_init=4, dissimilarity='precomputed', metric=True)
pers_coords = mds.fit_transform(a)
x, y, z = np.split(pers_coords, 3, axis=1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title('MDS of persistence barcode vectors')
ax.scatter(x, y, z, c='r', s=30)


# run ripser on the SW embedding
maxdim = 2
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)

# print max persistence
a = pds[1]
print(np.array(sorted(a[:, 1] - a[:, 0]))[-20:])

fig = plt.figure(figsize=(16, 10))
plt.suptitle('MIT dataset %d d, %d tau, Z/%d Z coeff' % (d, tau, coeff))
ax1 = fig.add_subplot(1, 3, 1)
plt.title('H0')
# plt.xlim(-0.5, 2)
# plt.ylim(-0.5, 2)
tdap.plotDGM(pds[0])

ax2 = fig.add_subplot(1, 3, 2)
plt.title('H1')
# plt.xlim(-0.5, 2)
# plt.ylim(-0.5, 2)
tdap.plotDGM(pds[1])

ax3 = fig.add_subplot(1, 3, 3)
plt.title('H2')
# plt.xlim(-0.5, 2)
# plt.ylim(-0.5, 2)
tdap.plotDGM(pds[2])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/Users/biraj/Desktop/Presentations/figures/MIT.png')







