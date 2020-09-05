''' July 26, 2017
Compare SW embedding for changing graphs after feature extract '''
from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec

dionysus_path = os.path.abspath('./Downloads/dionysus/build/bindings/python/')
gudhi_path = os.path.abspath('./Downloads/Gudhi.0/build/cython/')
pipeline_path = os.path.abspath('./Downloads/pyRipser/ICERM_project/')
ripser_path = os.path.abspath('./Downloads/pyRipser/')

sys.path.append(gudhi_path)
sys.path.append(dionysus_path)
sys.path.append(pipeline_path)
sys.path.append(ripser_path)
import DGMTools as dgmt
import pipeline_fns as pf
import TDAPlotting as tdap
import dionysus as dnsys
import gudhi as gudhi
import TDA as tda

# distances = [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1],
#              [1, 0, 1], [1, 0, 1]]
# node_weights = [[1, 10, 1], [1, 11, 1], [1, 10, 1], [1, 5, 1], [1, 10, 1], [1, 10, 1],
#                 [1, 15, 1], [1, 10, 1], [1, 5, 1]]
seq_len = 60
k = 15
distances = [[1, 0, 1] for i in range(seq_len)]
node_weights = [[1, 10 + 5 * np.sin(np.pi / k * n), 1] for n in range(seq_len)]

barcodes = [[] for _ in distances]
birth_matrices = [[] for _ in distances]
birth_graphs = [[] for _ in distances]
orig_graphs = [[] for _ in distances]

lamda = 0.01
filtr_max_dim = 0
for i, (dist, nodes) in enumerate(zip(distances, node_weights)):
    barcodes[i], birth_matrices[i], birth_graphs[i], orig_graphs[i] = \
        pf.get_graph_barcodes(dist, nodes, lamda=lamda, filtr_max_dim=filtr_max_dim)

# fig = plt.figure()
# plt.suptitle('H0, lamda = %0.2f' % lamda)
# for i in range(len(barcodes)):
#     ax = fig.add_subplot(3, 3, i + 1)
#     plt.title('%s, %s' % (distances[i], lamda * np.array(node_weights[i])))
#     tdap.plotDGM(barcodes[i][0])
# plt.tight_layout()
# plt.subplots_adjust(top=0.9)

# run sliding window on the barcodes
d = 3
tau = 1
shift = 1
SW, SW_indices = pf.get_sliding_window(barcodes, d=d, tau=tau, shift=shift)
bottleneck_dist_matrix = pf.get_bottleneck_distance_matrix(barcodes)
pers_distance_matrix = pf.get_persistence_vector_dist_matrix(SW_indices, bottleneck_dist_matrix)

# run ripser on the SW embedding
maxdim = 2
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)

# plot
fig = plt.figure(figsize=(16, 10))

ax1 = fig.add_subplot(1, 3, 1)
plt.title('H0')
tdap.plotDGM(pds[0])
plt.xlim(0, 0.3)
plt.xlim(0, 0.3)

ax2 = fig.add_subplot(1, 3, 2)
plt.title('H1')
tdap.plotDGM(pds[1])
plt.xlim(0, 0.3)
plt.xlim(0, 0.3)

ax3 = fig.add_subplot(1, 3, 3)
plt.title('H2')
tdap.plotDGM(pds[2])
plt.xlim(0, 0.3)
plt.xlim(0, 0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# plt.savefig('/Users/biraj/Desktop/noperiodicity.pdf')





