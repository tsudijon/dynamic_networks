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
import TDA as tda
import pipeline_fns as pf
import TDAPlotting as tdap
import dionysus as dnsys
import gudhi as gudhi

# rotating periodic graphs
nPoints = 10
nRotations = 20
noise_mean = 10  # internal weights mean
noise_std = 0.00001  # internal noise std dev
edge_fn = lambda x: np.cos(x) + 1   # + np.abs(np.random.normal(0, 0.2))  # periodic fn for edges
node_fn = lambda x: np.cos(x) + 1  # + np.abs(np.random.normal(0, 0.2))  # periodic fn for nodes
distances, node_weights = pf.rotating_periodic_wave_graph(nPoints, nRotations,
                                                          edge_periodic_fn=edge_fn,
                                                          node_periodic_fn=node_fn,
                                                          noise_mean=noise_mean,
                                                          noise_std=noise_std,
                                                          const_edge_wt=0,
                                                          const_node_wt=0)


lamda = 2  # lambda parameter
filtr_max_dim = 0  # max filtration value
barcodes, birth_matrices, birth_graphs, orig_graphs = \
    pf.vectorize_get_graph_barcodes(distances, node_weights, lamda=lamda,
                                    filtr_max_dim=filtr_max_dim)

# # plot the barcodes to see periodicity
# fig = plt.figure()
# plt.suptitle('H0, lamda = %0.2f, noise_mean=%0.2f, noise_std=%0.2f' % (lamda,
#                                                                        noise_mean, noise_std))
# for i in range(len(barcodes)):
#     ax = fig.add_subplot(5, 5, i + 1)
#     plt.title(i)
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

# run ripser
maxdim = 2
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)


fig = plt.figure(figsize=(16, 10))
plt.suptitle('nNodes = %d; lamda = %0.1f, noise_mu = %0.1f, noise_std = %0.1f' % (nPoints, lamda,
                                                                                  noise_mean,
                                                                                  noise_std))
ax1 = fig.add_subplot(1, 3, 1)
plt.title('H0')
plt.xlim(-1, 8)
plt.ylim(-1, 8)
tdap.plotDGM(pds[0])

ax2 = fig.add_subplot(1, 3, 2)
plt.title('H1')
plt.xlim(-1, 8)
plt.ylim(-1, 8)
tdap.plotDGM(pds[1])

ax3 = fig.add_subplot(1, 3, 3)
plt.title('H2')
plt.xlim(-1, 8)
plt.ylim(-1, 8)
tdap.plotDGM(pds[2])

plt.tight_layout()
plt.subplots_adjust(top=0.9)

# plt.savefig('/Users/biraj/Desktop/noperiodicity.pdf')


