''' August 1, 2017
Generate fake data and see if you can capture periodicity'''

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
import neurosci_pipeline_fns as pf
import TDAPlotting as tdap
import dionysus as dnsys
import gudhi as gudhi
from data_gen_class import generate
import spike_to_graphs_fns as stg


nPeriod = 10
angle_list = list((np.random.uniform(0, 2 * np.pi, nPeriod)))
[angle_list.extend(angle_list) for _ in range(2)]

angle_list = list(np.random.uniform(0, 2 * np.pi, 50))

dt = 0.1
gen_data = generate(delta_t=dt, angle_list=angle_list)

nCells = 10
nBins = 30
kap = 1
peak_rate = 50
peak_range = [1, 50]

tuning_curves = gen_data.uniform_tuning_curves(nCells, nBins, kap, peak_rate=peak_rate)
# tuning_curves = gen_data.varied_tuning_curves(nCells, nBins, peak_range=peak_range)
spike_matrix = gen_data.counts_from_tcs_gaussian(alpha=0.006, asmatrix=True)

# convert the spikes to a graph
winsize = 2
distance_list, node_weights_list = stg.build_graph_correlation(spike_matrix, winsize=winsize,
                                                               scale_pow=-1, scale_param=100)

# interpolate between graphs
tSteps = 5
interp_distance_list, interp_node_weights_list = pf.interpolate_graphs(distance_list,
                                                                       node_weights_list,
                                                                       tSteps)

# get barcodes for the graphs
lamda = 1
max_filtration = 0
filtr_max_dim = 0
barcodes, birth_matrices, birth_graphs, orig_graphs = \
    pf.vectorize_get_graph_barcodes(interp_distance_list, interp_node_weights_list, lamda=lamda,
                                    filtr_max_dim=filtr_max_dim)

# # check the graphs for periodicity
# pos = nx.circular_layout(birth_graphs[0])
# fig = plt.figure()
# for i, g in enumerate(birth_graphs[:nPeriod * 2]):
#     ax = fig.add_subplot(5, 5, i + 1)
#     plt.title(i)
#     pf.plot_graph(g, pos, node_size=80, node_label_size=5, edge_label_size=5, edge_width=0.5,
#                   style='dashed', ax=ax)
# plt.subplots_adjust()

# check the barcodes for periodicity
fig = plt.figure()
plt.suptitle('H0, lamda = %0.2f' % lamda)
for i in range(len(barcodes[:20])):
    ax = fig.add_subplot(5, 5, i + 1)
    plt.title('%d' % i)
    tdap.plotDGM(barcodes[i][0])
plt.tight_layout()
plt.subplots_adjust(top=0.9)

# run sliding window on the barcodes
d = 3
tau = 2
shift = 1
SW, SW_indices = pf.get_sliding_window(barcodes, d=d, tau=tau, shift=shift)
bottleneck_dist_matrix = pf.get_bottleneck_distance_matrix(barcodes)
pers_distance_matrix = pf.get_persistence_vector_dist_matrix(SW_indices, bottleneck_dist_matrix)

# run ripser on the SW embedding
maxdim = 2
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)

fig = plt.figure(figsize=(16, 10))
plt.suptitle('%d cells, %d period, %d lambda, %d d, %d winsize, %d tau' % (nCells,
                                                                           nPeriod, lamda, d,
                                                                           winsize, tau))

ax2 = fig.add_subplot(2, 2, 3)
plt.title('H1')
tdap.plotDGM(pds[1])

ax3 = fig.add_subplot(2, 2, 4)
plt.title('H2')
tdap.plotDGM(pds[2])

ax4 = fig.add_subplot(2, 2, 1)
plt.title('Angle distribution')
plt.plot(angle_list, 'o--')

ax4 = fig.add_subplot(2, 2, 2)
plt.title('Tuning curves')
gen_data.plot_tuning_curves()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/Users/biraj/Desktop/Dyn_egs/without_nodes/%d_%dd_%dcells_%dperiod.png' % (nn, d,
                                                                                        nCells,
                                                                                        nPeriod))
plt.close()





