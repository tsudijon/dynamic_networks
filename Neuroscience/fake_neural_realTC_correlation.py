''' August 1, 2017
Generate fake data from real TCsand see if you can capture periodicity'''

from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn import manifold

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
from general_file_fns import load_file
import copy

with open(os.path.expanduser('~') + '/path_to_hd_data.txt', "r") as myfile:
    data_path = myfile.readlines()[0]
    data_path = data_path.rstrip()
TC_path = data_path + 'analyses/2016_04_tuning_curves/'

session_id = 'Mouse25-140131'
TC_data_path = TC_path + session_id + '_bins_30.p'
TC_data = load_file(TC_data_path)
tuning_curves = TC_data['tuning_curve_data']
# ADn_cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 0)]
ADn_cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)]
nCells = len(ADn_cells)
tuning_curves_ADn = {cell: tuning_curves[cell] for cell in ADn_cells}
# replacing the keys in the tuning curves w/ sth random
for i, key in enumerate(sorted(tuning_curves_ADn.keys())):
    tuning_curves_ADn[i] = tuning_curves_ADn.pop(key)


nPeriod = 10
f = lambda x: 2.7 + 20 * np.sin(2 * np.pi * x / (nPeriod * 2))
angle_list = f(np.arange(0, nPeriod, 0.3)).tolist()
angle_list = np.mod(np.arange(0, 4 * 2 * np.pi, 0.1).tolist(), 2 * np.pi)

# smooth the angle list
# [angle_list.extend(angle_list) for _ in range(1)]

gen_data = generate(angle_list=angle_list, delta_t=1, tuning_curves=tuning_curves_ADn)

# gen_data.plot_tuning_curves()
spike_matrix = gen_data.counts_from_tcs_gaussian(alpha=0.01, asmatrix=True)

# convert the spikes to a graph
winsize = 2
scale_param = 1
scale_pow = 1
distance_list, node_weights_list = stg.build_graph_correlation(spike_matrix, winsize=winsize,
                                                               scale_pow=1, scale_param=1)

# interpolate between graphs
tSteps = 1
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

# permute the barcodes
t = np.random.randint(0, len(barcodes), len(barcodes))
permuted_barcodes = np.array(barcodes)[t]

# run sliding window on the barcodes
d = 63
tau = 1
shift = 1
SW, SW_indices = pf.get_sliding_window(barcodes, d=d, tau=tau, shift=shift)
bottleneck_dist_matrix = pf.get_bottleneck_distance_matrix(barcodes)
pers_distance_matrix = pf.get_persistence_vector_dist_matrix(SW_indices, bottleneck_dist_matrix)

# run mds on the persistence matrix
a = copy.deepcopy(pers_distance_matrix)
# a[a == 0] = 0.001
# a[a == np.inf] = 0
mds = manifold.MDS(n_components=3, n_init=4, dissimilarity='precomputed', metric=True)
pers_coords = mds.fit_transform(a)
x, y, z = np.split(pers_coords, 3, axis=1)


# run ripser on the SW embedding
maxdim = 1
coeff = 2
pds = tda.doRipsFiltrationDM(pers_distance_matrix, maxdim, coeff=coeff)
print(np.max(pds[1][:, 1] - pds[1][:, 0]))

fig = plt.figure(figsize=(16, 10))
plt.suptitle('%d cells, %d period, %d lambda, %d d, %d winsize, %d tau' % (nCells,
                                                                           nPeriod, lamda, d,
                                                                           winsize, tau))

ax2 = fig.add_subplot(2, 2, 3)
plt.title('H1')
tdap.plotDGM(pds[1])

ax3 = fig.add_subplot(2, 2, 4)
plt.title('H0')
tdap.plotDGM(pds[0])

ax4 = fig.add_subplot(2, 2, 1)
plt.title('Impulse')
plt.plot(angle_list, 'o--')

ax4 = fig.add_subplot(2, 2, 2)
plt.title('Tuning curves')
gen_data.plot_tuning_curves()

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/Users/biraj/Desktop/Dyn_egs/without_nodes_realTCs/%dd_%dcells_%dperiod_scale%d_param%d.png' %
            (d, nCells, nPeriod, scale_pow, scale_param))
plt.close()


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title('MDS of persistence barcode vectors')
ax.scatter(x, y, z, s=30)
plt.savefig('/Users/biraj/Desktop/Dyn_egs/without_nodes_realTCs/mds_%dd_%dcells_%dperiod_scale%d_param%d.png' %
            (d, nCells, nPeriod, scale_pow, scale_param))
plt.close()


