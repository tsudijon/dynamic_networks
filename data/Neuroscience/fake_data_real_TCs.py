''' August 1, 2017
Generate fake data from real TCsand see if you can capture periodicity'''

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
from data_gen_class import generate
import spike_to_graphs_fns as stg
from general_file_fns import load_file

with open(os.path.expanduser('~') + '/path_to_hd_data.txt', "r") as myfile:
    data_path = myfile.readlines()[0]
    data_path = data_path.rstrip()
TC_path = data_path + 'analyses/2016_04_tuning_curves/'

session_id = 'Mouse25-140130'
TC_data_path = TC_path + session_id + '_bins_30.p'
TC_data = load_file(TC_data_path)
tuning_curves = TC_data['tuning_curve_data']
ADn_cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 0)]
tuning_curves_ADn = {cell: tuning_curves[cell] for cell in ADn_cells}
# replacing the keys in the tuning curves w/ sth random
for i, key in enumerate(sorted(tuning_curves_ADn.keys())):
    tuning_curves_ADn[i / 1.2] = tuning_curves_ADn.pop(key)


nPeriod = 10
angle_list = list(np.random.uniform(0, 2 * np.pi, nPeriod))
[angle_list.extend(angle_list) for _ in range(3)]
plt.plot(angle_list[:30])
gen_data = generate(angle_list=angle_list, tuning_curves=tuning_curves)

# gen_data.plot_tuning_curves()
spike_matrix = gen_data.counts_from_tcs_gaussian(alpha=0.0001, asmatrix=True)
distance_list, node_weights_list = stg.vectorize_build_graph_v1(spike_matrix)

# get barcodes for the graphs
lamda = 1
max_filtration = 0
filtr_max_dim = 0
barcodes, birth_matrices, birth_graphs, orig_graphs = \
    pf.vectorize_get_graph_barcodes(distance_list, node_weights_list, lamda=lamda,
                                    filtr_max_dim=filtr_max_dim)

# run sliding window on the barcodes
d = nPeriod
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
plt.suptitle('%d cells, %d period, %d lambda, %d winsize' % (len(tuning_curves_ADn), nPeriod,
                                                             lamda, d))
ax1 = fig.add_subplot(1, 3, 1)
plt.title('H0')
tdap.plotDGM(pds[0])

ax2 = fig.add_subplot(1, 3, 2)
plt.title('H1')
tdap.plotDGM(pds[1])

ax3 = fig.add_subplot(1, 3, 3)
plt.title('H2')
tdap.plotDGM(pds[2])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
