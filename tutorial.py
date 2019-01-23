''' Feb 28, 2018
BP: Walkthrough of the updated methodology'''

from __future__ import division
import numpy as np
import sys
import os
import dionysus as dns
import matplotlib.pyplot as plt
shared_scripts_path = os.path.expanduser('~/projects/dynamic_networks/shared_scripts/')
pyRipser_path = os.path.expanduser('~/projects/tda_libraries/pyRipser/')
sys.path.append(shared_scripts_path)
sys.path.append(pyRipser_path)
import graph_fns as gf
import persistence_fns as pf
import sliding_window_fns as sw
import TDA as r
import TDAPlotting as tdap


node_wts = [[1, 2, 3], [3, 4, 5, 6], [1, 2, 5, 4], [1, 5, 2, 3], [3, 1, 8, 6, 5]]
edge_wts = [[5, 4, 2], [4, 5, 5, 6, 6, 1], [4, 5, 5, 6, 6, 1], [2, 5, 3, 9, 8, 1],
            [3, 8, 6, 1, 2, 4, 7, 2, 7, 4]]

# pass thru the phi fn
phi_node_wts, phi_edge_wts = gf.weight_fn(node_wts, edge_wts, lamda=1, phi='softplus')

# get a valid filtration where w(x, y)= max(wv(x), wv(y), we(x, y))
filtration_matrix = list(map(lambda n, e: pf.get_filtration(n, e), phi_node_wts, phi_edge_wts))

# get the rips complexes from the filtration
rips = list(map(pf.get_rips_complex, filtration_matrix))

# get the H0 barcode
hom = list(map(dns.homology_persistence, rips))
dgms = list(map(lambda h, r: dns.init_diagrams(h, r)[0], hom, rips))
barcodes = [[[point.birth, point.death] for point in dgm] for dgm in dgms]

# get bottleneck distance between all the H0 diagrams;
bn_dist_matrix = pf.get_bottleneck_dist_matrix(barcodes)

# run sliding window and find the distance between sliding window vecs
# a) either compute the entire sliding window and then find pairwise bottleneck distance
sw_vecs = sw.sliding_window(barcodes, d=2, tau=2)
sw_dist_matrix = sw.sw_distance_matrix(sw_vecs)

# b) find sliding window indices and then using the bottleneck dist matrix, find dist matrix
# of the sw vectors
sw_vecs_indices = sw.sliding_window(range(len(barcodes)), d=2, tau=2)
sw_dist_matrix1 = sw.sw_distance_matrix(sw_vecs_indices, bn_dist_matrix)

# get H1 from the sliding window distance matrix
PDs = r.doRipsFiltrationDM(sw_dist_matrix, maxHomDim=1, coeff=2)

# plot
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('H1')
tdap.plotDGM(PDs[0])
plt.tight_layout()
plt.show()

