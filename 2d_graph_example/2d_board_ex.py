''' April 18, 2018
BP: Implementing Chris' 2D graphs'''

from __future__ import division
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import squareform
shared_scripts_path = os.path.abspath('shared_scripts/')
pyRipser_path = os.path.expanduser('~/projects/Research/dynamic_networks/pyRipser/')
sys.path.append(shared_scripts_path)
sys.path.append(pyRipser_path)
import graph_fns as gf
import persistence_fns as pf
import sliding_window_fns as sw
import pyRipser.TDA as r
import pyRipser.TDAPlotting as tdap
import dionysus as dns
from sklearn import manifold


def phi_fn(x):
    return np.array([np.exp(-a + 5) for a in x])


# generate the blinking graph
K = 10
nPeriods = 5
Translen = 10
graph = gf.getBlinkingVideo(K, nPeriods, Translen)
node_wts, adjacency_matrix, idx2pos, pos2idx = graph['XRet'], graph['A'], graph['idx2pos'], \
    graph['idx2pos']

edge_wts = [squareform(adjacency_matrix.toarray())] * len(node_wts)


# pass thru the phi fn
phi_node_wts, phi_edge_wts = gf.weight_fn(node_wts, edge_wts, lamda=1, phi=phi_fn)

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

# run sliding window and find the distance between sliding window vec
# find sliding window indices and then using the bottleneck dist matrix, find dist matrix
# of the sw vectors
d = 5
tau = 3
sw_vecs_indices = sw.sliding_window(range(len(barcodes)), d=2, tau=2)
sw_dist_matrix = sw.sw_distance_matrix(sw_vecs_indices, bn_dist_matrix)

# get H1 from the sliding window distance matrix
PDs = r.doRipsFiltrationDM(sw_dist_matrix, maxHomDim=1, coeff=2)

# get mds coords
mds = manifold.MDS(n_components=3, n_init=4, dissimilarity='precomputed', metric=True)
mds_coords = mds.fit_transform(sw_dist_matrix)
x, y, z = np.split(mds_coords, 3, axis=1)

# plot
fig = plt.figure()
plt.suptitle('K=%d, nPeriod=%d, Translen=%d, d=%d, tau=%d' % (K, nPeriods, Translen, d, tau))
ax = fig.add_subplot(221)
plt.title('H0')
tdap.plotDGM(PDs[0])

ax1 = fig.add_subplot(222)
plt.title('H1')
tdap.plotDGM(PDs[1])

ax2 = fig.add_subplot(223)
plt.title('H2')
tdap.plotDGM(PDs[2])

ax3 = fig.add_subplot(224, projection='3d')
plt.title('MDS coordinates')
ax3.scatter(x, y, z, c='r')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/Users/biraj/Desktop/dyn_networks/analyses/2018_04_2d_graph/graph_K%d_nP%d_T%d_d%d_tau%d.png' %
            (K, nPeriods, Translen, d, tau))
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, c='r')


