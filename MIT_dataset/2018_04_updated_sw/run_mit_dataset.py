''' April 3, 2018
BP: Run updated SW embedding on the MIT dataset'''

from __future__ import division
import numpy as np
import sys
import os
import dionysus as dns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
shared_scripts_path = os.path.expanduser('~/projects/dynamic_networks/shared_scripts/')
pyRipser_path = os.path.expanduser('~/projects/tda_libraries/pyRipser/')
sys.path.append(shared_scripts_path)
sys.path.append(pyRipser_path)
from general_file_fns import load_file
import graph_fns as gf
import persistence_fns as pf
import sliding_window_fns as sw
import TDA as r
import TDAPlotting as tdap
from sklearn import manifold

data_dir = os.path.expanduser('~/Desktop/dyn_networks/analyses/')
data = load_file(data_dir + 'binned_data/bin_30mins.p')
edge_wts = data['edge_wts']
node_wts = data['node_wts']


def phi_fn(x):
    return np.array([np.exp(-a + 5) for a in x])


# pass thru the phi fn
mod_node_wts = [np.ones_like(wt) * 1000 for wt in node_wts]  # have to change from 0 due to phi fn
phi_node_wts, phi_edge_wts = gf.weight_fn(mod_node_wts, edge_wts, lamda=1, phi=phi_fn)

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
d = 48
tau = 1
sw_vecs_indices = sw.sliding_window(range(len(barcodes)), d=d, tau=tau)
sw_dist_matrix = sw.sw_distance_matrix(sw_vecs_indices, bn_dist_matrix)

# get H1 from the sliding window distance matrix
PDs = r.doRipsFiltrationDM(sw_dist_matrix, maxHomDim=2, coeff=2)

# get mds coords
mds = manifold.MDS(n_components=3, n_init=4, dissimilarity='precomputed', metric=True)
mds_coords = mds.fit_transform(sw_dist_matrix)
x, y, z = np.split(mds_coords, 3, axis=1)

# plot
fig = plt.figure(figsize=(6, 6))
plt.suptitle('MIT Dataset, bin=30mins, d=%d, tau=%d' % (d, tau))

ax1 = fig.add_subplot(221, projection='3d')
ax1.set_title('MDS coordinates')
ax1.scatter(x, y, z, c='r')

ax = fig.add_subplot(222)
mp = np.max(PDs[0][:, 1] - PDs[0][:, 0])
ax.set_title('H0, mp=%0.2f' % mp)
tdap.plotDGM(PDs[0])

ax2 = fig.add_subplot(223)
mp = np.max(PDs[1][:, 1] - PDs[1][:, 0])
ax2.set_title('H1, mp=%0.2f' % mp)
tdap.plotDGM(PDs[1])

ax3 = fig.add_subplot(224)
mp = np.max(PDs[2][:, 1] - PDs[2][:, 0])
ax3.set_title('H2, mp=%0.2f' % mp)
tdap.plotDGM(PDs[2])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
save_dir = data_dir + '2018_04_mit/'
plt.savefig(save_dir + 'bin_30mins_%dd_%dtau.png' % (d, tau))
plt.show()



