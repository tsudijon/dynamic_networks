''' Compute periodicity of simple time evolving graph'''
from __future__ import division
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import seaborn as sns
import networkx as nx
from sklearn import manifold

pyRipser_path = '/Users/biraj/Downloads/pyRipser/'
os.chdir(pyRipser_path)
import general_fns as gf
import TDA as tda
import TDAPlotting as tdap


# define all the graphs
nVertices = 20
vertices = np.arange(nVertices)
edges = list(zip(vertices[:-1], vertices[1:]))
edges.append((nVertices - 1, 0))

# define the graphs
graphs = []
for i in range(nVertices + 1):
    g = nx.Graph(edges[:i])
    g.add_nodes_from(vertices)
    graphs.append(g)
graphs.extend(list(reversed(graphs)))

# adding the same series twice
for i in range(2):
    graphs.extend(graphs)

# visualizing the graph series
fig = plt.figure()
nx.draw_circular(graphs[-1], node_size=10, with_labels=True)

# compute the sliding window
d = 4
tau = 1
SW_embed = []
for i in range(int(len(graphs) / (d * tau))):
    SW_embed.extend(np.concatenate())


# compute persistent homology
maxDim = 2
PDs = tda.doRipsFiltration(SW_embed, maxDim, coeff=2)

# LLE
lle = manifold.LocallyLinearEmbedding(2, 2, eigen_solver='dense')
lle_coords = lle.fit_transform(SW_embed)
x, y = np.split(lle_coords, 2, axis=1)

# plot
fig = plt.figure()
ax = fig.add_subplot(211)
ax.scatter(x, y)

ax1 = fig.add_subplot(212)
tdap.plotDGM(PDs[0])
