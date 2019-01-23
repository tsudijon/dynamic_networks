''' July 19, 2017
Compare bottleneck distance between similar graphs'''
from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec

pipeline_path = os.path.abspath('./Downloads/pyRipser/ICERM_project/')
ripser_path = os.path.abspath('./Downloads/pyRipser/')
sys.path.append(pipeline_path)
sys.path.append(ripser_path)
import DGMTools as dgmt
import pipeline_fns as pf
import TDAPlotting as tdap


# rotating nGons
# graph 1
distances_1 = [3, 5, 4, 4, 5, 3]
node_weights_1 = [0, 0, 0, 0]

# graph 2
distances_2 = [3, 5, 4, 4, 5, 3]
node_weights_2 = [1, 5, 4, 3]

# filter and compute barcodes
maxDim = 1
coeff = 2
lamda = 1
condition = 'max'  # max or sum of node weight and diameter

# filter graph 1
barcodes_1, birth_time_matrix_1, orig_graph_1, birth_graph_1 = pf.get_graph_barcodes_dist(distances_1,
                                                                                     node_weights_1,
                                                                                     lamda, maxDim,
                                                                                     filtr_coeff=2,
                                                                                     condition=condition)
# filter graph 2
barcodes_2, birth_time_matrix_2, orig_graph_2, birth_graph_2 = pf.get_graph_barcodes_dist(distances_2,
                                                                                     node_weights_2,
                                                                                     lamda, maxDim,
                                                                                     filtr_coeff=2,
                                                                                     condition=condition)


# extract features
large_vecs_1 = dgmt.sortAndGrab(barcodes_1[0], NBars=10)
large_vecs_2 = dgmt.sortAndGrab(barcodes_2[0], NBars=10)

# compute bottle neck distances from 0-d homology barcode
bottleneck_dist = dgt.getBottleneckDist(barcodes_1[0], barcodes_2[0])
print(bottleneck_dist)








# # plot persistent bar codes and the graphs
# fig = plt.figure()
# gs = gridspec.GridSpec(4, 4)

# ax = fig.add_subplot(gs[:2, :2])
# plt.title('original graph')
# pos = nx.circular_layout(orig_graph_1)
# node_labels = {node: (node, 'wt=%d' % weight) for node, weight in zip(orig_graph_1.nodes(),
#                                                                       node_weights_1)}
# edge_labels = nx.get_edge_attributes(orig_graph_1, 'weight')
# nx.draw(orig_graph_1, pos, labels=node_labels)
# edge_labels = nx.draw_networkx_edge_labels(orig_graph_1, pos, edge_labels, font_size=20)

# ax1 = fig.add_subplot(gs[:2, 2:4])
# plt.title('Birth time graph')
# pos = nx.circular_layout(birth_graph_1)
# edge_labels = nx.get_edge_attributes(birth_graph_1, 'weight')
# nx.draw(birth_graph_1, pos, with_labels=True)
# edge_labels = nx.draw_networkx_edge_labels(birth_graph_1, pos, edge_labels, font_size=20)

# ax1 = fig.add_subplot(gs[2:4, :2])
# plt.title('H0')
# tdap.plotDGM(barcodes_1[0])

# ax2 = fig.add_subplot(gs[2:4, 2:4])
# plt.title('H1')
# tdap.plotDGM(barcodes_1[1])

# plt.tight_layout()
