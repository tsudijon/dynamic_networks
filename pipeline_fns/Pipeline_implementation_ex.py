''' July 21, 2017
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


distances_1 = [3, 0, 4, 4, 0, 3]
n_weights_1 = [1, 2, 3, 4]

# graph 2
distances_2 = [1, 5, 3, 4, 40, 3]
n_weights_2 = [1, 5, 4, 3]

lamda = 1
filtr_max_dim = 1
filtr_coeff = 2
barcodes_1, birth_time_matrix1, orig_graph1, birth_graph1 = pf.get_graph_barcodes(distances_1,
                                                                                  n_weights_1,
                                                                                  lamda,
                                                                                  filtr_max_dim,
                                                                                  filtr_coeff)

barcodes_2, birth_time_matrix2, orig_graph2, birth_graph2 = pf.get_graph_barcodes(distances_2,
                                                                                  n_weights_2,
                                                                                  lamda,
                                                                                  filtr_max_dim,
                                                                                  filtr_coeff)

# compute the bottleneck distance
bottleneck = gudhi.bottleneck_distance(barcodes_1[0], barcodes_2[0], 0)
print("Bottleneck dist = %0.2f" % bottleneck)

# plot the diagram
tdap.plotDGM(barcodes_1[0])

# plotting to make sure
fig = plt.figure()
pos = nx.circular_layout(birth_graph)
edge_labels = nx.get_edge_attributes(birth_graph, 'weight')
nx.draw(birth_graph, pos, with_labels=True)
edge_labels = nx.draw_networkx_edge_labels(birth_graph, pos, edge_labels, font_size=20)
