'''August 1, 2017
Functions to turn spikes to graphs'''

from __future__ import division
import os
import sys
import numpy as np
import networkx as nx
from scipy.spatial.distance import squareform
import numpy.linalg as la

ripser_path = os.path.abspath('./Downloads/pyRipser/')
dionysus_path = os.path.abspath('./Downloads/dionysus/build/bindings/python/')
sys.path.append(ripser_path)
sys.path.append(dionysus_path)
import TDA as tda
import dionysus as dnsys
import gudhi


def build_graph_v1(spike_instance):
    node_weights = spike_instance
    distances = []
    for i, spike1 in enumerate(spike_instance):
        for j, spike2 in enumerate(spike_instance[i + 1:]):
            if spike1 * spike2 == 0:
                distances.append(0)
            else:
                distances.append(1)
    return distances, node_weights


def vectorize_build_graph_v1(spike_matrix):
    distance_list = []
    node_weight_list = []
    for spike_instance in spike_matrix:
        tmp_dist, tmp_node = build_graph_v1(spike_instance)
        distance_list.append(tmp_dist)
        node_weight_list.append(tmp_node)
    return distance_list, node_weight_list


def build_graph_correlation(spike_matrix, winsize=2, scale_param=10, scale_pow=-1):
    node_weights = np.zeros_like(spike_matrix[winsize:-winsize]).tolist()
    distance_list = []
    distances = []
    nCells = spike_matrix.shape[1]
    for i in range(winsize, len(spike_matrix[:-winsize])):
        for cell1 in range(nCells):
            for cell2 in range(cell1 + 1, nCells):
                dot_prod = np.dot(spike_matrix.T[cell1][i - winsize:i + winsize + 1],
                                  spike_matrix.T[cell2][i - winsize:i + winsize + 1])
                distances.append(round(float(scale_param * (dot_prod ** scale_pow)), 2))
        distance_list.append(distances)
        distances = []
    return distance_list, node_weights


def build_graph_correlation_v2(spike_matrix, winsize=2, node_param=1, node_pow=1, edge_param=1,
                               edge_pow=-1):
    node_weights = node_param * spike_matrix[winsize:-winsize] ** node_pow
    for i, tmp in enumerate(node_weights):
        node_weights[i] = [round(float(x)) for x in tmp]
    distance_list = []
    distances = []
    nCells = spike_matrix.shape[1]
    for i in range(winsize, len(spike_matrix[:-winsize])):
        for cell1 in range(nCells):
            for cell2 in range(cell1 + 1, nCells):
                dot_prod = np.dot(spike_matrix.T[cell1][i - winsize:i + winsize + 1],
                                  spike_matrix.T[cell2][i - winsize:i + winsize + 1])
                distances.append(round(float(edge_param * (dot_prod ** edge_pow)), 2))
        distance_list.append(distances)
        distances = []
    return distance_list, node_weights





