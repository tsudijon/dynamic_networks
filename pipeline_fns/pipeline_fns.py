''' July 19, 2017
Writing functions for bottleneck pipeline; to compare graph edit distance to bottleneck distance'''
from __future__ import division
import os
import sys
import numpy as np
import networkx as nx
from scipy.spatial.distance import squareform
import numpy.linalg as la
import copy
with open(os.path.expanduser("~/path_to_dyn_networks.txt")) as pathfile:
    project_dir = pathfile.readline()
    project_dir = project_dir.rstrip()
    pyRipser_path = project_dir + 'pyRipser/'
dionysus_path = os.path.abspath('./Downloads/dionysus/build/bindings/python/')
sys.path.append(pyRipser_path)
sys.path.append(dionysus_path)
import TDA as tda
import dionysus as dnsys
import gudhi


def get_rotating_nGon(nNodes, nRotations):
    ''' Get graph of a rotating nGon rotating nRotations times '''
    graphs = [[]] * nRotations
    adj_matrices = [[]] * nRotations
    nodes = list(range(nNodes))
    for r in range(nRotations):
        rot_node_list = np.roll(nodes, r)
        edge_list = list(zip(rot_node_list[: -1], rot_node_list[1:]))
        graphs[r] = nx.Graph(edge_list)
        adj_matrices[r] = nx.adjacency_matrix(graphs[r]).todense()
    return graphs, adj_matrices


def get_adjacency_matrix(up_trian_distances):
    ''' Symmetrizes upper triangular part of the distance matrix'''
    return squareform(up_trian_distances)


def get_metric_adjacency_matrix(adjacency_matrix):
    ''' Convert a graph to a finite metric space using Dijkstra's algo; if no shortest
    path exists, put a ridiculously high value'''
    g = nx.from_numpy_matrix(adjacency_matrix)
    shortest_path = nx.all_pairs_dijkstra_path_length(g)
    dijkstra_adj_matrix = np.zeros((len(g.nodes()), len(g.nodes())))
    for i, vi in enumerate(g.nodes()):
        for j, vj in enumerate(g.nodes()):
                if vj not in shortest_path[vi]:
                    dijkstra_adj_matrix[i, j] = 1000  # if a shortest path does not exists
                else:
                    dijkstra_adj_matrix[i, j] = shortest_path[vi][vj]
    # is not symmetric due to floating point error #KillMe; symmetrize
    symmetric_matrix = (dijkstra_adj_matrix + dijkstra_adj_matrix.T) / 2
    return symmetric_matrix


def get_metric_adjacency_matrix_v2(adjacency_matrix):
    '''turning it into a finite metric space is not needed; it's
    okay as long as you have a valid filtration'''
    return adjacency_matrix


def add_node_weights(graph, node_weights):
    ''' Easily add node weights to graph'''
    weights = {node: node_weights for node, node_weights in zip(graph.nodes(), node_weights)}
    nx.set_node_attributes(graph, 'node_weight', weights)


def get_max_node_weight_matrix(graph):
    ''' Like the adjacency matrix except each entry is max between two node weights'''
    nWeight_matrix = np.zeros((len(graph.nodes()), len(graph.nodes())))
    for i, vi in enumerate(graph.node.keys()):
        for j, vj in enumerate(graph.node.keys()):
            nWeight_matrix[i, j] = max(graph.node[vi]['node_weight'], graph.node[vj]['node_weight'])
    return nWeight_matrix


def get_birth_time_matrix_max(adjacency_matrix, node_weights, lamda):
    ''' Rips filtration such that the birth time is max of node weight and diam of the simplex;
    the diagonal is the node weight'''
    metric_adjacency_matrix = get_metric_adjacency_matrix_v2(adjacency_matrix)
    orig_graph = nx.from_numpy_matrix(metric_adjacency_matrix)
    add_node_weights(orig_graph, node_weights)
    max_nWeight_matrix = get_max_node_weight_matrix(orig_graph)
    birth_time_matrix = np.maximum(metric_adjacency_matrix, lamda * max_nWeight_matrix)
    np.fill_diagonal(birth_time_matrix, lamda * np.array(node_weights))
    birth_graph = nx.from_numpy_matrix(birth_time_matrix)
    return birth_time_matrix, orig_graph, birth_graph


def get_birth_time_matrix_sum(adjacency_matrix, node_weights, lamda):
    ''' Rips filtration such that the birth time is sum of node weight and diam of the simplex;
    the diagonal is the node weight'''
    metric_adjacency_matrix = get_metric_adjacency_matrix_v2(adjacency_matrix)
    orig_graph = nx.from_numpy_matrix(metric_adjacency_matrix)
    add_node_weights(orig_graph, node_weights)
    max_nWeight_matrix = get_max_node_weight_matrix(orig_graph)
    birth_time_matrix = np.sum((metric_adjacency_matrix, lamda * max_nWeight_matrix), axis=0)
    np.fill_diagonal(birth_time_matrix, lamda * np.array(node_weights))
    birth_graph = nx.from_numpy_matrix(birth_time_matrix)
    return birth_time_matrix, orig_graph, birth_graph


def get_graph_barcodes(distances, node_weights, lamda=1, filtr_max_dim=1,
                       filtr_coeff=2, condition='max'):
    ''' Gte barcodes for node weighted and edge graphs using Dionysus'''
    adjacency_matrix = get_adjacency_matrix(distances)
    if condition == 'max':
        birth_time_matrix, orig_graph, birth_graph = get_birth_time_matrix_max(adjacency_matrix,
                                                                               node_weights, lamda)
    elif condition == 'sum':
        birth_time_matrix, orig_graph, birth_graph = get_birth_time_matrix_sum(adjacency_matrix,
                                                                               node_weights, lamda)
    # filtration using dionysus
    np.fill_diagonal(birth_time_matrix, 0)  # fill diagonal with zeros
    sq_distances = squareform(birth_time_matrix)
    # make max_dim + 1 skeleton for max_dim homology group
    rips = dnsys.fill_rips(sq_distances, filtr_max_dim + 1, 2000)  # rips filtration;
    for i, simplex in enumerate(rips):
        if simplex.dimension() == 0:  # adjust birth time of a node with node wt
            rips[i] = dnsys.Simplex([simplex[0]], lamda * node_weights[i])

    # compute barcodes
    hom = dnsys.homology_persistence(rips)
    dgms = dnsys.init_diagrams(hom, rips)

    barcodes = [np.empty((0, 2), int) for _ in range(filtr_max_dim + 1)]

    # return barcodes
    for i in range(filtr_max_dim + 1):
        for point in dgms[i]:
            barcodes[i] = np.append(barcodes[i], np.array([[point.birth, point.death]]), axis=0)
    np.fill_diagonal(birth_time_matrix, lamda * np.array(node_weights))
    return barcodes, birth_time_matrix, orig_graph, birth_graph, dgms


def vectorize_get_graph_barcodes(distances_list, node_weights_list, lamda=1, filtr_max_dim=1,
                                 filtr_coeff=2, condition='max'):
    ''' version of get_graph_barcodes that accepts a list of distances and node_weights'''
    barcodes = [[] for _ in distances_list]
    birth_matrices = [[] for _ in distances_list]
    birth_graphs = [[] for _ in distances_list]
    orig_graphs = [[] for _ in distances_list]
    dgms = [[] for _ in distances_list]

    for i, (distances, nodes) in enumerate(zip(distances_list, node_weights_list)):
        barcodes[i], birth_matrices[i], birth_graphs[i], orig_graphs[i], dgms[i] = \
            get_graph_barcodes(distances, nodes, lamda=lamda, filtr_max_dim=filtr_max_dim,
                               filtr_coeff=filtr_coeff, condition=condition)
    return barcodes, birth_matrices, birth_graphs, orig_graphs, dgms


def plot_graph(graph, position=None, node_size=50, node_label_size=50, edge_label_size=5,
               style='solid', edge_width=1, ax=None):
    if position is not None:
        pos = position
    else:

        pos = nx.circular_layout(graph)
    labels = {node: (node, 'wt=%0.2f' % wt['node_weight']) for node, wt in graph.node.items()}
    h = nx.draw(graph, pos, labels=labels, node_size=node_size, font_size=node_label_size,
                style=style, width=edge_width, ax=ax)
    edge_labels = nx.get_edge_attributes(graph, 'weight')
    edge_labels = nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=edge_label_size)
    return h


def sine_fn(points):
    return np.sin(points) + 1


def rotating_periodic_wave_graph(nPoints, nRotations, edge_periodic_fn=sine_fn,
                                 node_periodic_fn=sine_fn, noise_mean=0.1, noise_std=0.01,
                                 const_edge_wt=0, const_node_wt=0):
    ''' Create an nGon whose edge weights and node weights are periodic functions'''
    # if you want edges to be constant
    if const_edge_wt > 0:
        t_points_edge = [const_edge_wt] * (nPoints - 1)
    elif const_edge_wt == 0:
        t_points_edge = np.linspace(0, 2 * np.pi, nPoints - 1)

    # if you want nodes to be constant
    if const_node_wt > 0:
        t_points_node = [const_node_wt] * (nPoints)
    elif const_node_wt == 0:
        t_points_node = np.linspace(0, 2 * np.pi, nPoints)

    distance_matrix = [[] for i in range(nRotations)]
    node_weight_matrix = [[] for i in range(nRotations)]

    for i in range(nRotations):
        node_weights = np.array(list(map(node_periodic_fn, np.roll(t_points_node, i)))) + 0.0001
        edge_weights = np.array(list(map(edge_periodic_fn, np.roll(t_points_edge, i + 2)))) + 0.0001
        sq_distances = np.diagflat(edge_weights, 1)
        distances = []
        [distances.extend(sq_distances[i, i + 1:]) for i in range(nPoints)]

        # replace the internal connections by noise
        for j, dist in enumerate(distances):
            if dist == 0:
                distances[j] = np.abs(np.random.normal(noise_mean, noise_std))  # no -ve weights

        # enter into the final matrix
        distance_matrix[i] = distances
        node_weight_matrix[i] = node_weights
    return distance_matrix, node_weight_matrix


def get_sliding_window(vectors, d, tau, shift=1):
    SW_embedding = []
    SW_embedding_indices = []
    for i in np.arange(0, len(vectors) - (d * tau), shift):
        embedding = []
        embedding_indices = []
        for j in range(d + 1):
            embedding.append(vectors[i + j * tau])
            embedding_indices.append(i + j * tau)
        SW_embedding.append(embedding)
        SW_embedding_indices.append(embedding_indices)
    return SW_embedding, SW_embedding_indices


def dist_persistence_vectors(vec1, vec2):
    dist = [gudhi.bottleneck_distance(vec1[i], vec2[i], 0.1) for i in range(len(vec1))]
    return la.norm(dist)


def get_bottleneck_distance_matrix(barcode_vectors):
    ''' Get a distance matrix between barcode diagrams'''
    distance_matrix = []
    for i in range(len(barcode_vectors)):
        for j in range(i + 1, len(barcode_vectors)):
            bottle_dist = gudhi.bottleneck_distance(barcode_vectors[i][0], barcode_vectors[j][0],
                                                    0)
            distance_matrix.append(bottle_dist)
    return squareform(distance_matrix)


def get_persistence_vector_dist_matrix(SW_indices, bottle_dist_matrix):
    distance_matrix = []
    for i, pt1_idx in enumerate(SW_indices):
        for j, pt2_idx in enumerate(SW_indices[i + 1:]):
            dist = [bottle_dist_matrix[k][l] for (k, l) in zip(pt1_idx, pt2_idx)]
            pers_dist = la.norm(dist)
            distance_matrix.append(pers_dist)
    return squareform(distance_matrix)


def interpolate_graphs(distance_list, node_weights_list, nTsteps):
    ''' Interpolate between graphs using straight line homotopy; add nTsteps -1 interpolated
    graphs between two graphs'''
    interp = lambda x, y, t: list(t * np.array(y) + (1 - t) * np.array(x))
    interp_dist_list = copy.deepcopy(distance_list)
    interp_node_weight_list = copy.deepcopy(node_weights_list)
    for i, (gx, gy) in enumerate(zip(distance_list[:-1], distance_list[1:])):
        tmp_distance = []
        tmp_nodes = []
        for t_step in np.arange(1 / nTsteps, 1, 1 / nTsteps):
            tmp_distance.append([round(float(x), 2) if np.isfinite(x) else np.inf for x in
                                interp(gx, gy, t_step)])
            tmp_nodes.append([round(float(y), 2) if np.isfinite(y) else np.inf for y in
                             interp(node_weights_list[i], node_weights_list[i + 1], t_step)])
        interp_dist_list[i * nTsteps + 1: i * nTsteps + 1] = tmp_distance
        interp_node_weight_list[i * nTsteps + 1:i * nTsteps + 1] = tmp_nodes
    return interp_dist_list, interp_node_weight_list









