from __future__ import division
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import importlib
import argparse
import scipy.spatial as sp

## Load in sphere code
import tdadynamicnetworks.examples.PlaneExample as plane
import tdadynamicnetworks.examples.ContinuousTimeSensorModel as ctsm

## Load in d.n. analysis code
import tdadynamicnetworks.graph_fns as gf
import tdadynamicnetworks.persistence_fns as pf
import tdadynamicnetworks.sliding_window_fns as sw
from ripser import ripser
from persim import plot_diagrams
import time

import deepdish as dd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument('--do_plot', type=int, default=0, help='Save plots of iterations to disk')
    
    opt = parser.parse_args()
    do_plot = bool(opt.do_plot)
    seed = opt.seed
    path = opt.path

    results = {}

    for manifold in ["plane", "torus"]:
        for lambda1 in [1, 100, 1000]:
            for fac in [1, 5]:
                ## Setup Sensor Lifetimes 
                max_lifetime = 4
                T = 1
                obsfn = lambda t, p: plane.periodic_plane_random_cos_series(t,p,1, seed) # T doesn't matter here
                lambda2 = lambda1

                start = time.time()
                sensor_lifetimes = ctsm.get_sensor_lifetimes(1000, max_lifetime, lambda1, lambda2,
                                                            domain_lengths = (1,1), manifold = 'plane', seed = seed)

                end = time.time()
                print("Sampling Sensor Lifetimes", end - start) 

                ## Create the Dynamic Network
                step_size = 0.05/fac
                ts = np.arange(0,max_lifetime,step_size) 

                start = time.time()
                (node_wts,edges, allpoints) = ctsm.sample_dynamic_geometric_graph(sensor_lifetimes, ts,
                                                                                    obsfn = obsfn, manifold = manifold)
                end = time.time()
                print("Sampling Dynamic Network", end - start) 

                import multiprocessing as mp
                from joblib import Parallel, delayed

                start = time.time()
                filtration_matrix = list(map(lambda n, e: pf.get_filtration(n, e), node_wts, edges))
                end = time.time()
                print("Converting to filtration matrices", end - start) 

                start = time.time()
                num_cores = mp.cpu_count() - 4
                barcodes = Parallel(n_jobs = num_cores)(delayed(pf.get_rips_complex)(filt) for filt in filtration_matrix)
                end = time.time()
                print("Computing barcodes", end - start) 

                start = time.time()
                bn_dist_matrix = pf.get_bottleneck_dist_matrix(barcodes)
                end = time.time()
                perm = np.random.permutation(bn_dist_matrix.shape[0])
                bn_dist_matrix_perm = bn_dist_matrix[perm, :]
                bn_dist_matrix_perm = bn_dist_matrix_perm[:, perm]
                print("Computing bottleneck", end - start) 

                for name, D in zip(["permuted", "normal"], [bn_dist_matrix_perm, bn_dist_matrix]):
                    wl = 2.0*T
                    d = int(wl/step_size)
                    swe = sw.sliding_window(range(len(barcodes)), d=d, tau=1, # Dummy time series?
                                                        max_index = int(6.0*T/step_size) )

                    print("Number of points in SW Embedding:", len(swe))
                    sw_dist_matrix = sw.sw_distance_matrix(swe, D)
                    sw_dist_matrix /= np.max(sw_dist_matrix)
                    print("Doing ripser on {} points...".format(D.shape[0]))
                    start = time.time()
                    PDs = ripser(sw_dist_matrix, distance_matrix=True, maxdim=1, coeff=41)['dgms']
                    print("Finished ripser, elapsed time {:.3f}".format(time.time()-start))
                    # Seed, manifold, lambda1, fac, name
                    results["{}_{}_{}_{}_{}".format(seed, manifold, lambda1, fac, name)] = {"PDs":PDs}

                dd.io.save("{}/{}.h5".format(path, seed), results)

                if do_plot:
                    score = 0
                    I = PDs[1]
                    if I.size > 0:
                        score = np.max(I[:, 1] - I[:, 0])
                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plot_diagrams(PDs)
                    plt.title("$\\lambda = {}$, fac = {}, Score = {:.3f}".format(lambda1, fac, score))
                    plt.subplot(122)
                    plt.imshow(sw_dist_matrix, cmap='magma_r')
                    plt.colorbar()
                    plt.savefig("NetworkDist_{}_{}_{}.png".format(manifold, lambda1, fac))

                    """
                    plt.figure(figsize=(12, 6))
                    for i, t in enumerate(ts):
                        plt.clf()
                        plt.subplot(121)
                        X = allpoints[i]
                        f = node_wts[i]
                        plt.scatter(X[:, 0], X[:, 1], c=f, cmap='magma_r', zorder=10)
                        for e in edges[i]:
                            x1, y1 = X[e[0], :]
                            x2, y2 = X[e[1], :]
                            plt.plot([x1, x2], [y1, y2], c='C0', linewidth=1, linestyle='--')
                        plt.subplot(122)
                        plot_diagrams(barcodes[i])
                        plt.savefig("Network{}.png".format(i))
                    """
        