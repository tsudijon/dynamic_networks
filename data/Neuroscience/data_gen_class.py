''' September 27 2016
Script to generate fake binned spike data using a fixed response to angle, and then
compute tuning curves from that data using the functions included
'''

from __future__ import division
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sbn
import math
import numpy.random as random
import pickle
from pandas import cut
import sys, time, glob, datetime, os, re

with open(os.path.expanduser("~/path_to_hd_data.txt")) as pathfile:
    datadir = pathfile.readline()
    datadir = datadir.rstrip()


class generate:

    '''Class to generate data of varying types simulating cells and inputs to the head direction
    system'''

    def __init__(self, diffusion_rate=0.3, delta_t=0.25, angle_list=None, tuning_curves=None):
        # Generate angle data if no data given
        if angle_list is not None:
            self.angle_list = angle_list
        else:
            self.angle_data(diffusion_rate, delta_t)

        # Generate tuning curves if none given
        # if given, figures out the angle bins
        if tuning_curves is not None:
            self.tuning_curves = tuning_curves
            self.angle_bins = np.linspace(0, 2 * np.pi, len(list(self.tuning_curves.values())[1]))
        # Define variables to be used in functions within self
        self.delta_t = delta_t
        self.diffusion_rate = diffusion_rate

    def vonmises(self, mu, kappa, nBins):
        x = np.linspace(0, 2 * np.pi, nBins)
        fx = np.exp(kappa * np.cos(x - mu))
        fx = fx / np.sum(fx)
        self.angle_bins = x
        return fx

    def angle_data(self, diffusion_rate, delta_t=0.0256, length=5000):
        '''Generates angle data which initialized at 0 rad and diffuses in a random walk
        along the HD space'''
        pos_times = np.arange(0, length * delta_t, delta_t)
        angle_list = np.zeros([length, ])
        for i in range(len(angle_list) - 1):
            angle_list[i + 1] = (angle_list[i] + np.random.poisson(lam=1) * diffusion_rate *
                                 random.choice([-1, 1]))
            if angle_list[i + 1] < 0:
                angle_list[i + 1] = angle_list[i + 1] + 2 * np.pi
            elif angle_list[i + 1] > (2 * np.pi):
                angle_list[i + 1] = angle_list[i + 1] - 2 * np.pi
        self.angle_list = angle_list

    def varied_tuning_curves(self, num_curves, nBins, angle_space=1, kappa_range=[1, 10],
                             peak_range=[1, 50]):
        '''Produces tuning curves with a varied set of preferences and kappa drawn from a uniform
        distribution. All curves are defined by a von mises curve. These curves are stored within
        the instance to be used when the generate responses function is called.'''
        centers = random.uniform(0, 2 * np.pi * angle_space, num_curves)
        kappas = random.uniform(kappa_range[0], kappa_range[1], num_curves)
        peak_ranges = random.uniform(peak_range[0], peak_range[1], num_curves)
        tuning_curves = {}
        for i in range(num_curves):
            curve = self.vonmises(centers[i], kappas[i], nBins)
            tuning_curves[centers[i]] = curve * (peak_ranges[i] / np.amax(curve)) +\
                random.normal(scale=peak_ranges[i] * 0.005, size=len(curve))
            tuning_curves[centers[i]][tuning_curves[centers[i]] < 0] = 0
        self.tuning_curves = tuning_curves

    def uniform_tuning_curves(self, num_curves, nBins, kappa, angle_space=1, peak_rate=50):
        '''Produces uniform tuning curves that evenly cover the HD space with a given kappa. The
        output is stored within the instance for use in generating cell responses.'''
        centers = np.linspace(0, 2 * np.pi * angle_space - (2 * np.pi / num_curves), num_curves)
        tuning_curves = {}
        for mu in centers:
            curve = self.vonmises(mu, kappa, nBins)
            tuning_curves[mu] = curve * (peak_rate / np.amax(curve)) +\
                random.normal(scale=peak_rate * 0.005, size=len(curve))
            tuning_curves[mu][tuning_curves[mu] < 0] = 0
        self.tuning_curves = tuning_curves

    def plot_tuning_curves(self):
        colors = sbn.color_palette(palette='husl', n_colors=100)
        angle_bins = np.linspace(0, 2 * np.pi, 101)
        color_idxs = cut(sorted(self.tuning_curves.keys()), angle_bins, labels=False)
        # plt.figure()
        for i, mu in enumerate(sorted(self.tuning_curves)):
            if np.isnan(color_idxs[i]):
                color_idx = 0
            else:
                color_idx = int(color_idxs[i])
            plt.plot(self.angle_bins, self.tuning_curves[mu], color=colors[color_idx])

    def counts_from_tcs(self, asmatrix=False):
        ''' Uses the instances defined set of tuning curves to generate spike counts using
        a poisson process model'''
        tc_bins = len(list(self.tuning_curves.values())[0])
        tuning_curves = {x: self.tuning_curves[x] * self.delta_t for x in self.tuning_curves}
        angle_bins = np.linspace(0, 2 * np.pi, tc_bins)
        angles_binned = cut(self.angle_list, angle_bins, labels=False)
        angles_binned[np.isnan(angles_binned)] = len(angle_bins) - 1
        angles_binned = angles_binned[np.isfinite(angles_binned)]
        cell_responses = {}
        for cell in tuning_curves:
            cell_responses[cell] = [np.random.poisson(tuning_curves[cell][x])
                                    for x in angles_binned]
        self.cell_responses = cell_responses
        if asmatrix:
            count_matrix = np.array([cell_responses[x] for x in sorted(cell_responses)]).T
            return count_matrix
        else:
            return cell_responses

    def counts_from_tcs_gaussian(self, noise=None, alpha=None, asmatrix=False):
        ''' Uses the instances defined set of tuning curves to generate spike counts using
        a gaussian process model. Either fix constant noise or noise is proportional (alpha) to the 
        firing rate or no noise'''
        tc_bins = len(list(self.tuning_curves.values())[0])
        tuning_curves = {x: self.tuning_curves[x] * self.delta_t for x in self.tuning_curves}
        angle_bins = np.linspace(0, 2 * np.pi, tc_bins)
        angles_binned = cut(self.angle_list, angle_bins, labels=False)
        angles_binned[np.isnan(angles_binned)] = len(angle_bins) - 1
        angles_binned = angles_binned[np.isfinite(angles_binned)]
        cell_responses = {}
        if noise is not None:
            for cell in tuning_curves:
                cell_responses[cell] = [np.abs(int(np.random.normal(tuning_curves[cell][x], noise)))
                                        for x in angles_binned]
        elif alpha is not None:
            for cell in tuning_curves:
                cell_responses[cell] = [np.abs(int(np.random.normal(tuning_curves[cell][x],
                                            alpha * tuning_curves[cell][x] + 0.0001))) for x in angles_binned]
        # no noise case; I put VERY low baseline noise (0.00001) as the std dev. of the normal distribution
        else:
            for cell in tuning_curves:
                cell_responses[cell] = [np.abs(int(np.random.normal(tuning_curves[cell][x], 0.00001)))
                                    for x in angles_binned]
        self.cell_responses = cell_responses
        if asmatrix:
            count_matrix = np.array([cell_responses[x] for x in sorted(cell_responses)]).T
            return count_matrix
        else:
            return cell_responses

    def counts_from_tcs_nbinm(self, phi=1):
        ''' Uses the instances defined set of tuning curves to generate spike counts using
        a negative binomial model'''
        tc_bins = len(list(self.tuning_curves.values())[0])
        tuning_curves = {x: self.tuning_curves[x] * self.delta_t for x in self.tuning_curves}
        angle_bins = np.linspace(0, 2 * np.pi, tc_bins)
        angles_binned = cut(self.angle_list, angle_bins, labels=False)
        angles_binned[np.isnan(angles_binned)] = len(angle_bins) - 1
        angles_binned = angles_binned[np.isfinite(angles_binned)]
        cell_responses = {}
        for cell in tuning_curves:
            cell_responses[cell] = [np.random.negative_binomial(phi, phi/(tuning_curves[cell][x] + phi))
                                    for x in angles_binned]
        self.cell_responses = cell_responses
        return cell_responses
