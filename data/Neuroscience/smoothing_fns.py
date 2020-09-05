from __future__ import division
import numpy as np
import os
import sys


def gaussian_wind_fn(mu, sigma, a, x):
    '''Normalized Gaussian'''
    return a * np.exp(-((x - mu)**2) / (2 * sigma**2)) / 


def get_kernel_sum(spike_list, spikes, t_points, win_fun):
    result = np.zeros_like(t_points)
    for spike_time, spike in zip(spike_list, spikes):
        result = result + win_fun(spike_time, spike, t_points)
    return result


spike_matrix = gen_data.counts_from_tcs(asmatrix=True)
spikes = spike_matrix[:, 0]
spike_list = np.arange(0, interval[1], dt)
smoothed_dt = 0.50
smoothed_t_points = np.arange(0, interval[1], new_dt)

spikes = spike_matrix[:, 0]
t_points = smoothed_t_points
win_fun = lambda spike_time, spike, t_points: gaussian_wind_fn(spike_time,
                                                               smoothed_dt,
                                                               spike, t_points)
smoothed_rates = get_kernel_sum(spike_list, spikes, t_points, win_fun)

plt.plot(spike_list, spikes)
plt.plot(smoothed_t_points, smoothed_rates, 'r')