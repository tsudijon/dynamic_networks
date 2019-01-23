''' Sep 8, 2017
BP: Functions to generate different shapes'''

from __future__ import division
import numpy as np
import numpy.linalg as la


def samp_unit_nSphere(n_dim, n_points):
    ''' Sample points from a unit n_Dim-sphere'''
    pts = np.random.randn(n_dim + 1, n_points)
    norm_pts = np.stack(pts / la.norm(pts, axis=0), axis=1)
    return norm_pts


def samp_nTorus(inner_radius, outer_radius, n_points):
    n = int(np.sqrt(n_points))
    [theta, phi] = np.meshgrid(np.linspace(0, 2 * np.pi, n), np.linspace(0, 2 * np.pi, n))
    theta = theta.flatten()
    phi = phi.flatten()
    x = (outer_radius + inner_radius * np.cos(phi)) * np.cos(theta)
    y = (outer_radius + inner_radius * np.cos(phi)) * np.sin(theta)
    z = inner_radius * np.sin(phi)
    X = np.stack((x, y, z), axis=1)
    return X


def filled_S2(radius):
    X = np.linspace(-1 * radius, radius, 20)
    Y = np.linspace(-1 * radius, radius, 20)
    Z = np.linspace(-1 * radius, radius, 20)
    x, y, z = np.meshgrid(X, Y, Z)
    x, y, z = x.flatten(), y.flatten(), z.flatten()
    stacked = np.column_stack((x, y, z))
    return stacked[la.norm(stacked, axis=1) <= radius]





