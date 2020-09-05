from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
import copy

dionysus_path = os.path.abspath('./Downloads/dionysus/build/bindings/python/')
gudhi_path = os.path.abspath('./Downloads/Gudhi.0/build/cython/')
pipeline_path = os.path.abspath('./Downloads/pyRipser/ICERM_project/')
ripser_path = os.path.abspath('./Downloads/pyRipser/')

sys.path.append(gudhi_path)
sys.path.append(dionysus_path)
sys.path.append(pipeline_path)
sys.path.append(ripser_path)
import DGMTools as dgmt
import TDA as tda
import pipeline_fns as pf
import TDAPlotting as tdap
import dionysus as dnsys
import gudhi as gudhi
import spike_to_graphs_fns as stg

import numpy as np
from scipy.spatial.distance import squareform
import xlrd


# plot H0
h = 0
sns.set_style('whitegrid')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('H%s' % str(h), fontsize=30)
tdap.plotDGM(pds[h], sz=80, marker='o', color='r')
plt.ylabel('Death time', fontsize=20)
plt.xlabel('Birth time', fontsize=20)
plt.plot([-4, 150], [-4, 150], linewidth=4, c='k')
ax.grid(False, which='both')
for s in ['top', 'right']:
   ax.spines[s].set_visible(False)

for s in ['bottom','left']:
    ax.spines[s].set_linewidth(5)
    ax.spines[s].set_color('k')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k', labelsize=20)
plt.tight_layout()
plt.savefig('/Users/biraj/Desktop/Presentations/figures/H%s_MIT.png' % (str(h)))




# plot MDS
fig = plt.figure()
sns.set_style('whitegrid')
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title('MDS', fontsize=30)
ax.scatter(x, y, z, s=50, c='r')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k', labelsize=15)
plt.tight_layout()


# plot permuted H0
h = 1
sns.set_style('whitegrid')
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('H%s' % str(h), fontsize=30)
tdap.plotDGM(pds[h], sz=80, marker='o', color='r')
plt.ylabel('Death time', fontsize=20)
plt.xlabel('Birth time', fontsize=20)
plt.plot([-4, 140], [-4, 140], linewidth=4, c='k')
ax.grid(False, which='both')
for s in ['top', 'right']:
   ax.spines[s].set_visible(False)

for s in ['bottom','left']:
    ax.spines[s].set_linewidth(5)
    ax.spines[s].set_color('k')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k', labelsize=20)
plt.tight_layout()
plt.savefig('/Users/biraj/Desktop/Presentations/figures/perm_H%s_MIT.png' % (str(h)))


# plot permuted MDS
fig = plt.figure()
sns.set_style('whitegrid')
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title('MDS', fontsize=30)
ax.scatter(x, y, z, s=50, c='r')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k', labelsize=15)
plt.tight_layout()


