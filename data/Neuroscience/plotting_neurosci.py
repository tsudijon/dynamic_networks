''' August 1, 2017
Generate fake data from real TCsand see if you can capture periodicity'''

from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec
from sklearn import manifold
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
sns.set_style('white')
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
import neurosci_pipeline_fns as pf
import TDAPlotting as tdap
import dionysus as dnsys
import gudhi as gudhi
from data_gen_class import generate
import spike_to_graphs_fns as stg
from general_file_fns import load_file
import copy

with open(os.path.expanduser('~') + '/path_to_hd_data.txt', "r") as myfile:
    data_path = myfile.readlines()[0]
    data_path = data_path.rstrip()
TC_path = data_path + 'analyses/2016_04_tuning_curves/'

session_id = 'Mouse25-140131'
TC_data_path = TC_path + session_id + '_bins_30.p'
TC_data = load_file(TC_data_path)
tuning_curves = TC_data['tuning_curve_data']
# ADn_cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8), (5, 0)]
ADn_cells = [(4, 0), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (4, 7), (4, 8)]
nCells = len(ADn_cells)
tuning_curves_ADn = {cell: tuning_curves[cell] for cell in ADn_cells}
# replacing the keys in the tuning curves w/ sth random
for i, key in enumerate(sorted(tuning_curves_ADn.keys())):
    tuning_curves_ADn[i] = tuning_curves_ADn.pop(key)


angle_list = np.mod(np.arange(0, 4 * 2 * np.pi, 0.1).tolist(), 2 * np.pi)
epsilon = np.random.uniform(0, 0.4, len(angle_list))
angle_list += epsilon

t = np.arange(0, len(angle_list)) * 0.5 / 30
fig = plt.figure()
ax = plt.subplot(111)
plt.plot(t, angle_list, '--o')
plt.xlabel('Time(s)', fontsize=20)
plt.ylabel('Angle(radians)', fontsize=20)
ax.set_yticks(np.arange(0, 2 * np.pi + 0.2, np.pi / 2))
ax.set_yticklabels([r'$0$', r"$\frac{\pi}{2}$", r'$\pi$', r'$\frac{3 \pi}{2}$', r'$2 \pi$'], 
                   fontsize=20)
ax.grid(False, which='both')
for s in ['top', 'right']:
   ax.spines[s].set_visible(False)

for s in ['bottom','left']:
    ax.spines[s].set_linewidth(5)
    ax.spines[s].set_color('k')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k')
plt.savefig('/Users/biraj/Desktop/Presentations/figures/angle_list.png')


angle_bins = np.linspace(0, 2 * np.pi, 29)
gen_data = generate(angle_list=angle_list, delta_t=1, tuning_curves=tuning_curves_ADn)

fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(tuning_curves_ADn.keys())):
    plt.plot(angle_bins, tuning_curves_ADn[i], lw=4)
plt.xlim(0, 2 * np.pi + 0.2)
plt.ylabel('Neuron activity', fontsize=25)
plt.xlabel('Input (Angle)', fontsize=25)
ax.set_yticklabels([])
ax.set_xticks(np.arange(0, 2 * np.pi + 0.2, np.pi / 2))
ax.set_xticklabels([r'$0$', r"$\frac{\pi}{2}$", r'$\pi$', r'$\frac{3 \pi}{2}$', r'$2 \pi$'], 
                   fontsize=20)
ax.grid(False, which='both')
for s in ['top', 'right']:
   ax.spines[s].set_visible(False)

for s in ['bottom','left']:
    ax.spines[s].set_linewidth(5)
    ax.spines[s].set_color('k')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k')
plt.tight_layout()
plt.savefig('/Users/biraj/Desktop/Presentations/figures/real_tuning_curves.png')





# plot H0
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('H1', fontsize=30)
tdap.plotDGM(pds[1], sz=80, marker='o', color='r')
plt.ylabel('Death time', fontsize=20)
plt.xlabel('Birth time', fontsize=20)
plt.plot([0, 4500], [0, 4500], linewidth=4, c='k')
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
plt.savefig('/Users/biraj/Desktop/Presentations/figures/H1_neuroscience.png')



# plot MDS
fig = plt.figure()
sns.set_style('whitegrid')
ax = fig.add_subplot(1, 1, 1, projection='3d')
plt.title('MDS', fontsize=30)
ax.scatter(x, y, z, s=50, c='r')
ax.tick_params(axis='both', direction='in', length=10, width=3, colors='k', labelsize=15)
plt.tight_layout()




















