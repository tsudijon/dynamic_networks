''' August 1, 2017
Generate fake data and see if you can capture periodicity'''

from __future__ import division
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.gridspec as gridspec
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

nPeriod = 10
angle_list = list((np.random.uniform(0, 2 * np.pi, nPeriod)))
[angle_list.extend(angle_list) for _ in range(2)]

angle_list = list(np.random.uniform(0, 2 * np.pi, 50))

dt = 0.1
gen_data = generate(delta_t=dt, angle_list=angle_list)

nCells = 3
nBins = 30
kap = 1
peak_rate = 50
peak_range = [1, 50]

tuning_curves = gen_data.uniform_tuning_curves(nCells, nBins, kap, peak_rate=peak_rate)
angle_bins = np.linspace(0, 2 * np.pi, 30)
a = gen_data.tuning_curves
k = list(a.keys())
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(angle_bins, a[k[0]], lw=4)
plt.plot(angle_bins, a[k[1]], lw=4)
plt.plot(angle_bins, a[k[2]], lw=4)
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
plt.savefig('/Users/biraj/Desktop/Presentations/figures/multiple_tuning_curves.png')


