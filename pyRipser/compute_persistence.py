from __future__ import division
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import seaborn as sns

pyRipser_path = '/Users/biraj/Downloads/pyRipser/'
os.chdir(pyRipser_path)
import general_fns as gf
import TDA as tda
import TDAPlotting as tdap


# generate points from an n_sphere
nDim = 3
nPoints = 500
n_sphere = gf.samp_nSphere(nDim, nPoints)

# computer persistence homology
maxDim = 2
PDs = tda.doRipsFiltration(n_sphere, maxDim, coeff=2)

# plot
gs = gridspec.GridSpec(4, 4)
fig = plt.figure()
ax = fig.add_subplot(gs[:, :2], projection='3d')
plt.title('%d points from S2' % nPoints)
ax.scatter(n_sphere[:,0], n_sphere[:,1], n_sphere[:,2])
plt.show()

# compute the persistent homology
ax1 = fig.add_subplot(gs[0, 2:4])
plt.title('H0')
tdap.plotDGM(PDs[0])

ax2 = fig.add_subplot(gs[1, 2:4])
plt.title('H1')
tdap.plotDGM(PDs[1])

ax3 = fig.add_subplot(gs[2, 2:4])
plt.title('H2')
tdap.plotDGM(PDs[2])

plt.tight_layout()

