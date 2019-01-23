''' Sep 13, 2017
BP: Testing out pyRipser'''

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import numpy.linalg as la
with open(os.path.expanduser("~/path_to_dyn_networks.txt")) as pathfile:
    project_dir = pathfile.readline()
    project_dir = project_dir.rstrip()
    pyRipser_path = project_dir + 'pyRipser/'
sys.path.append(pyRipser_path)
import TDA as tda
import TDAPlotting as tdap
import shape_fns as sf

# generate a point cloud
nDim = 1  # 2 sphere
nPoints = 100
nSphere = sf.samp_unit_nSphere(nDim, nPoints)
X = nSphere

# run Ripser
maxDim = 2
PDs = tda.doRipsFiltration(X, maxDim, coeff=2)

# plot the sphere and the 
a, b = np.split(X, 2, axis=1)
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(a, b)

ax = fig.add_subplot(122)
plt.title('H1')
tdap.plotDGM(PDs[1])
plt.tight_layout()
plt.show()
