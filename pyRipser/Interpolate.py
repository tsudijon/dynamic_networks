from TDA import *
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import scipy.io as sio
import scipy.interpolate as interp
import scipy.misc
from scipy import signal
from scipy import interpolate
from sklearn.decomposition import PCA
from CSMSSMTools import *

def getSlidingWindow(x, dim, Tau, dT):
    N = len(x)
    NWindows = int(np.floor((N-dim*Tau)/dT))
    X = np.zeros((NWindows, dim))
    idx = np.arange(len(x))
    xidx = [] #Index the original samples into the sliding window array (assumes dT evenly divides 1)
    for i in range(NWindows):
        if dT*i == int(dT*i):
            xidx.append(i)
        idxx = dT*i + Tau*np.arange(dim)
        start = int(np.floor(idxx[0]))
        end = int(np.ceil(idxx[-1]))
        X[i, :] = interp.spline(idx[start:end+1], x[start:end+1], idxx)
    while len(xidx) < len(x):
        xidx.append(xidx[-1])
    return (X, xidx)

def interpolateSSM(D, NIters, convolve = False):
    """
    Linearly interpolate an SSM
    :param D: An NxN self-similarity matrix (SSM)
    :param NIters: Double the resolution this number of times
    :param returns: DRet: An (N*2^NIters) x (N*2^NIters) interpolated SSM
    """
    DRet = np.array(D)
    for i in range(NIters):
        xx = np.arange(DRet.shape[0]*2)
        x = xx[0::2]
        f = interpolate.interp2d(x, x, DRet, kind = 'linear')
        DRet = f(xx, xx)
        #Convolve along diagonals
        if convolve:
            N = DRet.shape[0]
            for i in range(1, N-1):
                d = np.diagonal(DRet, i)
                dnew = np.convolve(d, [1.0, 1.0], 'same')
                DRet[np.arange(N-i), np.arange(N-i)+i] = dnew
            [I, J] = np.meshgrid(np.arange(N), np.arange(N))
            DRet[I <= J] = 0
            DRet = 0.5*(DRet + DRet.T)
            DRet = DRet[1:-1, 1:-1]
        else:
            np.fill_diagonal(DRet, 0)
    return DRet

if __name__ == '__main__':
    T1 = 4 #The period of the sine in number of samples
    NPeriods = 3 #How many periods to go through
    N = T1*NPeriods #The total number of samples
    t = np.arange(N) #Time indices
    NIters = 1 #Number of doublings to do in interpolated matrix
    t1 = 2*np.pi*(1.0/T1)*t
    x = np.cos(t1)
    (X, xidx) = getSlidingWindow(x, T1, 1, 1)
    print "X.shape = ", X.shape
    D1 = getSSM(X)
    D2 = interpolateSSM(D1, NIters, False)
    print "D2[0, 1] = ", D2[0, 1]
    
    PDs = doRipsFiltrationDM(D1, 1, coeff = 41)
    I1 = PDs[1]
    mp1 = I1[np.argmax(I1[:, 1] - I1[:, 0]), :]
    
    PDs = doRipsFiltrationDM(D2, 1, coeff = 41)
    I2 = PDs[1]
    mp2 = I2[np.argmax(I2[:, 1] - I2[:, 0]), :]
    
    plt.figure(figsize = (12, 10))
    plt.subplot(221)
    plt.imshow(D1, cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
    plt.colorbar()
    plt.subplot(222)
    plt.imshow(D2, cmap = 'afmhot', aspect = 'auto', interpolation = 'nearest')
    plt.title("Interpolated %ix"%(2**NIters))
    plt.colorbar()
    plt.subplot(223)
    plotDGM(I1, 'r')
    plt.hold(True)
    plotDGM(I2, 'b')
    plt.title("mp1 = %.3g, mp2 = %.3g, b2/b1 = %g"%(mp1[1]-mp1[0], mp2[1]-mp2[0], mp2[0]/mp1[0]))
    plt.subplot(224)
    print "mp1[0] = ", mp1[0]
    print "mp2[0] = ", mp2[0]
    print "mp1[1] = ", mp1[1]
    print "mp2[1] = ", mp2[1]
    plt.imshow(np.abs((D2 - mp2[0])) < 1e-2, aspect = 'auto', interpolation = 'nearest')
    plt.title("Interpolated Birth Time Pixels")
    plt.savefig("Interpolated%ix.svg"%(2**NIters), bbox_inches = 'tight')
    
